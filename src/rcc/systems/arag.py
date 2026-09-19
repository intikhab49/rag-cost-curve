"""A-RAG: agentic retrieval over hierarchical interfaces (arXiv:2602.03442).

Three tools exposed directly to the model -- keyword search, semantic search,
chunk read -- driven by a ReAct loop, with a context tracker that refuses to
re-serve a chunk the agent has already read.

The hierarchy is the load-bearing part, and it is why this is reimplemented
here rather than taken on faith: search returns cheap snippets, and only an
explicit `chunk_read` pays for full text. The paper's own ablation (a variant
with just the embedding tool) burned ~10x the retrieved tokens for lower
accuracy, so if snippets were billed at full chunk size here the claim could
not be tested at all. `present_hits` bills snippets at snippet length;
`present` bills reads at chunk length.

What the paper does not report -- and what this implementation exists to
measure -- is what the loop costs in total: LLM calls, resent history, and
latency, none of which appear in its "retrieved tokens" table.
"""

from __future__ import annotations

from typing import Any

from rcc.meter import QueryScope
from rcc.systems.base import ANSWER_INSTRUCTION, Answer, RagSystem

SYSTEM = (
    "You are answering a question that usually requires combining facts from "
    "several documents. You have three tools over a fixed corpus:\n"
    "- keyword_search: exact lexical match. Best for rare names, titles, "
    "numbers and any term that must appear verbatim.\n"
    "- semantic_search: meaning-based match. Best when you know what you are "
    "looking for but not how the corpus words it.\n"
    "- chunk_read: the full text of chunks you have already located. Search "
    "returns only short snippets, so read a chunk when its snippet looks like "
    "it holds the answer or the next link in the chain.\n\n"
    "Work one step at a time: search, look at what came back, then decide "
    "whether to search again with a better query or read a promising chunk. "
    "Multi-hop questions usually need you to find one fact in order to know "
    "what to search for next.\n\n" + ANSWER_INSTRUCTION
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "keyword_search",
        "description": (
            "Exact lexical search over the corpus. Returns short snippets with "
            "their chunk ids. Use for verbatim terms: proper nouns, titles, "
            "dates, numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Terms to match exactly."},
                "k": {"type": "integer", "description": "How many results (1-10)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "semantic_search",
        "description": (
            "Meaning-based search over the corpus. Returns short snippets with "
            "their chunk ids. Use when the corpus may word the fact differently "
            "than the question does."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you are looking for."},
                "k": {"type": "integer", "description": "How many results (1-10)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "chunk_read",
        "description": (
            "Read the full text of chunks by id. Ids come from search results. "
            "A chunk already read is not served twice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chunk ids from earlier search results.",
                }
            },
            "required": ["chunk_ids"],
            "additionalProperties": False,
        },
    },
]


class ARag(RagSystem):
    name = "arag"

    #: hard ceiling on loop iterations. Hitting it is recorded as max_steps, not
    #: silently treated as a wrong answer -- running out of budget and being
    #: wrong are different failures and the report separates them.
    max_steps: int = 12

    #: cap on results per search, regardless of what the model asks for
    max_k: int = 10

    def run(self, question: str, scope: QueryScope) -> Answer:
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        already_read: set[str] = set()

        for _ in range(self.max_steps):
            msg = self.generate(messages, system=SYSTEM, tools=TOOLS)
            calls = self.tool_uses(msg)

            if not calls:
                return Answer(text=self.extract(msg), terminated=scope.terminated)

            messages.append({"role": "assistant", "content": msg.content})

            # All results for one assistant turn go back in ONE user message.
            # Splitting them teaches the model to stop calling tools in
            # parallel, which would quietly change the thing being measured.
            results = [
                self.tool_result(call.id, self._dispatch(call, scope, already_read))
                for call in calls
            ]
            messages.append({"role": "user", "content": results})

        scope.terminated = "max_steps"
        # Give it one forced chance to answer from what it gathered, so a
        # budget-exhausted run is scored on its retrieval rather than on an
        # empty string.
        messages.append(
            {
                "role": "user",
                "content": "Step budget reached. Answer now from what you have.",
            }
        )
        final = self.generate(messages, system=SYSTEM)
        return Answer(text=self.extract(final), terminated="max_steps")

    # -- tool dispatch -----------------------------------------------------

    def _dispatch(self, call: Any, scope: QueryScope, already_read: set[str]) -> str:
        name = call.name
        args = call.input if isinstance(call.input, dict) else {}
        scope.tool(name)

        if name == "keyword_search":
            k = self._clamp_k(args.get("k"))
            hits = self.index.keyword_search(str(args.get("query", "")), k)
            return self.present_hits(hits, scope) or "No matches."

        if name == "semantic_search":
            k = self._clamp_k(args.get("k"))
            hits = self.index.semantic_search(str(args.get("query", "")), k)
            return self.present_hits(hits, scope) or "No matches."

        if name == "chunk_read":
            return self._read(args.get("chunk_ids") or [], scope, already_read)

        return f"Unknown tool: {name}"

    def _read(
        self, chunk_ids: Any, scope: QueryScope, already_read: set[str]
    ) -> str:
        if not isinstance(chunk_ids, list):
            return "chunk_ids must be a list of ids."

        parts: list[str] = []
        for raw in chunk_ids[: self.max_k]:
            chunk_id = str(raw)
            if chunk_id in already_read:
                # The context tracker. Not billed again -- this is the
                # mechanism the paper credits for its token efficiency, so it
                # has to actually withhold the text, not just say so.
                parts.append(f"[{chunk_id}] already read earlier in this session.")
                continue
            try:
                chunk = self.index.get_chunk(chunk_id)
            except KeyError:
                parts.append(f"[{chunk_id}] no such chunk.")
                continue
            already_read.add(chunk_id)
            parts.append(self.present([chunk], scope))

        return "\n\n".join(parts) if parts else "Nothing to read."

    def _clamp_k(self, value: Any) -> int:
        try:
            k = int(value)
        except (TypeError, ValueError):
            return self.k
        return max(1, min(self.max_k, k))
