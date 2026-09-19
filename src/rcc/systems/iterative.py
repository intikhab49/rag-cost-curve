"""Iterative RAG: a fixed, code-controlled retrieve -> read -> retrieve workflow.

This is the middle baseline, and it exists to isolate one variable. It can do
multiple hops, like the agent can -- but the *number* of hops is fixed in code
and the model never chooses a strategy. It only writes the next query.

So the comparison against A-RAG answers a specific question: how much of the
agentic gain comes from being able to retrieve more than once, and how much
comes from the model actually steering retrieval? If iterative closes most of
the gap, the ReAct loop is mostly ceremony.
"""

from __future__ import annotations

from rcc.meter import QueryScope
from rcc.systems.base import ANSWER_INSTRUCTION, Answer, Chunk, RagSystem

FOLLOWUP_INSTRUCTION = (
    "You are researching a question that usually needs facts from more than one "
    "document. Given the question and what has been retrieved so far, write ONE "
    "search query that would find a fact you are still missing. "
    "Write only the query text -- no explanation, no quotes. "
    "If nothing further is needed, write exactly: DONE"
)


class IterativeRag(RagSystem):
    name = "iterative"

    #: retrieval rounds, including the first. Fixed in code, not chosen.
    rounds: int = 3

    def run(self, question: str, scope: QueryScope) -> Answer:
        collected: dict[str, Chunk] = {}
        query = question

        for round_index in range(self.rounds):
            scope.tool("semantic_search")
            for hit in self.index.semantic_search(query, self.k):
                collected.setdefault(hit.chunk.chunk_id, hit.chunk)

            if round_index == self.rounds - 1:
                break

            # Ask for the next query. This generation sees only titles and
            # snippets, not full chunks -- billing the full corpus text once per
            # planning step would be a strawman, and no real system does it.
            seen = "\n".join(
                f"[{c.chunk_id}] {c.title}: {c.text[:200]}" for c in collected.values()
            )
            for chunk in collected.values():
                # Bill the preview at its real size, not the parent chunk's.
                preview_tokens = max(1, round(chunk.tokens * 200 / max(1, len(chunk.text))))
                self.meter.note_retrieval(f"{chunk.chunk_id}::preview", preview_tokens)

            plan = self.generate(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Retrieved so far:\n{seen}\n\nNext search query:"
                        ),
                    }
                ],
                system=FOLLOWUP_INSTRUCTION,
                # Generous: a reasoning model spends most of this on thinking
                # tokens before it emits the query. At 128 it hit the cap on
                # 120/200 queries and returned nothing, silently collapsing the
                # multi-hop loop into a single round.
                max_tokens=2048,
                terminal=False,
            )
            next_query = "".join(
                b.text for b in plan.content if b.type == "text"
            ).strip()
            if not next_query or next_query.upper().startswith("DONE"):
                break
            query = next_query

        context = self.present(list(collected.values()), scope)
        msg = self.generate(
            [
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
            system=ANSWER_INSTRUCTION,
        )
        return Answer(text=self.extract(msg), terminated=scope.terminated)
