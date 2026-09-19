"""The contract every RAG system implements, and the shared substrate.

Fairness is the whole point of this file. Every system gets the same chunks,
the same BM25 table, the same embeddings, the same generator model, and -- this
is the one people forget -- the same answer-format instruction. If A-RAG got a
better "answer concisely" line than naive RAG, the benchmark would be measuring
prompts, not architectures.

So: `ANSWER_INSTRUCTION` is defined once here and every system must use it
verbatim. A system that needs extra instructions adds them for its *tools*,
never for how the final answer is formatted.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from rcc.meter import Meter, QueryScope

# --------------------------------------------------------------------------
# Shared prompt fragments -- identical across every system
# --------------------------------------------------------------------------

ANSWER_INSTRUCTION = (
    "Answer the question using only the provided context. "
    "Give the shortest span that answers it -- a name, date, number or noun "
    "phrase. No sentence, no explanation, no restating the question. "
    "If the context does not contain the answer, answer exactly: unanswerable\n"
    "Wrap the final answer in <answer></answer> tags, like <answer>1974</answer>."
)

UNANSWERABLE = "unanswerable"

# Systems wrap their final answer so extraction is unambiguous across all of
# them. Applied identically everywhere, so it cannot advantage one system.
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------
# Corpus substrate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One unit of the frozen corpus.

    `tokens` is counted once at index build time with messages.count_tokens and
    carried here, so retrieval accounting never costs a round trip.
    """

    chunk_id: str
    doc_id: str
    title: str
    text: str
    tokens: int


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    snippet: str  # the matched window, for search results that show less than all


class Index(Protocol):
    """The shared retrieval substrate. Built once, reused by every system.

    Systems may only reach the corpus through this. That is what keeps the
    comparison about retrieval *strategy* rather than chunking or embedding
    choices.
    """

    def keyword_search(self, query: str, k: int) -> list[Hit]:
        """Exact lexical match (BM25), scored by term frequency and length."""
        ...

    def semantic_search(self, query: str, k: int) -> list[Hit]:
        """Dense cosine similarity over sentence embeddings."""
        ...

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        """Cross-encoder rescoring of an existing candidate list."""
        ...

    def get_chunk(self, chunk_id: str) -> Chunk:
        """Full text of one chunk."""
        ...


# --------------------------------------------------------------------------
# System contract
# --------------------------------------------------------------------------


@dataclass
class Answer:
    text: str
    terminated: str = "answered"  # answered | max_steps | context_limit | refusal | error


class RagSystem(ABC):
    """Base class for every architecture under comparison.

    Subclasses implement `run`. They reach the model only through
    `self.generate(...)` and the corpus only through `self.index`, which is how
    the meter can guarantee complete accounting.
    """

    #: stable identifier used in every emitted record
    name: str = "unnamed"

    def __init__(
        self,
        index: Index,
        meter: Meter,
        *,
        model: str,
        max_tokens: int = 4096,
        effort: str = "high",
        thinking: dict[str, Any] | None = None,
        auto_cache: bool = True,
        k: int = 5,
    ) -> None:
        self.index = index
        self.meter = meter
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        # Run-wide knobs. Identical across systems by construction -- the
        # runner builds every system from one config block.
        self.auto_cache = auto_cache
        self.k = k

    # -- subclass API ------------------------------------------------------

    @abstractmethod
    def run(self, question: str, scope: QueryScope) -> Answer:
        """Answer `question`, recording tool use on `scope`."""
        raise NotImplementedError

    # -- helpers shared by every subclass ----------------------------------

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        terminal: bool = True,
    ) -> Any:
        """Every model call goes through here, so every call is metered."""
        return self.meter.generate(
            model=self.model,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens or self.max_tokens,
            effort=self.effort,
            thinking=self.thinking,
            auto_cache=self.auto_cache,
            terminal=terminal,
        )

    def present(self, chunks: Sequence[Chunk], scope: QueryScope) -> str:
        """Render chunks into prompt text AND bill them.

        Subclasses must use this rather than formatting chunks themselves --
        it is the single place where `note_retrieval` is guaranteed to fire, so
        corpus text cannot enter a prompt uncounted.
        """
        parts = []
        for c in chunks:
            self.meter.note_retrieval(c.chunk_id, c.tokens)
            parts.append(f"[{c.chunk_id}] {c.title}\n{c.text}")
        return "\n\n".join(parts)

    def present_hits(self, hits: Sequence[Hit], scope: QueryScope) -> str:
        """Render search results (snippets, not full chunks) and bill them.

        Snippets are billed at their own length, not the parent chunk's -- that
        asymmetry is exactly what the hierarchical-interface claim rests on, so
        it has to be measured honestly.
        """
        parts = []
        for h in hits:
            snippet_tokens = max(1, round(h.chunk.tokens * len(h.snippet) / max(1, len(h.chunk.text))))
            self.meter.note_retrieval(f"{h.chunk.chunk_id}::snippet", snippet_tokens)
            parts.append(f"[{h.chunk.chunk_id}] {h.chunk.title}\n{h.snippet}")
        return "\n\n".join(parts)

    @staticmethod
    def extract(message: Any) -> str:
        """Pull the final answer out of a response, identically for all systems."""
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        m = _ANSWER_TAG.search(text)
        if m:
            return m.group(1).strip()
        # No tag: take the last non-empty line. Short-answer prompts rarely
        # need this, but a silent empty string would look like a retrieval miss.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def tool_uses(message: Any) -> list[Any]:
        return [b for b in message.content if b.type == "tool_use"]

    @staticmethod
    def tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        }

    @staticmethod
    def dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
