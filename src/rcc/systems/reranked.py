"""Naive RAG plus a cross-encoder reranker. The honest baseline.

This is the system most teams should actually be comparing against, and the one
most "agentic RAG beats RAG" claims quietly skip. It costs exactly the same
number of LLM calls as naive (one), adds only local cross-encoder compute, and
recovers most of the accuracy lost to a bi-encoder ranking the right chunk at
position 7 instead of 3.

If this lands close to the agentic systems on accuracy, the cost curve is the
whole story -- which is precisely the result worth publishing.
"""

from __future__ import annotations

from rcc.meter import QueryScope
from rcc.systems.base import ANSWER_INSTRUCTION, Answer, RagSystem


class RerankedRag(RagSystem):
    name = "reranked"

    #: how many candidates the bi-encoder proposes before cross-encoder rescoring
    candidates: int = 50

    def run(self, question: str, scope: QueryScope) -> Answer:
        scope.tool("semantic_search")
        candidates = self.index.semantic_search(question, self.candidates)

        scope.tool("rerank")
        hits = self.index.rerank(question, candidates, self.k)

        # Identical k and identical presentation to naive: the ONLY difference
        # between these two systems is which k chunks get chosen.
        context = self.present([h.chunk for h in hits], scope)

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
