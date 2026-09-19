"""Naive RAG: embed the question once, take top-k, generate. The control.

This is the floor the other three have to beat, and it is deliberately the
textbook implementation -- one embedding call, one retrieval, one generation,
no query rewriting and no second look. Its structural limit is the thing the
whole benchmark is about: the second hop of a multi-hop question is not present
in the question, so a single embedding of the question cannot find it.
"""

from __future__ import annotations

from rcc.meter import QueryScope
from rcc.systems.base import ANSWER_INSTRUCTION, Answer, RagSystem


class NaiveRag(RagSystem):
    name = "naive"

    def run(self, question: str, scope: QueryScope) -> Answer:
        scope.tool("semantic_search")
        hits = self.index.semantic_search(question, self.k)

        # Full chunks, not snippets: a single-shot system has no second chance
        # to open anything, so giving it snippets would handicap it unfairly.
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
