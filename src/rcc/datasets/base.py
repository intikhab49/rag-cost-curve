"""Shared data structures and interfaces for RAG benchmark datasets."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    qid: str
    question: str
    gold: list[str]
    dataset: str


class DatasetLoader(ABC):
    """Abstract interface for a benchmark dataset loader."""

    name: str

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @abstractmethod
    def documents(self, qids: set[str] | None = None) -> Iterator[Document]:
        """Yield documents, optionally only those belonging to `qids`.

        Filtering matters for slice builds: these benchmarks ship each question
        with its own paragraph set, so a corpus built from the FIRST n documents
        will not contain the gold paragraphs for a randomly sampled query set,
        and every system would score ~0 for reasons unrelated to retrieval.
        """

    @abstractmethod
    def queries(self) -> list[Query]:
        """Return benchmark queries."""

    def sample(self, n: int, seed: int) -> list[Query]:
        """Return a deterministic sample of queries ordered by query ID."""
        queries = sorted(self.queries(), key=lambda query: query.qid)
        if n >= len(queries):
            return queries
        return random.Random(seed).sample(queries, n)


def dedupe_documents(docs: Iterable[Document]) -> Iterator[Document]:
    """Yield only the first document encountered for each document ID."""
    seen: set[str] = set()
    for document in docs:
        if document.doc_id not in seen:
            seen.add(document.doc_id)
            yield document
