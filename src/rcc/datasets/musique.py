"""Loader for the MuSiQue answerable development JSONL set."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

from .base import DatasetLoader, Document, Query, dedupe_documents


def slug(title: str) -> str:
    """Convert a title into a stable identifier."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


class MuSiQue(DatasetLoader):
    """Load MuSiQue answerable records from a JSON Lines file."""

    name = "musique"
    _expected_filename = "musique_ans_v1.0_dev.jsonl"
    _download_url = (
        "https://raw.githubusercontent.com/StonyBrookNLP/"
        "MuSiQue/main/data/musique_ans_v1.0_dev.jsonl"
    )

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._records: list[dict[str, Any]] | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Expected {self._expected_filename} at {self.path}; "
                f"download it from {self._download_url}"
            )
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record '<line {line_number}>' must be an object"
                    )
                record_id = str(record.get("id", f"<line {line_number}>"))
                self._require(record, record_id)
                records.append(record)
        self._records = records

    @property
    def _items(self) -> list[dict[str, Any]]:
        if self._records is None:
            self._load()
        assert self._records is not None
        return self._records

    @staticmethod
    def _require(record: dict[str, Any], record_id: str) -> None:
        required = (
            "id",
            "question",
            "answer",
            "answer_aliases",
            "answerable",
            "paragraphs",
        )
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(
                f"Record {record_id!r} is missing required key(s): "
                f"{', '.join(missing)}"
            )
        for paragraph in record["paragraphs"]:
            if not isinstance(paragraph, dict):
                raise ValueError(
                    f"Record {record_id!r} contains an invalid paragraph"
                )
            paragraph_missing = [
                key
                for key in ("idx", "title", "paragraph_text", "is_supporting")
                if key not in paragraph
            ]
            if paragraph_missing:
                raise ValueError(
                    f"Record {record_id!r} paragraph is missing required key(s): "
                    f"{', '.join(paragraph_missing)}"
                )

    def documents(self, qids: set[str] | None = None) -> Iterator[Document]:
        def generate() -> Iterator[Document]:
            for record in self._items:
                if not record["answerable"]:
                    continue
                if qids is not None and str(record["id"]) not in qids:
                    continue
                for paragraph in record["paragraphs"]:
                    title = str(paragraph["title"])
                    text = str(paragraph["paragraph_text"])
                    # doc_id is derived from CONTENT, not from `idx`. MuSiQue's
                    # idx is the paragraph's position within one question's
                    # 20-paragraph set, so it is not stable across records:
                    # keying on (title, idx) both collides distinct paragraphs
                    # that happen to share a slot and duplicates identical
                    # paragraphs that sit at different slots. Either one
                    # corrupts every retrieval metric computed over the corpus.
                    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
                    yield Document(
                        doc_id=f"{slug(title)}::{digest}",
                        title=title,
                        text=text,
                    )

        return dedupe_documents(generate())

    def queries(self) -> list[Query]:
        queries: list[Query] = []
        for record in self._items:
            if not record["answerable"]:
                continue
            gold: list[str] = []
            for answer in [record["answer"], *record["answer_aliases"]]:
                value = str(answer)
                if value and value not in gold:
                    gold.append(value)
            queries.append(
                Query(
                    qid=str(record["id"]),
                    question=str(record["question"]),
                    gold=gold,
                    dataset=self.name,
                )
            )
        return queries
