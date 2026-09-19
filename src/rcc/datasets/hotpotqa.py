"""Loader for the HotpotQA distractor development set."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from .base import DatasetLoader, Document, Query, dedupe_documents


def slug(title: str) -> str:
    """Convert a title into a stable identifier."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


class HotpotQA(DatasetLoader):
    """Load the HotpotQA distractor development JSON array."""

    name = "hotpotqa"
    _expected_filename = "hotpot_dev_distractor_v1.json"
    _download_url = (
        "https://raw.githubusercontent.com/hotpotqa/hotpot/master/"
        "dataset/hotpot_dev_distractor_v1.json"
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
        with self.path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array in {self.path}")
        self._records = records

    @property
    def _items(self) -> list[dict[str, Any]]:
        if self._records is None:
            self._load()
        assert self._records is not None
        return self._records

    @staticmethod
    def _require(record: dict[str, Any], record_id: str) -> None:
        required = ("_id", "question", "answer", "context", "supporting_facts")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(
                f"Record {record_id!r} is missing required key(s): "
                f"{', '.join(missing)}"
            )

    def documents(self, qids: set[str] | None = None) -> Iterator[Document]:
        def generate() -> Iterator[Document]:
            for record in self._items:
                if not isinstance(record, dict):
                    raise ValueError("Record '<unknown>' must be an object")
                record_id = str(record.get("_id", "<unknown>"))
                self._require(record, record_id)
                if qids is not None and record_id not in qids:
                    continue
                for context_item in record["context"]:
                    try:
                        title, sentences = context_item
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Record {record_id!r} has an invalid context entry"
                        ) from exc
                    yield Document(
                        doc_id=slug(str(title)),
                        title=str(title),
                        text="".join(str(sentence) for sentence in sentences),
                    )

        return dedupe_documents(generate())

    def queries(self) -> list[Query]:
        queries: list[Query] = []
        for record in self._items:
            if not isinstance(record, dict):
                raise ValueError("Record '<unknown>' must be an object")
            record_id = str(record.get("_id", "<unknown>"))
            self._require(record, record_id)
            queries.append(
                Query(
                    qid=str(record["_id"]),
                    question=str(record["question"]),
                    gold=[str(record["answer"])],
                    dataset=self.name,
                )
            )
        return queries
