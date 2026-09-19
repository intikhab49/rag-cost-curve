"""Dataset registry.

`runner.py` resolves `dataset:` in run.yaml through this map, so adding a
benchmark means adding one entry here and nothing else.
"""

from __future__ import annotations

from rcc.datasets.base import DatasetLoader, Document, Query, dedupe_documents
from rcc.datasets.hotpotqa import HotpotQA
from rcc.datasets.musique import MuSiQue
from rcc.datasets.twowiki import TwoWikiMultihopQA

REGISTRY: dict[str, type[DatasetLoader]] = {
    HotpotQA.name: HotpotQA,
    MuSiQue.name: MuSiQue,
    TwoWikiMultihopQA.name: TwoWikiMultihopQA,
}

__all__ = [
    "REGISTRY",
    "DatasetLoader",
    "Document",
    "Query",
    "dedupe_documents",
    "HotpotQA",
    "MuSiQue",
    "TwoWikiMultihopQA",
]
