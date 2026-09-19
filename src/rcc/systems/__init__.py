"""System registry.

The runner builds every system from one shared config block, which is what
guarantees they get identical k, model, effort and caching settings. Adding a
system means adding it here; there is no other wiring.
"""

from __future__ import annotations

from rcc.systems.arag import ARag
from rcc.systems.base import Answer, Chunk, Hit, Index, RagSystem
from rcc.systems.iterative import IterativeRag
from rcc.systems.naive import NaiveRag
from rcc.systems.reranked import RerankedRag

REGISTRY: dict[str, type[RagSystem]] = {
    NaiveRag.name: NaiveRag,
    RerankedRag.name: RerankedRag,
    IterativeRag.name: IterativeRag,
    ARag.name: ARag,
}

__all__ = [
    "REGISTRY",
    "RagSystem",
    "Answer",
    "Chunk",
    "Hit",
    "Index",
    "NaiveRag",
    "RerankedRag",
    "IterativeRag",
    "ARag",
]
