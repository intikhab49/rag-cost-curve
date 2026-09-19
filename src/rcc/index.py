"""Frozen corpus index shared by every system, so benchmarks compare retrieval
strategy rather than differing chunking choices.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .systems.base import Chunk, Hit

SNIPPET_CHARS = 320
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _config_hash(chunk_size: int, overlap: int, embedding_model: str) -> str:
    payload = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _snippet_window(text: str, query_terms: set[str]) -> str:
    if not text:
        return ""
    if len(text) <= SNIPPET_CHARS:
        return text
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return text[:SNIPPET_CHARS].rstrip() + "…"
    scores: list[tuple[int, int, int]] = []
    radius = 80
    for i, match in enumerate(matches):
        left = max(0, i - radius // 8)
        right = min(len(matches), i + radius // 8 + 1)
        density = sum(
            1 for candidate in matches[left:right]
            if candidate.group(0).lower() in query_terms
        )
        scores.append((density, -i, i))
    _, _, center = max(scores)
    # Reserve room for the ellipses, or the returned snippet exceeds
    # SNIPPET_CHARS. Snippets are billed by length, so the cap has to be a real
    # ceiling rather than an approximate one.
    body = SNIPPET_CHARS - 2
    start = max(0, matches[center].start() - body // 2)
    end = min(len(text), start + body)
    if end - start < body:
        start = max(0, end - body)
    result = text[start:end].strip()
    if start > 0:
        result = "…" + result
    if end < len(text):
        result += "…"
    return result


def _semantic_snippet(text: str, query_terms: set[str]) -> str:
    if not text:
        return ""
    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        return text[:SNIPPET_CHARS]
    scores = [
        sum(1 for token in _words(sentence) if token in query_terms)
        for sentence in sentences
    ]
    best = max(range(len(sentences)), key=lambda index: (scores[index], -index))
    first = max(0, best - 1)
    last = min(len(sentences), best + 2)
    result = " ".join(sentences[first:last])
    if len(result) <= SNIPPET_CHARS:
        return result
    return _snippet_window(result, query_terms)


class CorpusIndex:
    """A persisted, immutable retrieval index."""

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
        embedding_model: str | None = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"index manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_chunk_size = int(self.manifest["chunk_size"])
        actual_overlap = int(self.manifest["overlap"])
        actual_embedding_model = str(self.manifest["embedding_model"])
        requested_chunk_size = (
            actual_chunk_size if chunk_size is None else chunk_size
        )
        requested_overlap = actual_overlap if overlap is None else overlap
        requested_embedding_model = (
            actual_embedding_model
            if embedding_model is None
            else embedding_model
        )
        expected_hash = _config_hash(
            requested_chunk_size, requested_overlap, requested_embedding_model
        )
        if self.manifest.get("config_hash") != expected_hash:
            raise ValueError(
                "index configuration hash mismatch: requested chunking or "
                "embedding settings do not match the frozen index"
            )

        self.chunk_size = actual_chunk_size
        self.overlap = actual_overlap
        self.embedding_model = actual_embedding_model
        self.reranker_model = reranker_model
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        chunks_path = self.path / "chunks.jsonl"
        with chunks_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    item = json.loads(line)
                    chunk = Chunk(**item)
                    self.chunks.append(chunk)
                    self._by_id[chunk.chunk_id] = chunk
        self.embeddings = np.load(self.path / "embeddings.npy", allow_pickle=False)
        if len(self.chunks) != int(self.manifest["n_chunks"]):
            raise ValueError("index chunk count does not match its manifest")
        if self.embeddings.shape[0] != len(self.chunks):
            raise ValueError("embedding rows do not match chunks.jsonl")
        self._counters = [Counter(_words(chunk.text)) for chunk in self.chunks]
        self._lengths = np.asarray(
            [sum(counter.values()) for counter in self._counters], dtype=np.float32
        )
        # Precomputed document frequencies. Recomputing these per query is an
        # O(corpus) scan that would land inside the measured latency of every
        # system that uses keyword search, making it look slower than it is.
        self._df: Counter[str] = Counter()
        for counter in self._counters:
            self._df.update(counter.keys())

        self._embedding_encoder: Any = None
        self._reranker: Any = None

    @classmethod
    def build(
        cls,
        documents: Iterable[Any],
        path: str | Path,
        *,
        chunk_size: int = 256,
        overlap: int = 32,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        # Token counts must come from the same model the run generates with,
        # or retrieved_tokens is measured against a different tokenizer than
        # the one being billed. The runner passes the run's model here.
        token_count_model: str = "claude-opus-5",
        anthropic_client: Any = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> "CorpusIndex":
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        if anthropic_client is None:
            import anthropic

            anthropic_client = anthropic.Anthropic()

        records: list[tuple[str, str, str]] = []
        for document in documents:
            if isinstance(document, Mapping):
                doc_id = str(document["doc_id"])
                title = str(document.get("title", ""))
                text = str(document["text"])
            elif hasattr(document, "doc_id"):
                # A Document dataclass. getattr(..., default) would evaluate the
                # tuple fallback eagerly and raise on a non-subscriptable object,
                # so the branch has to come before the indexing, not inside it.
                doc_id = str(document.doc_id)
                title = str(getattr(document, "title", ""))
                text = str(document.text)
            else:
                doc_id, title, text = (str(document[0]), str(document[1]), str(document[2]))
            records.append((doc_id, title, text))

        chunks: list[Chunk] = []
        for doc_id, title, text in records:
            token_words = text.split()
            step = chunk_size - overlap
            for start in range(0, len(token_words), step):
                words = token_words[start : start + chunk_size]
                if not words:
                    break
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::{len([c for c in chunks if c.doc_id == doc_id])}",
                        doc_id=doc_id,
                        title=title,
                        text=" ".join(words),
                        tokens=0,
                    )
                )
                if start + chunk_size >= len(token_words):
                    break

        def count_tokens(chunk: Chunk) -> int:
            result = anthropic_client.messages.count_tokens(
                model=token_count_model,
                messages=[{"role": "user", "content": chunk.text}],
            )
            return int(result.input_tokens)

        with ThreadPoolExecutor(max_workers=8) as executor:
            token_counts = list(executor.map(count_tokens, chunks))
        chunks = [
            Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=chunk.text,
                tokens=tokens,
            )
            for chunk, tokens in zip(chunks, token_counts)
        ]

        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(embedding_model)
        matrix = np.asarray(
            encoder.encode(
                [chunk.text for chunk in chunks],
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "chunks.jsonl").open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")
        np.save(target / "embeddings.npy", matrix)
        manifest = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embedding_model": embedding_model,
            "token_count_model": token_count_model,
            "n_chunks": len(chunks),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": _config_hash(chunk_size, overlap, embedding_model),
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return cls(target, reranker_model=reranker_model)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
        embedding_model: str | None = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> "CorpusIndex":
        return cls(
            path,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )

    def keyword_search(self, query: str, k: int) -> list[Hit]:
        if k <= 0:
            return []
        query_terms = _words(query)
        if not query_terms:
            return []
        qcounter = Counter(query_terms)
        n = len(self.chunks)
        average_length = float(self._lengths.mean()) if n else 1.0
        document_frequency = self._df
        scored: list[tuple[float, int]] = []
        for index, counter in enumerate(self._counters):
            length = float(self._lengths[index])
            score = 0.0
            for term, query_frequency in qcounter.items():
                frequency = counter.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(
                    1.0
                    + (n - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1.0 - 0.75 + 0.75 * length / average_length
                )
                score += idf * (frequency * 2.5 / denominator) * query_frequency
            if score:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            Hit(
                chunk=self.chunks[index],
                score=score,
                snippet=_snippet_window(self.chunks[index].text, set(query_terms)),
            )
            for score, index in scored[:k]
        ]

    def semantic_search(self, query: str, k: int) -> list[Hit]:
        if k <= 0 or not self.chunks:
            return []
        if self._embedding_encoder is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_encoder = SentenceTransformer(self.embedding_model)
        vector = np.asarray(
            self._embedding_encoder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=False
            )[0],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        scores = self.embeddings @ vector
        indices = np.argsort(-scores, kind="stable")[:k]
        terms = set(_words(query))
        return [
            Hit(
                chunk=self.chunks[int(index)],
                score=float(scores[int(index)]),
                snippet=_semantic_snippet(self.chunks[int(index)].text, terms),
            )
            for index in indices
        ]

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        if k <= 0 or not hits:
            return []
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model)
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self._reranker.predict(pairs)
        ranked = sorted(
            zip(scores, hits), key=lambda item: float(item[0]), reverse=True
        )
        return [
            Hit(chunk=hit.chunk, score=float(score), snippet=hit.snippet)
            for score, hit in ranked[:k]
        ]

    def get_chunk(self, chunk_id: str) -> Chunk:
        try:
            return self._by_id[chunk_id]
        except KeyError:
            raise KeyError(chunk_id) from None
