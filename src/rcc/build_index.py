"""Build the frozen corpus index that every system shares.

Run once per (dataset, chunking config). The index is the experiment's control:
every system queries this exact chunk set, so the comparison is about retrieval
strategy rather than chunking or embedding choices. `CorpusIndex.load` refuses
to open an index whose manifest hash does not match the settings being asked
for, which is what stops a silently mismatched index from invalidating a run.

Cost and time: token counts are computed once per chunk with
`messages.count_tokens`. That endpoint is free, but it is one HTTP call per
chunk. Measured on MuSiQue dev: 21,100 unique documents -> ~7,400 chunks, a
few minutes at 8 workers.

For a SLICE, use `--for-queries N --seed S` with the same N and seed as the
run. Slicing by `--max-docs` instead takes the first N documents while the run
samples queries from across the file, so the gold paragraphs would be missing
from the index and every system would score ~0 for reasons unrelated to
retrieval.

Usage:
    python -m rcc.build_index --dataset musique \\
        --path data/raw/musique_ans_v1.0_dev.jsonl \\
        --out data/corpus/musique --model claude-sonnet-5
    python -m rcc.build_index --config configs/corpus.yaml --max-docs 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

from rcc.datasets import REGISTRY as DATASETS
from rcc.datasets.base import Document
from rcc.index import CorpusIndex


def _counted(docs: Iterator[Document], limit: int | None) -> Iterator[Document]:
    """Yield documents, optionally truncated, reporting progress as it goes."""
    for i, doc in enumerate(docs):
        if limit is not None and i >= limit:
            return
        if i and i % 1000 == 0:
            print(f"  {i} documents...", flush=True)
        yield doc


def build(
    dataset: str,
    dataset_path: str | Path,
    out: str | Path,
    *,
    model: str,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    max_docs: int | None = None,
    for_queries: int | None = None,
    seed: int = 0,
) -> Path:
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}; have {sorted(DATASETS)}")

    loader = DATASETS[dataset](dataset_path)
    out_path = Path(out)

    print(f"dataset   {dataset}  <- {dataset_path}")
    print(f"chunking  {chunk_size} words, {overlap} overlap")
    print(f"embedding {embedding_model}")
    print(f"tokens    counted once per chunk with {model}")
    if max_docs:
        print(f"LIMIT     {max_docs} documents (slice, not the full corpus)")
    print()

    qids: set[str] | None = None
    if for_queries:
        # Build the corpus from exactly the sampled queries' documents. Slicing
        # by document order instead would leave the gold paragraphs out of the
        # index and every system would score ~0 for reasons that have nothing
        # to do with retrieval.
        sampled = loader.sample(for_queries, seed)
        qids = {q.qid for q in sampled}
        print(f"QUERY SLICE  {len(qids)} queries (seed {seed}); corpus = their documents")

    started = time.perf_counter()
    documents = list(_counted(loader.documents(qids), max_docs))
    print(f"{len(documents)} unique documents; chunking and counting tokens...")

    index = CorpusIndex.build(
        documents,
        out_path,
        chunk_size=chunk_size,
        overlap=overlap,
        embedding_model=embedding_model,
        token_count_model=model,
    )

    manifest = json.loads((out_path / "manifest.json").read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - started
    print(
        f"\nbuilt {manifest['n_chunks']} chunks in {elapsed:.0f}s -> {out_path}"
        f"\nconfig_hash {manifest['config_hash'][:16]}..."
    )
    if max_docs:
        print(
            "\nThis is a SLICE. Any accuracy number from it is not comparable "
            "to published results -- the corpus is missing documents."
        )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen corpus index.")
    parser.add_argument("--config", type=Path, help="configs/corpus.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--path", help="raw dataset file")
    parser.add_argument("--out", help="index output directory")
    parser.add_argument("--model", help="model whose tokenizer counts chunk tokens")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument(
        "--max-docs",
        type=int,
        help="build from only the first N documents (a slice, for pipeline checks)",
    )
    parser.add_argument(
        "--for-queries",
        type=int,
        help="build the corpus from the documents of N sampled queries "
             "(use the SAME n_queries and seed as the run)",
    )
    parser.add_argument("--seed", type=int, default=0, help="query sample seed")
    args = parser.parse_args(argv)

    settings: dict[str, Any] = {
        "chunk_size": 256,
        "overlap": 32,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "model": "claude-sonnet-5",
    }
    if args.config:
        import yaml

        raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        settings.update({k: v for k, v in raw.items() if v is not None})

    # CLI flags win over the config file.
    for key, value in (
        ("dataset", args.dataset),
        ("dataset_path", args.path),
        ("out", args.out),
        ("model", args.model),
        ("chunk_size", args.chunk_size),
        ("overlap", args.overlap),
        ("embedding_model", args.embedding_model),
    ):
        if value is not None:
            settings[key] = value

    missing = [k for k in ("dataset", "dataset_path", "out") if not settings.get(k)]
    if missing:
        parser.error(f"missing required setting(s): {', '.join(missing)}")

    build(
        settings["dataset"],
        settings["dataset_path"],
        settings["out"],
        model=settings["model"],
        chunk_size=int(settings["chunk_size"]),
        overlap=int(settings["overlap"]),
        embedding_model=settings["embedding_model"],
        max_docs=args.max_docs,
        for_queries=args.for_queries,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
