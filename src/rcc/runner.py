"""Run a sweep: every system, over the same queries, under one frozen config.

This module is the fairness guarantee. Every system is constructed from ONE
config block and handed the SAME index object, so none of them can quietly get
a different k, model, effort, or caching setting. If you need a per-system knob,
it belongs in that system's class as a documented deviation -- not here.

Two properties worth stating plainly, because they are what make a published
number defensible:

* The run is resumable. Every (system, qid) already present in raw.jsonl is
  skipped, so a crash at query 150 of 200 does not cost the first 150 again.
* The config is copied into the run directory before any spending starts. A
  result whose settings you cannot reconstruct is not a result.

Usage:
    python -m rcc.runner configs/run.yaml
    python -m rcc.runner configs/run.yaml --dry-run    # cost estimate, no calls
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rcc.datasets import REGISTRY as DATASETS
from rcc.index import CorpusIndex
from rcc.meter import Meter, load_pricing
from rcc.score import score_run
from rcc.systems import REGISTRY as SYSTEMS


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    """The frozen settings for one sweep. Everything measurable lives here."""

    run_id: str
    dataset: str
    dataset_path: str
    index_path: str
    systems: list[str]
    model: str

    n_queries: int = 200
    seed: int = 0

    # Shared retrieval + generation settings. Identical for every system by
    # construction -- this is the point of the whole file.
    k: int = 5
    effort: str = "high"
    max_tokens: int = 4096
    auto_cache: bool = True
    thinking: dict[str, Any] | None = None

    runs_dir: str = "runs"
    pricing_path: str | None = "configs/pricing.yaml"

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            # A typo'd key silently falling back to a default would change the
            # experiment without changing the recorded config.
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def validate(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}; have {sorted(DATASETS)}")
        bad = [s for s in self.systems if s not in SYSTEMS]
        if bad:
            raise ValueError(f"unknown systems {bad}; have {sorted(SYSTEMS)}")
        if not self.systems:
            raise ValueError("no systems selected")


# --------------------------------------------------------------------------
# Resume support
# --------------------------------------------------------------------------


def completed_pairs(run_dir: Path) -> set[tuple[str, str]]:
    """(system, qid) pairs already recorded, so a resumed run skips them.

    Records that terminated in `error` are NOT counted as complete. Those are
    usually transient -- a dropped connection, a DNS blip -- and treating them
    as done would bake a network hiccup permanently into the accuracy figure.
    A resume retries them; the later record supersedes the earlier one.
    """
    path = run_dir / "raw.jsonl"
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line from a hard kill; it will be re-run
        if rec.get("terminated") == "error":
            continue
        done.add((rec["system"], rec["qid"]))
    return done


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


@dataclass
class Progress:
    done: int = 0
    total: int = 0
    usd: float = 0.0
    errors: int = 0
    by_system: dict[str, int] = field(default_factory=dict)


def run_sweep(config: RunConfig, config_path: Path, *, dry_run: bool = False) -> Progress:
    config.validate()

    run_dir = Path(config.runs_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy the config BEFORE spending anything. A result you cannot reconstruct
    # the settings for is not a result.
    shutil.copy2(config_path, run_dir / "run.yaml")

    loader = DATASETS[config.dataset](config.dataset_path)
    queries = loader.sample(config.n_queries, config.seed)

    if dry_run:
        print(f"dataset   {config.dataset}  ({len(queries)} queries after sampling)")
        print(f"systems   {', '.join(config.systems)}")
        print(f"model     {config.model}  effort={config.effort}  cache={config.auto_cache}")
        print(f"work      {len(queries) * len(config.systems)} (system, query) pairs")
        print("\nNo calls made. Cost depends on corpus and loop depth; run a")
        print("10-query slice first and multiply -- an estimate from chunk sizes")
        print("alone will be wrong for the agentic systems.")
        return Progress(total=len(queries) * len(config.systems))

    index = CorpusIndex.load(
        config.index_path,
        chunk_size=None,
        overlap=None,
        embedding_model=None,
    )

    already = completed_pairs(run_dir)
    progress = Progress(total=len(queries) * len(config.systems))
    progress.done = len(already)

    meter = Meter(
        config.run_id,
        run_dir,
        pricing=load_pricing(config.pricing_path),
    )

    try:
        for system_name in config.systems:
            # One config block, every system. No per-system overrides here.
            system = SYSTEMS[system_name](
                index,
                meter,
                model=config.model,
                max_tokens=config.max_tokens,
                effort=config.effort,
                thinking=config.thinking,
                auto_cache=config.auto_cache,
                k=config.k,
            )

            for query in queries:
                if (system_name, query.qid) in already:
                    continue

                with meter.query(
                    system_name,
                    config.dataset,
                    query.qid,
                    query.question,
                    list(query.gold),
                ) as scope:
                    answer = system.run(query.question, scope)
                    scope.prediction = answer.text
                    # Do not clobber a terminal state the meter already set
                    # from stop_reason (refusal / context_limit).
                    if scope.terminated == "answered":
                        scope.terminated = answer.terminated

                progress.done += 1
                progress.usd += sum(c.usd for c in scope.calls)
                progress.by_system[system_name] = progress.by_system.get(system_name, 0) + 1
                if scope.terminated == "error":
                    progress.errors += 1

                if progress.done % 10 == 0:
                    print(
                        f"  {progress.done}/{progress.total}  "
                        f"${progress.usd:.3f}  errors={progress.errors}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        # Partial results stay on disk and a rerun resumes from here.
        print("\ninterrupted -- partial results kept; rerun to resume", file=sys.stderr)
    finally:
        meter.close()

    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": config.run_id,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "n_queries": len(queries),
                "systems": config.systems,
                "model": config.model,
                "auto_cache": config.auto_cache,
                "k": config.k,
                "effort": config.effort,
                "usd_this_session": round(progress.usd, 6),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    correct = score_run(run_dir)
    print(f"\nscored: {correct}")
    print(f"spent this session: ${progress.usd:.4f}")
    return progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a RAG cost/accuracy sweep.")
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and make no API calls",
    )
    args = parser.parse_args(argv)

    config = RunConfig.load(args.config)
    run_sweep(config, args.config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
