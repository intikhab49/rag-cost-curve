"""Aggregate a run into the headline table, the chart, and aggregate.json.

The metric this repo exists for is `usd_per_correct` -- total spend divided by
the number of questions actually answered right. Accuracy alone says agentic
retrieval wins; cost per correct answer says whether it is worth it, and that
is the number nobody published.

Three measurement rules are enforced here rather than left to the reader:

* Latency excludes cache-served queries. A replayed run has latency 0 and would
  otherwise make every system look instant.
* Accuracy carries a bootstrap confidence interval, and every system is
  compared to the baseline with a PAIRED bootstrap over the shared query set.
  With n=200 a 3-point gap is frequently noise; shipping a claim without an
  interval is how benchmarks get corrected in public.
* Errored and budget-exhausted queries stay in the denominator. They cost money
  and failed to answer.

Usage:
    python -m rcc.report runs/2026-09-19_musique_sonnet5
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

# Fixed categorical order from the validated palette. Assigned by system
# identity and never cycled, so a system keeps its colour across every chart in
# the repo even when a run omits one.
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]
SERIES_ORDER = ["naive", "reranked", "iterative", "arag"]

SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"
INK_LIGHT = ("#0b0b0b", "#52514e")
INK_DARK = ("#ffffff", "#c3c2b7")

BOOTSTRAP_SAMPLES = 2000
BASELINE = "naive"


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


@dataclass
class SystemStats:
    system: str
    n: int
    em: float
    f1: float
    em_ci_low: float
    em_ci_high: float
    usd_total: float
    usd_per_query: float
    usd_per_correct: float | None
    llm_calls_mean: float
    input_tokens_mean: float
    cache_read_tokens_mean: float
    cache_write_tokens_mean: float
    output_tokens_mean: float
    retrieved_unique_mean: float
    retrieved_billed_mean: float
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    latency_n: int
    terminated: dict[str, int]
    tool_calls_mean: dict[str, float]


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: Sequence[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return int(ordered[idx])


def load_run(run_dir: Path) -> tuple[list[dict], list[dict], dict]:
    def jsonl(name: str) -> list[dict]:
        path = run_dir / name
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn final line from a hard kill
        return out

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    # raw.jsonl is append-only and a resumed run retries errored queries, so a
    # (system, qid) can appear more than once. The LAST record wins -- it is the
    # successful retry -- and counting both would inflate n and double-count
    # spend.
    latest: dict[tuple[str, str], dict] = {}
    for rec in jsonl("raw.jsonl"):
        latest[(rec["system"], rec["qid"])] = rec
    return list(latest.values()), jsonl("calls.jsonl"), meta


def _cached_qids(calls: list[dict]) -> set[tuple[str, str]]:
    """(system, qid) pairs whose every call was served from the response cache.

    Their wall-clock latency measures the disk, not the system.
    """
    seen: dict[tuple[str, str], bool] = {}
    for call in calls:
        key = (call["system"], call["qid"])
        seen[key] = seen.get(key, True) and bool(call.get("cached_response"))
    return {key for key, all_cached in seen.items() if all_cached}


def _bootstrap_ci(values: Sequence[float], rng: random.Random) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        means.append(_mean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return lo, hi


def paired_win_rate(
    by_system: dict[str, dict[str, int]], challenger: str, baseline: str, rng: random.Random
) -> tuple[float, int] | None:
    """P(challenger beats baseline) by paired bootstrap over shared queries.

    Paired, because both systems answered the SAME questions -- treating them as
    independent samples throws away the pairing and widens the interval for no
    reason.
    """
    if challenger not in by_system or baseline not in by_system:
        return None
    shared = sorted(set(by_system[challenger]) & set(by_system[baseline]))
    if not shared:
        return None

    diffs = [by_system[challenger][q] - by_system[baseline][q] for q in shared]
    n = len(diffs)
    wins = 0
    for _ in range(BOOTSTRAP_SAMPLES):
        total = sum(diffs[rng.randrange(n)] for _ in range(n))
        wins += total > 0
    return wins / BOOTSTRAP_SAMPLES, n


def reprice(
    calls: list[dict], model: str, pricing_path: str | Path | None
) -> dict[tuple[str, str], float]:
    """Recompute per-query spend at another model's rates.

    A run on a free model records real token buckets and real call counts; only
    the dollar figure is zero. Cost is a linear function of those buckets, so
    applying published rates gives a defensible estimate of what the same
    workload costs on a paid model.

    This is an ESTIMATE, and the report labels it as one. Two reasons it is not
    a substitute for running on the target model: the token counts come from
    the run model's tokenizer, and cache behaviour differs between providers.
    It bounds the answer; it does not settle it.
    """
    from rcc.meter import CallRecord, Meter, load_pricing

    prices = load_pricing(pricing_path)
    if model not in prices:
        raise SystemExit(f"no pricing for {model!r}; add it to {pricing_path}")

    meter = Meter.__new__(Meter)  # pricing math only; no I/O, no client
    meter.pricing = prices

    out: dict[tuple[str, str], float] = {}
    for call in calls:
        rec = CallRecord(
            run_id="", system=call["system"], dataset="", qid=call["qid"], step=0,
            model=model,
            input_tokens=call.get("input_tokens", 0),
            cache_creation_input_tokens=call.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=call.get("cache_read_input_tokens", 0),
            output_tokens=call.get("output_tokens", 0),
            retrieved_tokens_in_prompt=0, latency_ms=0, stop_reason="",
            cached_response=False, usd=0.0,
        )
        key = (call["system"], call["qid"])
        out[key] = out.get(key, 0.0) + meter.price_call(rec)
    return out


def aggregate(
    run_dir: Path,
    seed: int = 0,
    price_as: str | None = None,
    pricing_path: str | Path | None = "configs/pricing.yaml",
) -> dict[str, Any]:
    records, calls, meta = load_run(run_dir)
    if not records:
        raise SystemExit(f"no records in {run_dir / 'raw.jsonl'}")

    unscored = [r for r in records if r.get("em") is None]
    if unscored:
        raise SystemExit(
            f"{len(unscored)} records are unscored -- run rcc.score.score_run first"
        )

    rng = random.Random(seed)
    cache_only = _cached_qids(calls)
    repriced = reprice(calls, price_as, pricing_path) if price_as else None

    systems = sorted({r["system"] for r in records}, key=_series_rank)
    em_by_system: dict[str, dict[str, int]] = {}
    stats: list[SystemStats] = []

    for name in systems:
        rows = [r for r in records if r["system"] == name]
        em_by_system[name] = {r["qid"]: int(r["em"]) for r in rows}

        ems = [float(r["em"]) for r in rows]
        if repriced is not None:
            usd_total = sum(repriced.get((name, r["qid"]), 0.0) for r in rows)
        else:
            usd_total = sum(float(r["usd"]) for r in rows)
        n_correct = sum(int(r["em"]) for r in rows)

        latencies = [
            int(r["latency_ms"])
            for r in rows
            if (name, r["qid"]) not in cache_only
        ]

        terminated: dict[str, int] = {}
        for r in rows:
            key = r.get("terminated") or "answered"
            terminated[key] = terminated.get(key, 0) + 1

        tool_totals: dict[str, float] = {}
        for r in rows:
            for tool, count in (r.get("tool_calls") or {}).items():
                tool_totals[tool] = tool_totals.get(tool, 0.0) + count

        lo, hi = _bootstrap_ci(ems, rng)
        stats.append(
            SystemStats(
                system=name,
                n=len(rows),
                em=round(_mean(ems), 4),
                f1=round(_mean([float(r["f1"]) for r in rows]), 4),
                em_ci_low=round(lo, 4),
                em_ci_high=round(hi, 4),
                usd_total=round(usd_total, 6),
                usd_per_query=round(usd_total / len(rows), 6),
                # None, not 0 -- a system that got nothing right has no
                # meaningful cost per correct answer, and printing 0 would read
                # as "free".
                usd_per_correct=round(usd_total / n_correct, 6) if n_correct else None,
                llm_calls_mean=round(_mean([float(r["llm_calls"]) for r in rows]), 2),
                input_tokens_mean=round(_mean([float(r["input_tokens"]) for r in rows]), 1),
                cache_read_tokens_mean=round(_mean([float(r["cache_read_tokens"]) for r in rows]), 1),
                cache_write_tokens_mean=round(_mean([float(r["cache_write_tokens"]) for r in rows]), 1),
                output_tokens_mean=round(_mean([float(r["output_tokens"]) for r in rows]), 1),
                retrieved_unique_mean=round(_mean([float(r["retrieved_unique"]) for r in rows]), 1),
                retrieved_billed_mean=round(_mean([float(r["retrieved_billed"]) for r in rows]), 1),
                latency_p50_ms=_percentile(latencies, 0.50),
                latency_p95_ms=_percentile(latencies, 0.95),
                latency_n=len(latencies),
                terminated=terminated,
                tool_calls_mean={k: round(v / len(rows), 2) for k, v in sorted(tool_totals.items())},
            )
        )

    comparisons = {}
    for name in systems:
        if name == BASELINE:
            continue
        result = paired_win_rate(em_by_system, name, BASELINE, rng)
        if result:
            win_rate, n_shared = result
            comparisons[f"{name}_vs_{BASELINE}"] = {
                "p_better": round(win_rate, 4),
                "n_shared_queries": n_shared,
            }

    return {
        "run_id": meta.get("run_id", run_dir.name),
        "model": meta.get("model"),
        "auto_cache": meta.get("auto_cache"),
        "k": meta.get("k"),
        "effort": meta.get("effort"),
        "baseline": BASELINE,
        "priced_as": price_as,
        "cost_is_estimate": bool(price_as),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "systems": [asdict(s) for s in stats],
        "comparisons": comparisons,
    }


def _series_rank(name: str) -> tuple[int, str]:
    return (SERIES_ORDER.index(name) if name in SERIES_ORDER else len(SERIES_ORDER), name)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def markdown_table(summary: dict[str, Any]) -> str:
    prefix = ""
    if summary.get("priced_as"):
        prefix = (
            f"> Cost columns are **estimated** at `{summary['priced_as']}` rates from "
            f"token counts measured on `{summary.get('model')}`. Not a billed figure.\n\n"
        )
    head = (
        "| system | EM | 95% CI | F1 | $/query | **$/correct** | calls | "
        "retr. unique | retr. billed | p50 ms |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for s in summary["systems"]:
        per_correct = f"**{s['usd_per_correct']:.4f}**" if s["usd_per_correct"] is not None else "n/a"
        p50 = s["latency_p50_ms"] if s["latency_p50_ms"] is not None else "-"
        rows.append(
            f"| {s['system']} | {s['em']:.3f} | "
            f"{s['em_ci_low']:.3f}–{s['em_ci_high']:.3f} | {s['f1']:.3f} | "
            f"{s['usd_per_query']:.4f} | {per_correct} | {s['llm_calls_mean']:.1f} | "
            f"{s['retrieved_unique_mean']:.0f} | {s['retrieved_billed_mean']:.0f} | {p50} |"
        )
    table = prefix + head + "\n".join(rows)

    notes = []
    for key, cmp in summary.get("comparisons", {}).items():
        challenger = key.split("_vs_")[0]
        notes.append(
            f"- `{challenger}` beats `{summary['baseline']}` in "
            f"{cmp['p_better'] * 100:.1f}% of paired bootstrap resamples "
            f"(n={cmp['n_shared_queries']})"
        )
    if notes:
        table += "\n\n" + "\n".join(notes)
    return table


def chart(summary: dict[str, Any], out_path: Path, *, dark: bool = False) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping chart")
        return None

    points = [s for s in summary["systems"] if s["usd_per_correct"] is not None]
    if not points:
        return None

    palette = PALETTE_DARK if dark else PALETTE_LIGHT
    surface = SURFACE_DARK if dark else SURFACE_LIGHT
    ink, ink_soft = INK_DARK if dark else INK_LIGHT

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for s in points:
        colour = palette[_series_rank(s["system"])[0] % len(palette)]
        ax.scatter(
            s["em"], s["usd_per_correct"],
            s=190, color=colour, zorder=3,
            edgecolors=surface, linewidths=2,   # 2px surface ring on overlap
            label=s["system"],
        )
        # Direct labels on every point: ≤4 series, and the palette's contrast
        # warning obliges visible labels rather than colour alone.
        ax.annotate(
            s["system"],
            (s["em"], s["usd_per_correct"]),
            textcoords="offset points", xytext=(11, 5),
            color=ink, fontsize=10.5, fontweight="medium", zorder=4,
        )

    # Direct labels sit to the right of each point, so the data area needs room
    # or the rightmost label runs off the canvas.
    ax.margins(x=0.16, y=0.18)
    ax.set_xlabel("Exact match", color=ink_soft, fontsize=10)
    ax.set_ylabel("USD per correct answer", color=ink_soft, fontsize=10)
    ax.set_title(
        "Accuracy vs cost per correct answer  ·  "
        + (f"{summary['priced_as']} rates (est.)" if summary.get("priced_as")
           else (summary.get("model") or "")),
        color=ink, fontsize=13, fontweight="semibold", loc="left", pad=14,
    )

    ax.grid(True, color=ink_soft, alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink_soft)
        ax.spines[side].set_alpha(0.35)
    ax.tick_params(colors=ink_soft, labelsize=9)

    ax.annotate(
        "better → lower cost, higher accuracy",
        xy=(0.99, 0.02), xycoords="axes fraction",
        ha="right", color=ink_soft, fontsize=9, style="italic",
    )
    ax.legend(
        frameon=False, labelcolor=ink_soft, fontsize=9,
        loc="upper left", handletextpad=0.4,
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor=surface)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate a run into the headline table.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument(
        "--price-as",
        help="recompute cost at this model's rates from the recorded token "
             "buckets (an estimate -- see reprice())",
    )
    parser.add_argument("--pricing", default="configs/pricing.yaml")
    args = parser.parse_args(argv)

    summary = aggregate(
        args.run_dir, seed=args.seed,
        price_as=args.price_as, pricing_path=args.pricing,
    )
    (args.run_dir / "aggregate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    table = markdown_table(summary)
    (args.run_dir / "results.md").write_text(table + "\n", encoding="utf-8")
    print(table)

    for dark in (False, True):
        name = "accuracy_vs_cost_dark.png" if dark else "accuracy_vs_cost.png"
        path = chart(summary, args.run_dir / name, dark=dark)
        if path:
            print(f"\nchart: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
