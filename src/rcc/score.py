"""Scoring: exact match and token F1, SQuAD-normalized.

Deliberately not an LLM judge. A judge would make every number in this repo
arguable -- someone can always claim the judge favoured the agentic system, and
they would have no way to check. EM/F1 against the benchmarks' own gold answers
is reproducible by anyone with the dataset and a text editor, which is the
whole point of publishing a cost comparison.

The normalization is the standard SQuAD recipe (lowercase, strip articles,
strip punctuation, collapse whitespace) so numbers here are comparable to every
published HotpotQA / MuSiQue / 2Wiki result rather than to a private variant.

Scoring is a separate pass over `raw.jsonl` rather than something the systems
do inline: the meter writes `em`/`f1` as null during the run, and `score_run`
fills them in afterwards. That means re-scoring after a bug fix costs nothing
and never re-spends on the API.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

#: what a system says when the context does not contain the answer
UNANSWERABLE = "unanswerable"

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_ASCII_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    """SQuAD normalization, extended to non-ASCII punctuation.

    `string.punctuation` misses curly quotes, en-dashes and every non-Latin
    punctuation mark, which show up in Wikipedia-derived gold answers often
    enough to cost real points. Unicode category `P*` catches them all.
    """
    text = text.lower()
    text = "".join(
        ch
        for ch in text
        if ch not in _ASCII_PUNCT and not unicodedata.category(ch).startswith("P")
    )
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize(prediction) == normalize(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)

    # Both empty means both said nothing, which is a match. One empty means a
    # miss. Without this branch the shared-token count is 0 and the answer
    # "" scores 0.0 against gold "" -- which would punish a correct abstention.
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    shared = Counter(pred_tokens) & Counter(gold_tokens)
    n_shared = sum(shared.values())
    if n_shared == 0:
        return 0.0

    precision = n_shared / len(pred_tokens)
    recall = n_shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score(prediction: str, gold: Sequence[str]) -> tuple[int, float]:
    """Score against every accepted gold answer, keeping the best.

    HotpotQA and 2Wiki ship aliases ("JFK" / "John F. Kennedy"); taking the max
    is the standard treatment and the one the published baselines use.
    """
    if not gold:
        return 0, 0.0
    ems = [exact_match(prediction, g) for g in gold]
    f1s = [token_f1(prediction, g) for g in gold]
    return max(ems), max(f1s)


def score_records(records: Iterable[dict]) -> list[dict]:
    """Fill in `em` / `f1` on QueryRecord dicts, returning them in order."""
    out = []
    for rec in records:
        prediction = rec.get("prediction") or ""
        gold = rec.get("gold") or []

        # An errored or budget-exhausted query is scored, not skipped. It cost
        # money and it failed to answer; dropping it would flatter whichever
        # system is flakiest.
        em, f1 = score(prediction, gold)
        rec["em"] = em
        rec["f1"] = round(f1, 6)
        out.append(rec)
    return out


def score_run(run_dir: str | Path) -> dict[str, int]:
    """Score `<run_dir>/raw.jsonl` in place. Idempotent and API-free."""
    path = Path(run_dir) / "raw.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = score_records(records)

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in scored:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)

    counts: dict[str, int] = {}
    for rec in scored:
        counts[rec["system"]] = counts.get(rec["system"], 0) + rec["em"]
    return counts


# --------------------------------------------------------------------------
# Self-check: the edge cases that silently move a headline number
# --------------------------------------------------------------------------

_CASES: list[tuple[str, list[str], int, float]] = [
    ("1921", ["1921"], 1, 1.0),
    ("The 1921", ["1921"], 1, 1.0),                    # articles stripped
    ("1921.", ["1921"], 1, 1.0),                       # trailing punctuation
    ("“1921”", ["1921"], 1, 1.0),            # curly quotes
    ("JFK", ["John F. Kennedy", "JFK"], 1, 1.0),       # alias list, max wins
    ("John Kennedy", ["John F. Kennedy"], 0, 0.8),     # partial credit
    ("unanswerable", ["unanswerable"], 1, 1.0),
    ("", [""], 1, 1.0),                                # both empty
    ("", ["1921"], 0, 0.0),                            # empty prediction
    ("1922", ["1921"], 0, 0.0),
]

if __name__ == "__main__":
    failures = 0
    for prediction, gold, want_em, want_f1 in _CASES:
        got_em, got_f1 = score(prediction, gold)
        ok = got_em == want_em and abs(got_f1 - want_f1) < 0.05
        failures += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {prediction!r:<18} vs {gold} "
            f"-> em={got_em} f1={got_f1:.3f} (want em={want_em} f1~{want_f1})"
        )
    raise SystemExit(1 if failures else 0)
