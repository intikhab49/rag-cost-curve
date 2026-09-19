"""Cost, token and latency accounting for RAG architecture comparison.

Design rule: every LLM call in this project goes through `Meter`. Systems never
report their own usage -- the meter reads it off the API response. An agent loop
cannot under-report by forgetting to instrument a step, because the only way to
reach the model is through `Meter.generate`.

The number this module exists to produce is `usd_per_correct`. Getting it right
depends on three things most harnesses get wrong:

1. Input tokens are not one bucket. Cache reads are ~0.1x and cache writes
   ~1.25x. An agent loop resends its whole history every step, so pricing all
   input at 1.0x overstates agentic cost by several multiples.
2. Retrieved tokens must be counted two ways -- distinct corpus text ever seen
   (what the A-RAG paper reports) and the same text summed over every call that
   re-sent it (what you actually pay for).
3. Determinism cannot come from `temperature=0`: sampling params are rejected
   with a 400 on Opus 5, Sonnet 5 and the 4.6+ family. It comes from the
   on-disk response cache here, keyed by the exact request.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import anthropic

# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

PRICING_FETCHED_ON = "2026-09-19"  # bump whenever the table below is re-checked


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens, per bucket."""

    input: float
    output: float
    cache_write: float
    cache_read: float

    @classmethod
    def derived(cls, input_: float, output: float) -> "ModelPrice":
        # Standard multipliers: 5m-TTL cache write is 1.25x input, read is 0.1x.
        return cls(input_, output, input_ * 1.25, input_ * 0.10)


DEFAULT_PRICING: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice.derived(5.00, 25.00),
    "claude-sonnet-5": ModelPrice.derived(2.00, 10.00),
    "claude-haiku-4-5": ModelPrice.derived(1.00, 5.00),
}


def load_pricing(path: str | Path | None = None) -> dict[str, ModelPrice]:
    """Load configs/pricing.yaml if present, else fall back to DEFAULT_PRICING.

    Kept in config so a reader can check the numbers against published rates
    without reading the source, and so a stale table shows up as a stale
    `fetched_on` rather than a silently wrong dollar figure.
    """
    if path is None:
        return dict(DEFAULT_PRICING)
    import yaml  # local import: only needed when a config is actually used

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: dict[str, ModelPrice] = {}
    for model, p in (raw.get("models") or {}).items():
        if "cache_write" in p and "cache_read" in p:
            out[model] = ModelPrice(p["input"], p["output"], p["cache_write"], p["cache_read"])
        else:
            out[model] = ModelPrice.derived(p["input"], p["output"])
    return out


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class CallRecord:
    """One LLM call. Written to runs/<id>/calls.jsonl."""

    run_id: str
    system: str
    dataset: str
    qid: str
    step: int  # 0-indexed step within this query's loop
    model: str
    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int
    retrieved_tokens_in_prompt: int
    latency_ms: int
    stop_reason: str
    cached_response: bool
    usd: float


@dataclass
class QueryRecord:
    """One (system, query) pair. Written to runs/<id>/raw.jsonl."""

    run_id: str
    system: str
    dataset: str
    qid: str
    question: str
    gold: list[str]
    prediction: str
    em: int | None
    f1: float | None
    llm_calls: int
    tool_calls: dict[str, int]
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    retrieved_unique: int
    retrieved_billed: int
    latency_ms: int
    usd: float
    terminated: str  # answered | max_steps | context_limit | refusal | error


@dataclass
class QueryScope:
    """Mutable handle for the query currently being metered.

    A system sets `prediction` / `terminated` and calls `tool()`; the runner
    sets `em` / `f1` after scoring. Everything else the meter fills in itself.
    """

    system: str
    dataset: str
    qid: str
    question: str
    gold: list[str] = field(default_factory=list)

    prediction: str = ""
    terminated: str = "answered"
    em: int | None = None
    f1: float | None = None
    tool_calls: dict[str, int] = field(default_factory=dict)

    # meter-owned
    step: int = 0
    calls: list[CallRecord] = field(default_factory=list)
    seen_chunks: dict[str, int] = field(default_factory=dict)  # chunk_id -> tokens
    retrieved_billed: int = 0
    retrieved_this_call: int = 0
    t0: float = 0.0

    def tool(self, name: str) -> None:
        """Systems call this once per retrieval-tool invocation."""
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1


# --------------------------------------------------------------------------
# Meter
# --------------------------------------------------------------------------


class Meter:
    """Run-scoped accountant. One per run, shared by every system in the run."""

    def __init__(
        self,
        run_id: str,
        out_dir: str | Path,
        *,
        pricing: dict[str, ModelPrice] | None = None,
        cache_dir: str | Path | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.run_id = run_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pricing = pricing or dict(DEFAULT_PRICING)
        self.cache_dir = Path(cache_dir) if cache_dir else self.out_dir.parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or anthropic.Anthropic()

        self._lock = threading.Lock()
        self._calls_fh = (self.out_dir / "calls.jsonl").open("a", encoding="utf-8")
        self._raw_fh = (self.out_dir / "raw.jsonl").open("a", encoding="utf-8")
        self._local = threading.local()

    def __enter__(self) -> "Meter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._calls_fh.close()
        self._raw_fh.close()

    # -- query scoping -----------------------------------------------------

    @property
    def scope(self) -> QueryScope:
        s = getattr(self._local, "current", None)
        if s is None:
            raise RuntimeError("no active query scope -- wrap work in Meter.query()")
        return s

    @contextmanager
    def query(
        self, system: str, dataset: str, qid: str, question: str, gold: list[str]
    ) -> Iterator[QueryScope]:
        """Scope one (system, query). Always emits a QueryRecord, even on error."""
        scope = QueryScope(
            system=system, dataset=dataset, qid=qid, question=question, gold=gold
        )
        scope.t0 = time.perf_counter()
        self._local.current = scope
        try:
            yield scope
        except anthropic.APIError:
            # APIError, not APIStatusError: connection and timeout errors are
            # siblings of status errors, not subclasses. Catching only the
            # narrow one let a single DNS blip escape and kill an 800-query
            # sweep at query 342.
            #
            # A failed query is data, not a lost row: it still cost money and it
            # still counts against the system's accuracy.
            scope.terminated = "error"
            scope.prediction = ""
        finally:
            self._local.current = None
            self._emit_query(scope)

    # -- retrieval accounting ---------------------------------------------

    def note_retrieval(self, chunk_id: str, tokens: int) -> None:
        """Call this the moment corpus text is placed into a prompt.

        `tokens` comes precomputed from the index (counted once at build time
        via messages.count_tokens), so this stays a dict update rather than a
        round trip per chunk.
        """
        s = self.scope
        s.seen_chunks.setdefault(chunk_id, tokens)
        s.retrieved_billed += tokens
        s.retrieved_this_call += tokens

    # -- the only path to the model ---------------------------------------

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        effort: str = "high",
        thinking: dict[str, Any] | None = None,
        auto_cache: bool = True,
        cache: bool = True,
        terminal: bool = True,
    ) -> anthropic.types.Message:
        """Metered wrapper around messages.create.

        Deliberately exposes no `temperature` / `top_p` / `top_k`: those are
        rejected with a 400 on Opus 5, Sonnet 5 and the 4.6+ family.
        Reproducibility comes from the response cache, keyed on the exact
        request.

        `auto_cache` turns on prompt caching (top-level `cache_control` caches
        the last cacheable block). It must be identical across every system in
        a run or the comparison is void -- an agent loop resends its whole
        history each step, so caching is worth several multiples to it and
        nothing to a single-shot system. Whether it changes the verdict is
        itself a result worth reporting, so run the sweep both ways.
        """
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "output_config": {"effort": effort},
        }
        if auto_cache:
            params["cache_control"] = {"type": "ephemeral"}
        if system is not None:
            params["system"] = system
        if tools:
            params["tools"] = tools
        if thinking is not None:
            params["thinking"] = thinking

        key = self._cache_key(params)
        cached = self._cache_get(key) if cache else None

        if cached is not None:
            msg = anthropic.types.Message.model_validate(cached)
            latency_ms = 0  # a cache hit measures nothing; report.py excludes these
            was_cached = True
        else:
            t0 = time.perf_counter()
            msg = self.client.messages.create(**params)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            was_cached = False
            if cache:
                self._cache_put(key, msg.to_dict())

        self._record_call(
            msg, model=model, latency_ms=latency_ms,
            cached=was_cached, terminal=terminal,
        )
        return msg

    # -- internals ---------------------------------------------------------

    def _record_call(
        self,
        msg: anthropic.types.Message,
        *,
        model: str,
        latency_ms: int,
        cached: bool,
        terminal: bool = True,
    ) -> None:
        s = self.scope
        u = msg.usage
        rec = CallRecord(
            run_id=self.run_id,
            system=s.system,
            dataset=s.dataset,
            qid=s.qid,
            step=s.step,
            model=model,
            input_tokens=u.input_tokens or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            output_tokens=u.output_tokens or 0,
            retrieved_tokens_in_prompt=s.retrieved_this_call,
            latency_ms=latency_ms,
            stop_reason=msg.stop_reason or "",
            cached_response=cached,
            usd=0.0,
        )
        rec.usd = self.price_call(rec)

        s.step += 1
        s.retrieved_this_call = 0
        s.calls.append(rec)
        with self._lock:
            self._calls_fh.write(json.dumps(asdict(rec)) + "\n")

        # stop_reason is load-bearing: an agent that ran out of room and guessed
        # is a different failure from one that retrieved the wrong thing.
        #
        # Only a TERMINAL call can set context_limit. A deliberately capped
        # auxiliary call -- a short planning step, say -- hits max_tokens by
        # design, and letting it set the query's terminal state mislabels a
        # perfectly good answer as a failure. That mislabelled 143/200
        # iterative queries in the first 200-query run.
        if msg.stop_reason == "refusal":
            s.terminated = "refusal"
        elif msg.stop_reason == "max_tokens" and terminal:
            s.terminated = "context_limit"

    def price_call(self, rec: CallRecord) -> float:
        p = self.pricing.get(rec.model)
        if p is None:
            raise KeyError(f"no pricing for {rec.model!r} -- add it to configs/pricing.yaml")
        return (
            rec.input_tokens * p.input
            + rec.cache_creation_input_tokens * p.cache_write
            + rec.cache_read_input_tokens * p.cache_read
            + rec.output_tokens * p.output
        ) / 1_000_000

    def _emit_query(self, s: QueryScope) -> None:
        rec = QueryRecord(
            run_id=self.run_id,
            system=s.system,
            dataset=s.dataset,
            qid=s.qid,
            question=s.question,
            gold=s.gold,
            prediction=s.prediction,
            em=s.em,
            f1=s.f1,
            llm_calls=len(s.calls),
            tool_calls=dict(s.tool_calls),
            input_tokens=sum(c.input_tokens for c in s.calls),
            cache_read_tokens=sum(c.cache_read_input_tokens for c in s.calls),
            cache_write_tokens=sum(c.cache_creation_input_tokens for c in s.calls),
            output_tokens=sum(c.output_tokens for c in s.calls),
            retrieved_unique=sum(s.seen_chunks.values()),
            retrieved_billed=s.retrieved_billed,
            latency_ms=int((time.perf_counter() - s.t0) * 1000),
            usd=sum(c.usd for c in s.calls),
            terminated=s.terminated,
        )
        with self._lock:
            self._raw_fh.write(json.dumps(asdict(rec)) + "\n")
            self._raw_fh.flush()
            self._calls_fh.flush()

    # -- response cache ----------------------------------------------------

    def _cache_key(self, params: dict[str, Any]) -> str:
        blob = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt entry costs one re-spend, not a crashed run

    def _cache_put(self, key: str, payload: dict[str, Any]) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        tmp.replace(p)
