"""Run all four systems against a fake corpus and a scripted fake model.

No network, no real SDK. The point is to exercise the control flow and the
accounting: does every system produce a QueryRecord, do tool loops terminate,
does the context tracker stop double-billing a re-read chunk.
"""
import json, sys, types, tempfile
from pathlib import Path

# ---- stub the anthropic SDK -------------------------------------------------
anthropic = types.ModuleType("anthropic")


class Usage:
    def __init__(self, input_tokens=100, output_tokens=20,
                 cache_creation_input_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class Block:
    def __init__(self, type, text=None, id=None, name=None, input=None):
        self.type, self.text, self.id, self.name, self.input = type, text, id, name, input


class Message:
    def __init__(self, content, stop_reason="end_turn", usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or Usage()

    def to_dict(self):
        return {
            "content": [
                {"type": b.type, "text": b.text, "id": b.id, "name": b.name, "input": b.input}
                for b in self.content
            ],
            "stop_reason": self.stop_reason,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
            },
        }

    @classmethod
    def model_validate(cls, d):
        return cls(
            [Block(**b) for b in d["content"]],
            d["stop_reason"],
            Usage(**d["usage"]),
        )


anthropic.types = types.ModuleType("anthropic.types")
anthropic.types.Message = Message


class APIStatusError(Exception):
    pass


anthropic.APIStatusError = APIStatusError

SCRIPT = {"n": 0}


class Messages:
    def create(self, **kw):
        SCRIPT["n"] += 1
        tools = kw.get("tools")
        # Agent path: first call searches, second reads, third answers.
        if tools:
            step = SCRIPT.setdefault("agent_step", 0)
            SCRIPT["agent_step"] = step + 1
            if step == 0:
                return Message([Block("tool_use", id="t1", name="keyword_search",
                                      input={"query": "einstein", "k": 3})],
                               stop_reason="tool_use")
            if step == 1:
                # read the same chunk twice on purpose -> context tracker test
                return Message([Block("tool_use", id="t2", name="chunk_read",
                                      input={"chunk_ids": ["d0::0", "d0::0"]})],
                               stop_reason="tool_use")
            return Message([Block("text", text="<answer>1921</answer>")])
        if kw.get("max_tokens") == 128:  # iterative's planner
            return Message([Block("text", text="nobel prize year")])
        return Message([Block("text", text="<answer>1921</answer>")])


class Anthropic:
    def __init__(self, *a, **k):
        self.messages = Messages()


anthropic.Anthropic = Anthropic
sys.modules["anthropic"] = anthropic
sys.modules["anthropic.types"] = anthropic.types

# ---- fake index -------------------------------------------------------------
sys.path.insert(0, str(Path(r"C:\Users\intikhab azam\rag-cost-curve\src")))
from rcc.systems.base import Chunk, Hit  # noqa: E402
from rcc.meter import Meter  # noqa: E402
from rcc.systems import REGISTRY  # noqa: E402

CHUNKS = [
    Chunk(f"d{d}::0", f"d{d}", f"Doc {d}", f"Body text for document {d}. " * 20, 120)
    for d in range(6)
]


class FakeIndex:
    def keyword_search(self, query, k):
        return [Hit(c, 1.0, c.text[:80]) for c in CHUNKS[:k]]

    def semantic_search(self, query, k):
        return [Hit(c, 0.9, c.text[:80]) for c in CHUNKS[:k]]

    def rerank(self, query, hits, k):
        return list(hits)[:k]

    def get_chunk(self, cid):
        for c in CHUNKS:
            if c.chunk_id == cid:
                return c
        raise KeyError(cid)


# ---- run --------------------------------------------------------------------
out = Path(tempfile.mkdtemp()) / "run"
meter = Meter("smoke", out)
idx = FakeIndex()

for name, cls in REGISTRY.items():
    SCRIPT["agent_step"] = 0
    system = cls(idx, meter, model="claude-opus-5")
    with meter.query(name, "fake", "q1", "When did Einstein win the Nobel?", ["1921"]) as scope:
        ans = system.run("When did Einstein win the Nobel?", scope)
        scope.prediction = ans.text
        scope.terminated = ans.terminated
meter.close()

print(f"{'system':<11}{'pred':<8}{'calls':>6}{'uniq':>7}{'billed':>8}{'usd':>10}  tools")
for line in (out / "raw.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    print(f"{r['system']:<11}{r['prediction']:<8}{r['llm_calls']:>6}"
          f"{r['retrieved_unique']:>7}{r['retrieved_billed']:>8}"
          f"{r['usd']:>10.6f}  {r['tool_calls']}")
