"""Experiment 34: a short piece of a prompt right after another, and after the GPU has had nothing to do.

Seen while timing pieces of prompts by their length (e30), whose reads each follow a rest of some seconds: 64
tokens took as long as 128. Here the same piece is read one time after another, then after rests of growing
length, and one token of an answer likewise. See ISSUES.md, issue 1; nothing is concluded here.

    uv run python bench/experiments/e34_after_idle.py --model <shard 1>
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from spec_decode import EDIT_ME, tokenize  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--lengths", nargs="+", type=int, default=[64, 256])
p.add_argument("--rests", nargs="+", type=float, default=[0.5, 2, 4, 8])
args = p.parse_args()

model, _ = load(args.model)
prompt = tokenize(args.model, "<|im_start|>user\nSay what this is for.\n\n```python\n" + EDIT_ME * 9 + "```<|im_end|>\n<|im_start|>assistant\n")
engine = Engine(model)
engine.warm()


def read(T: int) -> float:
    """One piece of T tokens into an empty conversation, in ms."""
    engine.reset()
    t0 = time.perf_counter()
    logits, w = engine._forward(mx.array([prompt[:T]]), last_only=True)
    mx.eval(logits)
    engine._commit(w, T)
    mx.eval(*[a for v in engine.gdn.values() for a in v])
    return (time.perf_counter() - t0) * 1e3


def one_token() -> float:
    engine.reset()
    engine.read(prompt[:40])
    t0 = time.perf_counter()
    engine.step(1)
    return (time.perf_counter() - t0) * 1e3


for T in args.lengths:
    read(T), read(T)
    row = f"{T:4} tokens: one after another " + " ".join(f"{read(T):5.0f}" for _ in range(5))
    for rest in args.rests:
        time.sleep(rest)
        row += f" | after {rest:g} s {read(T):5.0f}"
    print(row + " ms", flush=True)
ts = [one_token() for _ in range(3)]
row = f"one token of an answer: one after another {min(ts):.0f}"
for rest in args.rests:
    time.sleep(rest)
    row += f" | after {rest:g} s {one_token():.0f}"
print(row + " ms")
