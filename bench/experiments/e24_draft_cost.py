"""Experiment 24: what the draft head's guesses cost a step, and how much of that is the output layer.

A guess is a pass of the head's one layer and then the output layer, all 248,320 rows of it, for one token; the
guess after it needs this one's token, so they come one after another. Timed in a real answer with three
guesses a pass: the step, the part of it that is drafting, and drafting again with the output layer replaced
by something that reads no weights (the guesses are then wrong; only the time is of interest, and the engine
is one that is thrown away).

    uv run python bench/experiments/e24_draft_cost.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from spec_decode import PROMPTS, tokenize  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--mtp", required=True)
p.add_argument("--steps", type=int, default=80)
args = p.parse_args()
model, _ = load(args.model)
mtp = load_mtp(args.mtp, model.args.text, mx.float16)
head = model.lm_head
rows = getattr(head, "output_dims", None) or head.weight.shape[0]


def quartile(xs):
    return sorted(xs)[len(xs) // 4]


def run(stub: bool):
    engine = Engine(model, mtp, 3, as_sure=False)
    engine.warm()
    prompt = tokenize(args.model, f"<|im_start|>user\n{PROMPTS['code']}<|im_end|>\n<|im_start|>assistant\n")
    engine.read(prompt)
    draft_ms, step_ms, sizes = [], [], []
    real_draft = engine._draft

    def timed(residual, following):
        if stub:
            model.lm_head = lambda x: mx.zeros((*x.shape[:-1], rows), dtype=x.dtype) + x[..., :1]
        t0 = time.perf_counter()
        out = real_draft(residual, following)
        draft_ms.append((time.perf_counter() - t0) * 1e3)
        model.lm_head = head
        return out

    engine._draft = timed
    for _ in range(args.steps):
        t0 = time.perf_counter()
        sizes.append(len(engine.guesses) + 1)
        engine.step()
        step_ms.append((time.perf_counter() - t0) * 1e3)
    return quartile(step_ms), quartile(draft_ms), sum(sizes) / len(sizes)


for rep in range(2):
    a = run(False)
    b = run(True)
    print(f"three guesses a pass: a step {a[0]:.1f} ms of which drafting {a[1]:.2f} ms | drafting without the output layer "
          f"{b[1]:.2f} ms: the output layer costs the three guesses {a[1] - b[1]:.2f} ms", flush=True)
