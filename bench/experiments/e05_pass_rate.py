"""Experiment 5 (and the yardstick for later ones): the time of a pass of one token and of four, alternated.

MLX settings that are read once at start (command-buffer limits) can only be compared between processes, so this
prints one line to be collected by a loop over settings; run each setting several times and take its best.

    MLX_MAX_OPS_PER_BUFFER=80 MLX_MAX_MB_PER_BUFFER=1000000 uv run python bench/experiments/e05_pass_rate.py --model <shard 1>
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--passes", type=int, default=120)
p.add_argument("--label", default="")
args = p.parse_args()
model, _ = load(args.model)
engine = Engine(model)
engine.warm()
engine.read(list(range(1000, 1200)))
times = {1: [], 4: []}
for i in range(args.passes * 2):
    T = 1 if i % 2 == 0 else 4
    ids = mx.array([[1000 + (i * 7 + j) % 500 for j in range(T)]])
    t0 = time.perf_counter()
    logits, w = engine._forward(ids)
    mx.eval(logits)
    engine._commit(w, T)
    times[T].append((time.perf_counter() - t0) * 1e3)


def quartile(xs):
    return sorted(xs)[len(xs) // 4]


label = args.label or " ".join(f"{k}={v}" for k, v in os.environ.items() if k.startswith("MLX_"))
print(f"{label or 'default':60} one token {quartile(times[1]):6.2f} ms   four {quartile(times[4]):6.2f} ms", flush=True)
