"""Experiment 14: the n-gram rows worked out on the CPU, inside real passes.

The port hashes the tokens on the GPU and the table lookup waits for the result while the pass is being built.
Passes alternate between that and the same hash in numpy; the logits must be the same bits.

    uv run python bench/experiments/e14_ngram_on_cpu.py --model <shard 1>
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--passes", type=int, default=80)
args = p.parse_args()
model, _ = load(args.model)
ples = [l.ple for l in model.model.layers if l.ple is not None]
hosts = [x.ple_embedding for x in ples]
engines = {"hash on the GPU": Engine(model), "hash on the CPU": Engine(model)}
for e in engines.values():
    e.warm()
    e.read(list(range(1000, 1200)))
print(f"{len(ples)} n-gram layer(s)")
for T in (1, 4):
    times, same = {name: [] for name in engines}, True
    for i in range(args.passes):
        ids = mx.array([[1000 + (i * 7 + j) % 500 for j in range(T)]])
        out = {}
        for name, e in engines.items():
            for ple, host in zip(ples, hosts):
                ple.ple_embedding = host if name.endswith("CPU") else host.port
            t0 = time.perf_counter()
            logits, w = e._forward(ids)
            mx.eval(logits)
            e._commit(w, T)
            times[name].append((time.perf_counter() - t0) * 1e3)
            out[name] = logits
        same = same and mx.array_equal(*out.values()).item()
    q = {name: sorted(ts)[len(ts) // 4] for name, ts in times.items()}
    print(f"{T} tokens: " + "   ".join(f"{name} {ms:6.2f} ms" for name, ms in q.items())
          + f"   ({q['hash on the CPU'] - q['hash on the GPU']:+.2f})   same logits: {same}", flush=True)
for ple, host in zip(ples, hosts):
    ple.ple_embedding = host
