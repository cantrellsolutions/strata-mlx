"""Experiment 44: threads to a row of the Q2_0 table kernel, the rule before against the one now, alternated,
the real model's 48 expert layers chained, windows of 1 to 128 tokens.

    uv run python bench/experiments/e44_lookup_threads.py <shard 1>
"""

import sys, time
import mlx.core as mx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.load import load
from strata_mlx import blocks

model, _ = load(sys.argv[1])
glus = [l.mlp.switch_mlp for l in model.model.layers]

FORCE = {}
orig = blocks._threads_per_row
def lookup_matvec(lin, idx, table, share):
    pairs = idx.shape[0]
    per_row = FORCE.get(lin.input_dims)
    if per_row is None:
        per_row = blocks._threads_per_row(lin.input_dims // 32)
        if pairs > 20:
            per_row = min(per_row, 4)
    key = ("lookup", lin.output_dims, lin.input_dims, share, per_row)
    if key not in blocks._kernels:
        blocks._kernels[key] = mx.fast.metal_kernel(
            name="q2_lookup", input_names=["blocks", "idx", "table"], output_names=["out"], source=blocks._LOOKUP,
            header=blocks._header("Q2_0", lin.output_dims, lin.input_dims, P=per_row, SHARE=share))
    (out,) = blocks._kernels[key](inputs=[lin.blocks, idx, table], template=[("OutT", table.dtype)],
                           grid=(per_row * lin.output_dims, pairs, 1), threadgroup=(32, 1, 1),
                           output_shapes=[(pairs, lin.output_dims)], output_dtypes=[table.dtype])
    return out
blocks.lookup_matvec = lookup_matvec
print("default per_row: 2560 cols ->", blocks._threads_per_row(2560 // 32), "; inner cols", glus[0].down_proj.input_dims, "->", blocks._threads_per_row(glus[0].down_proj.input_dims // 32))

mx.random.seed(0)
def run(T, idx, x, n=25):
    ts = []
    for r in range(n):
        y = x
        t0 = time.perf_counter()
        for g in glus:
            y = g(y, idx).sum(-2) + x
        mx.eval(y)
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 4]

def rule(cols, pairs):
    if cols == 2560:
        return 8 if pairs <= 40 else 2
    return 2
for T in (1, 2, 4, 6, 8, 16, 32, 128):
    x = (mx.random.normal((T, 2560)) * 0.5).astype(mx.float16)
    idx = mx.random.randint(0, 512, (T, 10)).astype(mx.uint32)
    a, b = [], []
    for r in range(6):
        FORCE.clear(); a.append(run(T, idx, x, 8))
        FORCE.clear(); FORCE.update({2560: rule(2560, T * 10), 640: rule(640, T * 10)}); b.append(run(T, idx, x, 8))
    a, b = sorted(a)[2], sorted(b)[2]
    print(f"T={T}: now {a:.2f}  new {b:.2f}  ({(a - b) / a:+.0%})", flush=True)
