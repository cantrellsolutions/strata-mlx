"""Experiment 8: does a kernel read faster when each thread reads less?

A float16 matrix of 324 rows by 10,240 (a hyper-connection's projection down) times a vector, for one token and
for four, with 32 to 1,024 threads sharing a row. Each kernel waits for the one before, over the real model's
97 matrices (630 MB, so nothing stays in a cache). MLX's own matmul on the same matrices for comparison.

    uv run python bench/experiments/e08_thread_grain.py --model <shard 1>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.load import load  # noqa: E402

SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint j = thread_position_in_grid.x / (32 * G);
    threadgroup float shared[G * T];
    auto row = w + j * COLS;
    float acc[T];
    for (uint t = 0; t < T; ++t) { acc[t] = 0.0f; }
    uint per = COLS / (32 * G);
    for (uint i = lane * per; i < (lane + 1) * per; ++i) {
        float v = float(row[i]);
        for (uint t = 0; t < T; ++t) { acc[t] += v * x[t * COLS + i]; }
    }
    for (uint t = 0; t < T; ++t) {
        float s = simd_sum(acc[t]);
        if (lane % 32 == 0) { shared[(lane / 32) * T + t] = s; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float dep = prev[0] * 0.0f;
        for (uint t = 0; t < T; ++t) {
            float total = dep;
            for (uint g = 0; g < G; ++g) { total += shared[g * T + t]; }
            out[t * ROWS + j] = total;
        }
    }
"""


def kernel(rows, cols, groups, tokens):
    header = f"constant uint ROWS = {rows};\nconstant uint COLS = {cols};\nconstant uint G = {groups};\nconstant uint T = {tokens};\n"
    return mx.fast.metal_kernel(name=f"e08_mv_{groups}_{tokens}", input_names=["w", "x", "prev"], output_names=["out"],
                                source=SOURCE, header=header)


def best(build, n, rounds=9):
    out = float("inf")
    for _ in range(rounds):
        g = build()
        t0 = time.perf_counter()
        mx.eval(g)
        out = min(out, time.perf_counter() - t0)
    return out / n * 1e6


p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--json")
args = p.parse_args()
model, _ = load(args.model)
m = model.model
mats = [gr.input_mix_weight_down.weight for l in m.layers for gr in (l.attn_hyper_connection, l.mlp_hyper_connection)]
rows, cols = mats[0].shape
size = mats[0].nbytes
out_rows = []
for T in (1, 4):
    x = mx.random.normal((T, cols))
    mx.eval(x)

    def own():
        prev, xs = None, x.astype(mx.float16)
        for w in mats:
            prev = (xs if prev is None else xs + 0.0 * prev[:, :1]) @ w.T
        return prev
    us = best(own, len(mats))
    print(f"{T} tokens  MLX's own matmul (and an add)      {us:6.1f} us a matrix  {size / us / 1e3:4.0f} GB/s", flush=True)
    out_rows.append({"tokens": T, "threads_per_row": "mlx", "us": round(us, 1)})
    for G in (1, 2, 4, 8, 16, 32):
        kern = kernel(rows, cols, G, T)

        def chain():
            prev = mx.zeros((T, rows))
            for w in mats:
                (prev,) = kern(inputs=[w, x, prev], grid=(32 * G * rows, 1, 1), threadgroup=(32 * G, 1, 1),
                               output_shapes=[(T, rows)], output_dtypes=[mx.float32])
            return prev
        got = chain()
        want = x @ mats[-1].astype(mx.float32).T
        err = float(mx.abs(got - want).max() / mx.abs(want).max())
        us = best(chain, len(mats))
        print(f"{T} tokens  {32 * G:5} threads a row ({cols // (32 * G):3} weights each)  {us:6.1f} us a matrix  "
              f"{size / us / 1e3:4.0f} GB/s  differs by {err:.0e}", flush=True)
        out_rows.append({"tokens": T, "threads_per_row": 32 * G, "us": round(us, 1), "rel_err": err})
if args.json:
    Path(args.json).write_text(json.dumps(out_rows, indent=1) + "\n")
