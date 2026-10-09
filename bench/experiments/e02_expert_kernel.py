"""Experiment 2: how fast can the Q2_0 expert kernel read?

Experiment 1: the routed experts cost 4.3 ms for one token and 2.1 more for each further one, 154-252 GB/s where
MLX's own quantized product reads dense weights at 350-458. This times the kernel alone, over the real model's 48
layers in one evaluation, in variants:

    now       blocks.py: every run of 32 weights unpacked to floats in a local array, then multiplied
    direct    Q2_0 only: 16 bits of codes at a time, straight into the sum, the scale applied once a run

each with several numbers of threads to a row. Results are checked against the first variant.

    uv run python bench/experiments/e02_expert_kernel.py --model <shard 1>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx import blocks  # noqa: E402
from strata_mlx.load import load  # noqa: E402

DIRECT = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    auto x_ = x + (pair / SHARE) * COLS;
    float sum = 0.0f;
    for (uint n = part; n < RUNS; n += P) {
        const device ushort* b = (const device ushort*)(row + (n / 2) * 18);
        const device ushort* q = b + 1 + 4 * (n % 2);
        auto xr = x_ + 32 * n;
        float s = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            uint c = q[i];
            for (uint j = 0; j < 8; ++j) {
                s += (float((c >> (2 * j)) & 3) - 1.0f) * float(xr[8 * i + j]);
            }
        }
        sum += float(as_type<half>(b[0])) * s;
    }
    float total = sum;
    for (ushort k = 1; k < P; ++k) {
        total += simd_shuffle_xor(sum, k);
    }
    if (part == 0) {
        out[pair * ROWS + r] = static_cast<OutT>(total);
    }
"""


def kernel(source, rows, cols, per_row, name):
    return mx.fast.metal_kernel(name=name, input_names=["blocks", "idx", "x"], output_names=["out"], source=source,
                                header=blocks._header("Q2_0", rows, cols, P=per_row, SHARE=1))


def best(fn, rounds):
    out = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        mx.eval(fn())
        out = min(out, time.perf_counter() - t0)
    return out * 1e3


p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--tokens", nargs="+", type=int, default=[1, 4])
p.add_argument("--rounds", type=int, default=9)
p.add_argument("--json")
args = p.parse_args()
model, _ = load(args.model)
layers = model.model.layers
k = model.args.text.num_experts_per_tok
rng = np.random.default_rng(0)
rows_out = []
for part, threads in (("gate_proj", (4, 8, 16, 32)), ("down_proj", (1, 2, 4, 8, 16))):
    lins = [getattr(l.mlp.switch_mlp, part) for l in layers]
    rows, cols = lins[0].output_dims, lins[0].input_dims
    one = lins[0].blocks.nbytes // lins[0].num_experts
    for T in args.tokens:
        pairs = T * k
        idx = mx.array(np.concatenate([rng.choice(512, k, replace=False) for _ in range(T)]).astype(np.uint32))
        x = mx.random.normal((pairs, cols)).astype(mx.float16)
        mx.eval(idx, x)
        want = None
        for name, source in (("now", blocks._MATVEC), ("direct", DIRECT)):
            for P in threads:
                kern = kernel(source, rows, cols, P, f"e02_{name}_{part}_{P}")

                def run():
                    return [kern(inputs=[lin.blocks, idx, x], template=[("OutT", x.dtype)], grid=(P * rows, pairs, 1),
                                 threadgroup=(32, 1, 1), output_shapes=[(pairs, rows)], output_dtypes=[x.dtype])[0]
                            for lin in lins]
                got = run()[0].astype(mx.float32)
                mx.eval(got)
                if want is None:
                    want = got
                err = float(mx.abs(got - want).max() / mx.abs(want).max())
                ms = best(run, args.rounds)
                gbs = len(lins) * pairs * one / ms / 1e6
                default = " (as now)" if name == "now" and P == lins[0].per_row else ""
                rows_out.append({"tensor": part, "tokens": T, "kernel": name, "threads_per_row": P, "ms": round(ms, 2),
                                 "gb_s": round(gbs), "rel_err": err})
                print(f"{part:9} {rows}x{cols}  {T} tokens  {name:6} {P:2} threads a row  {ms:6.2f} ms  {gbs:4.0f} GB/s  "
                      f"differs by {err:.1e}{default}", flush=True)
if args.json:
    Path(args.json).write_text(json.dumps(rows_out, indent=1) + "\n")
