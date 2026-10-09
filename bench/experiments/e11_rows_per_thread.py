"""Experiment 11: several rows of an expert in one thread, the activations loaded once for all of them.

Experiments 3 and 10 fit one line in the bytes a run of 32 weights reads: 8 of codes and 64 of activations now.
Every row reads the whole input again: for four tokens a tensor reads 0.9 GB of weights and 6.3 GB of
activations. A thread that takes K rows loads a run's activations once (into registers, not an array) and
then K rows' codes: 8 + 64 / K bytes a run.

    now      blocks.py's kernel
    K rows   K = 2, 4, 8

Over the real model's 48 layers in one evaluation, as experiments 3 and 10.

    uv run python bench/experiments/e11_rows_per_thread.py --model <shard 1>
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

HEADER = """
inline float dot8(uint c, float4 a, float4 b) {
    float4 lo = float4(float(c & 3), float((c >> 2) & 3), float((c >> 4) & 3), float((c >> 6) & 3)) - 1.0f;
    float4 hi = float4(float((c >> 8) & 3), float((c >> 10) & 3), float((c >> 12) & 3), float((c >> 14) & 3)) - 1.0f;
    return dot(lo, a) + dot(hi, b);
}
inline float run_of(const device uchar* row, uint n, float4 x0, float4 x1, float4 x2, float4 x3, float4 x4,
                    float4 x5, float4 x6, float4 x7) {
    const device ushort* b = (const device ushort*)(row + (n / 2) * 18);
    ushort4 c = *(const device packed_ushort4*)(b + 1 + 4 * (n % 2));
    return float(as_type<half>(b[0])) * (dot8(c.x, x0, x1) + dot8(c.y, x2, x3) + dot8(c.z, x4, x5) + dot8(c.w, x6, x7));
}
"""


def source(k: int) -> str:
    sums = "".join(f"float sum{i} = 0.0f; " for i in range(k))
    rows = "".join(f"sum{i} += run_of(row + {i} * ROW_BYTES, n, x0, x1, x2, x3, x4, x5, x6, x7); " for i in range(k))
    outs = "".join(
        f"{{ float total = sum{i}; for (ushort j = 1; j < P; ++j) {{ total += simd_shuffle_xor(sum{i}, j); }} "
        f"if (part == 0) {{ out[pair * ROWS + {k} * rg + {i}] = static_cast<OutT>(total); }} }} " for i in range(k))
    return f"""
    uint part = thread_position_in_grid.x % P;
    uint rg = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + {k} * rg) * ROW_BYTES;
    auto x_ = (const device half4*)(x + (pair / SHARE) * COLS);
    {sums}
    for (uint n = part; n < RUNS; n += P) {{
        auto xr = x_ + 8 * n;
        float4 x0 = float4(xr[0]), x1 = float4(xr[1]), x2 = float4(xr[2]), x3 = float4(xr[3]);
        float4 x4 = float4(xr[4]), x5 = float4(xr[5]), x6 = float4(xr[6]), x7 = float4(xr[7]);
        {rows}
    }}
    {outs}
"""


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
for part, share_tokens, threads in (("gate_proj", True, (16, 32)), ("down_proj", False, (4, 16))):
    lins = [getattr(l.mlp.switch_mlp, part) for l in layers]
    rows, cols = lins[0].output_dims, lins[0].input_dims
    one = lins[0].blocks.nbytes // lins[0].num_experts
    for T in args.tokens:
        pairs = T * k
        share = k if share_tokens else 1
        idx = mx.array(np.concatenate([rng.choice(512, k, replace=False) for _ in range(T)]).astype(np.uint32))
        xs = [mx.random.normal((pairs // share, cols)).astype(mx.float16) for _ in lins]
        mx.eval(idx, xs)
        want = None
        for K in (0, 2, 4, 8):
            for P in threads:
                header = blocks._header("Q2_0", rows, cols, P=P, SHARE=share) + HEADER
                kern = mx.fast.metal_kernel(name=f"e11_{K}_{part}_{P}", input_names=["blocks", "idx", "x"],
                                            output_names=["out"], source=source(K) if K else blocks._MATVEC, header=header)
                grid = (P * rows // max(K, 1), pairs, 1)

                def run():
                    return [kern(inputs=[lin.blocks, idx, x], template=[("OutT", mx.float16)], grid=grid,
                                 threadgroup=(32, 1, 1), output_shapes=[(pairs, rows)], output_dtypes=[mx.float16])[0]
                            for lin, x in zip(lins, xs)]
                got = run()[0].astype(mx.float32)
                mx.eval(got)
                if want is None:
                    want = got
                err = float(mx.abs(got - want).max() / mx.abs(want).max())
                ms = best(run, args.rounds)
                name = f"{K} rows a thread" if K else "now"
                rows_out.append({"tensor": part, "tokens": T, "kernel": name, "threads": P, "ms": round(ms, 2), "rel_err": err})
                print(f"{part:9} {rows}x{cols}  {T} tokens  {name:16} {P:2} threads  {ms:6.2f} ms  "
                      f"{len(lins) * pairs * one / ms / 1e6:4.0f} GB/s  differs by {err:.1e}", flush=True)
if args.json:
    Path(args.json).write_text(json.dumps(rows_out, indent=1) + "\n")
