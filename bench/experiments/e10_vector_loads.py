"""Experiment 10: is it the number of loads that an expert kernel pays for?

Experiment 3's three kernels fit one line: 1.55 ms + 0.044 ms for every load a run of 32 weights makes (8 code
bytes and 32 activations now: 40; a 16-entry table: 24; codes only: 8). If so, loading four values at once should
count as one load: the codes of a run as one `packed_ushort4`, the activations as eight `half4`: 9 loads.

    now      blocks.py's kernel: a byte and four activations at a time, through a local array
    vec      one load for a run's codes, eight for its activations, no local array
    vec, x in float32   the same with the activations converted to float32 first (half as many conversions)

Over the real model's 48 layers in one evaluation, as experiment 3.

    uv run python bench/experiments/e10_vector_loads.py --model <shard 1>
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

VEC = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    auto x_ = (const device XV*)(x + (pair / SHARE) * COLS);
    float sum = 0.0f;
    for (uint n = part; n < RUNS; n += P) {
        const device ushort* b = (const device ushort*)(row + (n / 2) * 18);
        ushort4 c4 = *(const device packed_ushort4*)(b + 1 + 4 * (n % 2));
        auto xr = x_ + 8 * n;
        float s = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            uint c = c4[i];
            float4 lo = float4(float(c & 3), float((c >> 2) & 3), float((c >> 4) & 3), float((c >> 6) & 3)) - 1.0f;
            float4 hi = float4(float((c >> 8) & 3), float((c >> 10) & 3), float((c >> 12) & 3), float((c >> 14) & 3)) - 1.0f;
            s += dot(lo, float4(xr[2 * i])) + dot(hi, float4(xr[2 * i + 1]));
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
for part, share_tokens, threads in (("gate_proj", True, (4, 16)), ("down_proj", False, (1, 4))):
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
        for name, xdtype, xv in (("now", mx.float16, None), ("vec", mx.float16, "half4"), ("vec, x in float32", mx.float32, "float4")):
            for P in threads:
                header = blocks._header("Q2_0", rows, cols, P=P, SHARE=share)
                if xv:
                    header += f"typedef {xv} XV;\n"
                kern = mx.fast.metal_kernel(name=f"e10_{'now' if xv is None else xv}_{part}_{P}", input_names=["blocks", "idx", "x"],
                                            output_names=["out"], source=blocks._MATVEC if xv is None else VEC, header=header)

                def run():
                    return [kern(inputs=[lin.blocks, idx, x.astype(xdtype)], template=[("OutT", mx.float16)],
                                 grid=(P * rows, pairs, 1), threadgroup=(32, 1, 1), output_shapes=[(pairs, rows)],
                                 output_dtypes=[mx.float16])[0] for lin, x in zip(lins, xs)]
                got = run()[0].astype(mx.float32)
                mx.eval(got)
                if want is None:
                    want = got
                err = float(mx.abs(got - want).max() / mx.abs(want).max())
                ms = best(run, args.rounds)
                rows_out.append({"tensor": part, "tokens": T, "kernel": name, "threads_per_row": P, "ms": round(ms, 2), "rel_err": err})
                print(f"{part:9} {rows}x{cols}  {T} tokens  {name:18} {P:2} threads a row  {ms:6.2f} ms  "
                      f"{len(lins) * pairs * one / ms / 1e6:4.0f} GB/s  differs by {err:.1e}", flush=True)
if args.json:
    Path(args.json).write_text(json.dumps(rows_out, indent=1) + "\n")
