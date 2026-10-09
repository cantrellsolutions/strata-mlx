"""Experiment 3: Q2_0 experts by table lookup instead of a multiplication for every weight.

Experiment 2: the expert kernel does about 1 T weights a second whatever its inner loop and thread count, so it
is the work per weight that bounds it, not the bytes. A Q2_0 byte holds four weights, each -1, 0, 1 or 2 times
the block's scale: for four inputs there are 256 possible sums. So: for every token, a table of those sums for
each four inputs (one small matrix product), then a row is one lookup and one addition for every byte.

    now      blocks.py's kernel
    bytes    only adds the code bytes up: what the kernel costs with no arithmetic on the weights at all
    lut8     a table of 256 sums for every 4 inputs, float32 or float16
    lut4     a table of 16 sums for every 2 inputs (a tenth of the size, two lookups a byte)

Each over the real model's 48 layers in one evaluation; the tables' cost is included.

    uv run python bench/experiments/e03_expert_lookup.py --model <shard 1>
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

HEAD = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    float sum = 0.0f;
    for (uint n = part; n < RUNS; n += P) {
        const device uchar* b = row + (n / 2) * 18;
        const device uchar* q = b + 2 + 8 * (n % 2);
        float s = 0.0f;
"""
TAIL = """
        sum += half_at(b) * s;
    }
    float total = sum;
    for (ushort k = 1; k < P; ++k) {
        total += simd_shuffle_xor(sum, k);
    }
    if (part == 0) {
        out[pair * ROWS + r] = static_cast<OutT>(total);
    }
"""
BODY = {
    "bytes": "for (uint i = 0; i < 8; ++i) { s += float(q[i]); } s += float(x[0]) * 0.0f;",
    # x: (tokens or pairs, COLS / 4, 256)
    "lut8": "auto t = x + ((pair / SHARE) * (COLS / 4) + 8 * n) * 256;"
            "for (uint i = 0; i < 8; ++i) { s += float(t[256 * i + q[i]]); }",
    # x: (tokens or pairs, COLS / 2, 16)
    "lut4": "auto t = x + ((pair / SHARE) * (COLS / 2) + 16 * n) * 16;"
            "for (uint i = 0; i < 8; ++i) { uint c = q[i]; s += float(t[32 * i + (c & 15)]) + float(t[32 * i + 16 + (c >> 4)]); }",
}


def codes(bits_each: int) -> mx.array:
    """(values of a group of codes, weights in the group): each weight's code - 1."""
    n = bits_each // 2
    v = np.arange(1 << bits_each)[:, None]
    return mx.array((((v >> (2 * np.arange(n))) & 3) - 1).astype(np.float32))


C8, C4 = codes(8), codes(4)


def table(x: mx.array, kind: str, dtype) -> mx.array:
    """x (n, cols) -> the sums of every group of inputs under every value of its codes."""
    c = C8 if kind == "lut8" else C4
    return (x.astype(mx.float32).reshape(x.shape[0], -1, c.shape[1]) @ c.T).astype(dtype)


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
variants = [("now", None, None), ("bytes", None, None), ("lut8", mx.float32, None), ("lut8", mx.float16, None),
            ("lut4", mx.float32, None), ("lut4", mx.float16, None)]
for part, share_tokens, threads in (("gate_proj", True, (4, 16)), ("down_proj", False, (1, 4))):
    lins = [getattr(l.mlp.switch_mlp, part) for l in layers]
    rows, cols = lins[0].output_dims, lins[0].input_dims
    one = lins[0].blocks.nbytes // lins[0].num_experts
    for T in args.tokens:
        pairs = T * k
        share = k if share_tokens else 1
        idx = mx.array(np.concatenate([rng.choice(512, k, replace=False) for _ in range(T)]).astype(np.uint32))
        xs = [mx.random.normal((pairs // share, cols)).astype(mx.float16) for _ in lins]  # every layer its own input
        mx.eval(idx, xs)
        want = None
        for name, tdtype, _ in variants:
            for P in threads:
                source = blocks._MATVEC if name == "now" else HEAD + BODY[name] + TAIL
                kern = mx.fast.metal_kernel(name=f"e03_{name}_{part}_{P}", input_names=["blocks", "idx", "x"],
                                            output_names=["out"], source=source,
                                            header=blocks._header("Q2_0", rows, cols, P=P, SHARE=share))

                def run():
                    outs = []
                    for lin, x in zip(lins, xs):
                        a = table(x, name, tdtype) if name.startswith("lut") else x
                        outs.append(kern(inputs=[lin.blocks, idx, a], template=[("OutT", mx.float16)],
                                         grid=(P * rows, pairs, 1), threadgroup=(32, 1, 1), output_shapes=[(pairs, rows)],
                                         output_dtypes=[mx.float16])[0])
                    return outs
                got = run()[0].astype(mx.float32)
                mx.eval(got)
                if want is None:
                    want = got
                err = float(mx.abs(got - want).max() / mx.abs(want).max()) if name != "bytes" else float("nan")
                ms = best(run, args.rounds)
                label = name + ("" if tdtype is None else " " + str(tdtype).split(".")[-1])
                rows_out.append({"tensor": part, "tokens": T, "kernel": label, "threads_per_row": P, "ms": round(ms, 2),
                                 "rel_err": err})
                print(f"{part:9} {rows}x{cols}  {T} tokens  {label:13} {P:2} threads a row  {ms:6.2f} ms  "
                      f"{len(lins) * pairs * one / ms / 1e6:4.0f} GB/s  differs by {err:.1e}", flush=True)
if args.json:
    Path(args.json).write_text(json.dumps(rows_out, indent=1) + "\n")
