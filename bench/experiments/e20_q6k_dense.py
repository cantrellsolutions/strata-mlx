"""Experiment 20: a dense projection read as the file's Q6_K blocks, against MLX's product on float16.

The IQ3_S file holds most of its dense weights as Q6_K (1.88 GB: the delta-net and attention projections and the
output layer). Q6_K scales every 16 weights, MLX's quantized form every 32 at least, so the loader unpacks them
to floats: 4.58 GB as float16, read by every token, and the output layer's 1.27 GB again by every guess of the
draft head. A kernel on the blocks reads 6.56 bits a weight, once for all the tokens of a window.

A thread takes 64, 128 or 256 weights of a row (one, two or four runs of sixteen byte positions, each position
four weights under four scales) and every token; the threads of a row are one threadgroup. Chains of the real
model's Q6_K `attn_qkv` tensors, each product waiting for the one before, and the output layer by itself.

Result: MLX's product over float16 takes 130 us for a 10,240 x 2,560 tensor whatever the tokens (it reads its
52 MB at 400 GB/s) and 2.5 ms for the output layer. On the blocks, 64 weights to a thread: 88 / 85 / 109 / 134
us for 1 / 2 / 3 / 4 tokens, then 335 and 433 for six and eight, where the sums no longer fit the registers;
the output layer 1.2 / 1.4 / 2.2 / 2.8 ms. Faster for one and two tokens, not for a window. (MLX takes no scale
to 16 weights: `quantize` refuses the group size and the product has no kernel for it.)

    uv run python bench/experiments/e20_q6k_dense.py --model <IQ3_S shard 1>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.gguf import GGUF  # noqa: E402
from strata_mlx.load import linear  # noqa: E402

HEADER = """
inline float half_at(const device uchar* b) {
    return float(as_type<half>(ushort(uint(b[0]) | (uint(b[1]) << 8))));
}
inline float s8(uchar v) { return float(int(v) - (int(v & 128) << 1)); }
"""


def source(tokens: int, groups: int) -> str:
    """One thread: SUB runs of a row -> 64 * SUB weights, for every token. A run: block b, half h, sixteen byte
    positions; position i holds the weights at i, i + 32, i + 64, i + 96 of the half, under four scales."""
    T = range(tokens)
    accs = "".join(f"float a{t} = 0.0f; " for t in T)
    parts = "".join(f"float c{t}_{k} = 0.0f; " for t in T for k in range(4))
    adds = "".join(f"c{t}_{k} += w{k} * float(x[{t} * COLS + at + {32 * k}]); " for t in T for k in range(4))
    folds = "".join(f"a{t} += s0 * c{t}_0 + s1 * c{t}_1 + s2 * c{t}_2 + s3 * c{t}_3; " for t in T)
    if groups == 1:
        outs = "".join(f"{{ float s = simd_sum(a{t}); if (lane == 0) {{ out[{t} * ROWS + r] = static_cast<OutT>(s + 0.0f * float(prev[0])); }} }} "
                       for t in T)
        tail = ""
    else:
        outs = "".join(f"{{ float s = simd_sum(a{t}); if (lane % 32 == 0) {{ shared[(lane / 32) * T + {t}] = s; }} }} " for t in T)
        tail = """
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < T) {
        float total = 0.0f;
        for (uint g = 0; g < G; ++g) { total += shared[g * T + lane]; }
        out[lane * ROWS + r] = static_cast<OutT>(total + 0.0f * float(prev[0]));
    }
"""
    return f"""
    uint lane = thread_position_in_threadgroup.x;
    uint r = thread_position_in_grid.x / UNITS;
    {"threadgroup float shared[G * T];" if groups > 1 else ""}
    {accs}
    const device uchar* row = blocks + r * ROW_BYTES;
    for (uint run = lane * SUB; run < (lane + 1) * SUB; ++run) {{
        uint b = run / 4, h = (run / 2) % 2, is = run % 2;
        const device uchar* blk = row + b * 210;
        float d = half_at(blk + 208);
        const device uchar* ql = blk + 64 * h + 16 * is;
        const device uchar* qh = blk + 128 + 32 * h + 16 * is;
        const device uchar* sc = blk + 192 + 8 * h + is;
        float s0 = d * s8(sc[0]), s1 = d * s8(sc[2]), s2 = d * s8(sc[4]), s3 = d * s8(sc[6]);
        uint base = 256 * b + 128 * h + 16 * is;
        {parts}
        for (uint i = 0; i < 16; ++i) {{
            uint lo = ql[i], hi = ql[32 + i], top = qh[i];
            float w0 = float(int((lo & 15) | ((top & 3) << 4)) - 32);
            float w1 = float(int((hi & 15) | (((top >> 2) & 3) << 4)) - 32);
            float w2 = float(int((lo >> 4) | (((top >> 4) & 3) << 4)) - 32);
            float w3 = float(int((hi >> 4) | ((top >> 6) << 4)) - 32);
            uint at = base + i;
            {adds}
        }}
        {folds}
    }}
    {outs}
    {tail}
"""


_kernels = {}


def q6k_matvec(blocks: mx.array, rows: int, cols: int, x: mx.array, prev: mx.array, sub: int = 2) -> mx.array:
    """x (T, cols) -> (T, rows); `prev` only makes this product wait for another. A thread takes 64 * sub
    weights; the threads of a row are one threadgroup."""
    T = x.shape[0]
    units = cols // 64 // sub
    groups = -(-units // 32)
    key = (rows, cols, T, sub)
    if key not in _kernels:
        header = HEADER + (f"constant uint ROWS = {rows};\nconstant uint COLS = {cols};\nconstant uint ROW_BYTES = {cols // 256 * 210};\n"
                           f"constant uint UNITS = {units};\nconstant uint G = {groups};\nconstant uint T = {T};\n"
                           f"constant uint SUB = {sub};\n")
        _kernels[key] = mx.fast.metal_kernel(name=f"q6k_mv_{T}_{sub}", input_names=["blocks", "x", "prev"], output_names=["out"],
                                             source=source(T, groups), header=header)
    return _kernels[key](inputs=[blocks, x, prev], template=[("OutT", x.dtype)], grid=(units * rows, 1, 1),
                         threadgroup=(units, 1, 1), output_shapes=[(T, rows)], output_dtypes=[x.dtype])[0]


def best(build, n, rounds=9):
    out = float("inf")
    for _ in range(rounds):
        g = build()
        t0 = time.perf_counter()
        mx.eval(g)
        out = min(out, time.perf_counter() - t0)
    return out / n * 1e6


def check(t, raw, sub):
    from gguf import quants
    from gguf.constants import GGMLQuantizationType

    rows, cols = t.shape
    some = min(rows, 512)
    w = quants.dequantize(np.asarray(t.data).reshape(rows, -1)[:some], GGMLQuantizationType.Q6_K).reshape(some, cols)
    x = mx.random.normal((3, cols)).astype(mx.float16)
    want = np.array(x.astype(mx.float32)) @ w.T
    got = np.array(q6k_matvec(raw, rows, cols, x, mx.zeros((1,), dtype=mx.float16), sub).astype(mx.float32))[:, :some]
    return np.abs(got - want).max() / np.abs(want).max()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--json")
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    args = p.parse_args()
    g = GGUF(args.model)
    out_rows = []
    for what, tensors, repeat in (
            ("attn_qkv", [t for t in g.tensors.values() if t.name.endswith("attn_qkv.weight") and t.type == "Q6_K"][:12], 1),
            ("output layer", [g["output.weight"]], 3)):
        if not tensors or tensors[0].type != "Q6_K":
            continue
        rows, cols = tensors[0].shape
        raw = [mx.array(np.ascontiguousarray(t.data)) for t in tensors]
        own = [linear(t) for t in tensors]
        for m in own:
            m.set_dtype(mx.float16)
        mx.eval(raw, [m.parameters() for m in own])
        n = len(tensors) * repeat
        print(f"{what}: {len(tensors)} tensors of {rows} x {cols}: {tensors[0].data.nbytes / 1e6:.1f} MB each in the file, "
              f"{own[0].weight.nbytes / 1e6:.1f} MB as float16", flush=True)
        subs = (1, 2, 4)
        print("  against the dequantized weights, largest difference of an output (float16): "
              + ", ".join(f"{64 * s} weights a thread {check(tensors[0], raw[0], s):.1e}" for s in subs), flush=True)
        for T in args.tokens:
            x = mx.random.normal((T, cols)).astype(mx.float16)
            mx.eval(x)

            def mlx_chain():
                prev = None
                for _ in range(repeat):
                    for m in own:
                        prev = m(x if prev is None else x + 0.0 * prev[:, :1])
                return prev

            def block_chain(sub):
                def chain():
                    prev = mx.zeros((1,), dtype=mx.float16)
                    for _ in range(repeat):
                        for b in raw:
                            prev = q6k_matvec(b, rows, cols, x, prev, sub)
                    return prev
                return chain
            a = best(mlx_chain, n)
            ours = {64 * s: best(block_chain(s), n) for s in subs}
            out_rows.append({"what": what, "tokens": T, "mlx_float16_us": round(a, 1), **{f"blocks_{k}_us": round(v, 1) for k, v in ours.items()}})
            print(f"  {T} tokens: MLX on float16 (and an add) {a:8.1f} us   on the blocks, weights to a thread: "
                  + "   ".join(f"{k}: {v:8.1f} us" for k, v in ours.items()), flush=True)
        del raw, own
    if args.json:
        Path(args.json).write_text(json.dumps(out_rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
