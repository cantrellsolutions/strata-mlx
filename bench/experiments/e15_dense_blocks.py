"""Experiment 15: a dense projection read as the file's Q3_K blocks, against MLX's product on its 8-bit form.

MLX has no 3-bit-with-two-scales form, so a Q3_K tensor is held at 8 bits a weight (9 with its scales): 2.6 times
the file's bytes, and the delta-net and attention projections are mostly Q3_K and IQ4_XS (which is in the same
position): 1.3 GB a token more than the file has. A kernel on the blocks themselves reads 3.44 bits a weight;
it has to undo the format's interleaving for every weight, once for all the tokens of a window.

A thread takes 32 weights (eight bytes of two-bit codes, eight of high bits, their four scales) and every token;
the threads of a row meet through threadgroup memory. Chains of the real model's 18 Q3_K `attn_qkv` tensors,
each product waiting for the one before.

Result: no gain. MLX's product is flat from one token to four and reads its 29.5 MB at 370-400 GB/s (77, 78,
81 us); this kernel takes 74, 80, 91 us for the file's 11.3 MB. It is not bytes that bound it but its million
threads and what each must do to undo the format. A first run made it look a third faster at four tokens: the
MLX layers there still had float32 scales, which costs them half their speed.

    uv run python bench/experiments/e15_dense_blocks.py --model <shard 1>
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
// the k-th of a Q3_K block's sixteen 6-bit scales, less 32
inline float q3k_scale(const device uchar* s, uint k) {
    uint p = k % 4, word = k / 4;
    uint low = word < 2 ? (uint(s[4 * word + p]) & 15) : (uint(s[4 * (word - 2) + p]) >> 4);
    uint high = (uint(s[8 + p]) >> (2 * word)) & 3;
    return float(int(low | (high << 4)) - 32);
}
"""


def source(tokens: int) -> str:
    """One thread: block b, half h, OCT eighths of its 32 byte positions -> 32 * OCT weights, for every token."""
    accs = "".join(f"float a{t} = 0.0f; " for t in range(tokens))
    adds = "".join(f"a{t} += w * float(x[{t} * COLS + at]); " for t in range(tokens))
    outs = "".join(
        f"{{ float s = simd_sum(a{t}); if (lane % 32 == 0) {{ shared[(lane / 32) * T + {t}] = s; }} }} " for t in range(tokens))
    return f"""
    uint lane = thread_position_in_threadgroup.x;
    uint r = thread_position_in_grid.x / UNITS;
    threadgroup float shared[G * T];
    {accs}
    uint per_half = 4 / OCT;
    uint b = lane / (2 * per_half), h = (lane / per_half) % 2, o0 = (lane % per_half) * OCT;
    const device uchar* blk = blocks + r * ROW_BYTES + b * 110;
    float d = half_at(blk + 108);
    for (uint o = o0; o < o0 + OCT; ++o) {{
        const device uchar* hm = blk + 8 * o;
        const device uchar* qs = blk + 32 + 32 * h + 8 * o;
        for (uint j = 0; j < 4; ++j) {{
            float dl = d * q3k_scale(blk + 96, 8 * h + 2 * j + (o >> 1));
            uint base = 256 * b + 128 * h + 32 * j + 8 * o;
            for (uint i = 0; i < 8; ++i) {{
                float w = dl * (float((uint(qs[i]) >> (2 * j)) & 3) - (((uint(hm[i]) >> (4 * h + j)) & 1) ? 0.0f : 4.0f));
                uint at = base + i;
                {adds}
            }}
        }}
    }}
    {outs}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < T) {{
        float total = 0.0f;
        for (uint g = 0; g < G; ++g) {{ total += shared[g * T + lane]; }}
        out[lane * ROWS + r] = static_cast<OutT>(total + 0.0f * float(prev[0]));
    }}
"""


_kernels = {}


def q3k_matvec(blocks: mx.array, rows: int, cols: int, x: mx.array, prev: mx.array, octets: int = 1) -> mx.array:
    """x (T, cols) -> (T, rows); `prev` only makes this product wait for another. A thread takes 32 * octets
    weights; the threads of a row are one threadgroup."""
    T = x.shape[0]
    units = cols // 32 // octets
    groups = -(-units // 32)
    key = (rows, cols, T, octets)
    if key not in _kernels:
        header = HEADER + (f"constant uint ROWS = {rows};\nconstant uint COLS = {cols};\nconstant uint ROW_BYTES = {cols // 256 * 110};\n"
                           f"constant uint UNITS = {units};\nconstant uint G = {groups};\nconstant uint T = {T};\n"
                           f"constant uint OCT = {octets};\n")
        _kernels[key] = mx.fast.metal_kernel(name=f"q3k_mv_{T}_{octets}", input_names=["blocks", "x", "prev"], output_names=["out"],
                                             source=source(T), header=header)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--json")
    args = p.parse_args()
    g = GGUF(args.model)
    tensors = [t for t in g.tensors.values() if t.name.endswith("attn_qkv.weight") and t.type == "Q3_K"]
    rows, cols = tensors[0].shape
    raw = [mx.array(np.ascontiguousarray(t.data)) for t in tensors]
    own = [linear(t) for t in tensors]
    for m in own:
        m.set_dtype(mx.float16)  # as in the engine: with float32 scales MLX's product took 134 us for four tokens, not 81
    mx.eval(raw, [m.parameters() for m in own])
    file_mb, arrays_mb = tensors[0].data.nbytes / 1e6, sum(a.nbytes for a in (own[0].weight, own[0].scales, own[0].biases)) / 1e6
    print(f"{len(tensors)} tensors of {rows} x {cols}: {file_mb:.1f} MB each in the file, {arrays_mb:.1f} MB as MLX arrays")

    x = mx.random.normal((2, cols)).astype(mx.float16)
    from gguf import quants
    from gguf.constants import GGMLQuantizationType

    w = quants.dequantize(np.asarray(tensors[0].data).reshape(rows, -1), GGMLQuantizationType.Q3_K).reshape(rows, cols)
    want = np.array(x.astype(mx.float32)) @ w.T
    got = np.array(q3k_matvec(raw[0], rows, cols, x, mx.zeros((1,), dtype=mx.float16)).astype(mx.float32))
    print(f"against the dequantized weights: differs by at most {np.abs(got - want).max() / np.abs(want).max():.1e} (float16 output)")

    out_rows = []
    for T in (1, 2, 3, 4, 6, 8):
        x = mx.random.normal((T, cols)).astype(mx.float16)
        mx.eval(x)

        def mlx_chain():
            prev = None
            for m in own:
                prev = m(x if prev is None else x + 0.0 * prev[:, :1])
            return prev

        def block_chain(octets):
            def chain():
                prev = mx.zeros((1,), dtype=mx.float16)
                for b in raw:
                    prev = q3k_matvec(b, rows, cols, x, prev, octets)
                return prev
            return chain
        a = best(mlx_chain, len(own))
        ours = {32 * o: best(block_chain(o), len(raw)) for o in (1, 2, 4)}
        out_rows.append({"tokens": T, "mlx_us": round(a, 1), **{f"blocks_{k}_us": round(v, 1) for k, v in ours.items()}})
        print(f"{T} tokens: MLX on 8 bits (and an add) {a:6.1f} us   on the blocks, weights to a thread: "
              + "   ".join(f"{k}: {v:6.1f} us" for k, v in ours.items()), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out_rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
