"""e61: window kernels for Q3_K and IQ4_XS dense tensors held at the file's bits, against MLX's 8-bit product.

The 18 Q3_K and the 13 IQ4_XS attn_qkv tensors (10,240 x 2,560) of the Q2_0 file, each type chained (each one's input waits on the one
before's output), for windows of 1, 2, 4 and 8 tokens:

  mlx8      MLX's 8-bit form, as the engine holds them now (scales float16)
  window    the file's blocks through blocks.window_matvec (e36's kernel)
  planes-P  the blocks rearranged once into planes, the same bits of weight: the high-bit masks, the 2-bit codes
            and the 16 scales of a block each in an aligned array of their own, so a thread reads them as words
            (IQ4_XS: the 4-bit codes, and the eight scales less 32; its 16 values in threadgroup memory).
            A thread takes half a block (128 weights) at a time, P threads to a row.

Shuffled in one process, the lower quartile; each kernel's result checked against the weights unpacked.

    uv run python bench/experiments/e61_q3k_window_kernel.py --model <Q2_0 shard 1> --json out.json
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_mlx import blocks  # noqa: E402
from strata_mlx.gguf import GGUF  # noqa: E402
from strata_mlx.load import AS_BLOCKS, linear  # noqa: E402

_PLANES_Q3K = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    float sum[T];
    for (uint t = 0; t < T; ++t) sum[t] = 0.0f;
    for (uint h = part; h < 2 * BLOCKS; h += P) {
        uint blk = r * BLOCKS + h / 2, n = h % 2;
        const device uint* hmw = hm + blk * 8;           // 32 bytes of high bits
        const device uint* qsw = qs + blk * 16 + 8 * n;  // this half's 32 bytes of codes
        const device int8_t* scl = sc + blk * 16;
        float d = float(dd[blk]);
        uint hwords[8], qwords[8];
        for (uint w = 0; w < 8; ++w) { hwords[w] = hmw[w]; qwords[w] = qsw[w]; }
        uint base = (h / 2) * 256 + 128 * n;
        for (uint j = 0; j < 4; ++j) {
            uint rr = 4 * n + j;
            float acc0[T], acc1[T];
            for (uint t = 0; t < T; ++t) { acc0[t] = 0.0f; acc1[t] = 0.0f; }
            for (uint w4 = 0; w4 < 8; ++w4) {
                uint qw = qwords[w4], hw = hwords[w4];
                float4 v;
                for (uint k = 0; k < 4; ++k) {
                    v[k] = float(int((qw >> (8 * k + 2 * j)) & 3) - (((hw >> (8 * k + rr)) & 1) ? 0 : 4));
                }
                uint xi = base + 32 * j + 4 * w4;
                for (uint t = 0; t < T; ++t) {
                    float4 xx = float4(*(const device half4*)(x + t * COLS + xi));
                    if (w4 < 4) acc0[t] += dot(v, xx); else acc1[t] += dot(v, xx);
                }
            }
            float s0 = d * float(scl[2 * rr]), s1 = d * float(scl[2 * rr + 1]);
            for (uint t = 0; t < T; ++t) sum[t] += s0 * acc0[t] + s1 * acc1[t];
        }
    }
    for (uint t = 0; t < T; ++t) {
        float total = sum[t];
        for (ushort k = 1; k < P; k <<= 1) total += simd_shuffle_xor(total, k);
        if (part == 0) out[t * ROWS + r] = static_cast<OutT>(total);
    }
"""


_PLANES_IQ4XS = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    threadgroup float kv[16];
    if (thread_position_in_threadgroup.x < 16) kv[thread_position_in_threadgroup.x] = float(KV[thread_position_in_threadgroup.x]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum[T];
    for (uint t = 0; t < T; ++t) sum[t] = 0.0f;
    for (uint h = part; h < 2 * BLOCKS; h += P) {
        uint blk = r * BLOCKS + h / 2, n = h % 2;
        const device uint* qsw = qs + blk * 32 + 16 * n;  // this half's 64 bytes: four runs of 32 weights
        const device int8_t* scl = sc + blk * 8 + 4 * n;
        float d = float(dd[blk]);
        uint base = (h / 2) * 256 + 128 * n;
        for (uint ib = 0; ib < 4; ++ib) {
            float acc[T];
            for (uint t = 0; t < T; ++t) acc[t] = 0.0f;
            for (uint w = 0; w < 4; ++w) {
                uint word = qsw[4 * ib + w];
                float4 lo, hi;
                for (uint k = 0; k < 4; ++k) {
                    lo[k] = kv[(word >> (8 * k)) & 15];
                    hi[k] = kv[(word >> (8 * k + 4)) & 15];
                }
                uint xi = base + 32 * ib + 4 * w;
                for (uint t = 0; t < T; ++t) {
                    acc[t] += dot(lo, float4(*(const device half4*)(x + t * COLS + xi)))
                            + dot(hi, float4(*(const device half4*)(x + t * COLS + xi + 16)));
                }
            }
            float s = d * float(scl[ib]);
            for (uint t = 0; t < T; ++t) sum[t] += s * acc[t];
        }
    }
    for (uint t = 0; t < T; ++t) {
        float total = sum[t];
        for (ushort k = 1; k < P; k <<= 1) total += simd_shuffle_xor(total, k);
        if (part == 0) out[t * ROWS + r] = static_cast<OutT>(total);
    }
"""


def iq4xs_planes(raw: np.ndarray, rows: int, cols: int) -> dict:
    b = raw.reshape(rows, cols // 256, 136)
    sh = b[..., 2].astype(np.uint16) | (b[..., 3].astype(np.uint16) << 8)
    sc = np.empty(b.shape[:-1] + (8,), dtype=np.int16)
    for ib in range(8):
        low = (b[..., 4 + ib // 2] >> (4 * (ib % 2))) & 15
        sc[..., ib] = (low.astype(np.int16) | (((sh >> (2 * ib)) & 3).astype(np.int16) << 4)) - 32
    return {"qs": mx.array(np.ascontiguousarray(b[..., 8:136]).reshape(-1).view(np.uint32)),
            "sc": mx.array(sc.astype(np.int8).reshape(-1)),
            "dd": mx.array(np.ascontiguousarray(b[..., 0:2]).reshape(-1).view(np.float16))}


def q3k_scales(sc12: np.ndarray) -> np.ndarray:
    """(..., 12) bytes of a Q3_K block's scales -> (..., 16) int8, each less 32, as blocks.q3k_scale gives."""
    out = np.empty(sc12.shape[:-1] + (16,), dtype=np.int16)
    for k in range(16):
        p, word = k % 4, k // 4
        low = (sc12[..., 4 * word + p] & 15) if word < 2 else (sc12[..., 4 * (word - 2) + p] >> 4)
        high = (sc12[..., 8 + p] >> (2 * word)) & 3
        out[..., k] = (low.astype(np.int16) | (high.astype(np.int16) << 4)) - 32
    return out.astype(np.int8)


def planes(raw: np.ndarray, rows: int, cols: int) -> dict:
    b = raw.reshape(rows, cols // 256, 110)
    return {"hm": mx.array(np.ascontiguousarray(b[..., 0:32]).reshape(-1).view(np.uint32)),
            "qs": mx.array(np.ascontiguousarray(b[..., 32:96]).reshape(-1).view(np.uint32)),
            "sc": mx.array(q3k_scales(b[..., 96:108]).reshape(-1)),
            "dd": mx.array(np.ascontiguousarray(b[..., 108:110]).reshape(-1).view(np.float16))}


_kernels = {}


def planes_matvec(pl, rows, cols, x, per_row, kind="Q3_K"):
    T = x.shape[0]
    key = (kind, rows, cols, T, per_row)
    names = ["hm", "qs", "sc", "dd", "x"] if kind == "Q3_K" else ["qs", "sc", "dd", "x"]
    if key not in _kernels:
        header = "".join(f"constant uint {k} = {v};\n" for k, v in
                         {"ROWS": rows, "COLS": cols, "BLOCKS": cols // 256, "P": per_row, "T": T}.items())
        if kind != "Q3_K":
            from strata_mlx.gguf import KVALUES_IQ4NL
            header += "constant char KV[16] = {" + ",".join(str(int(v)) for v in KVALUES_IQ4NL) + "};\n"
        _kernels[key] = mx.fast.metal_kernel(name=f"{kind.lower()}_planes_{T}", input_names=names,
                                             output_names=["out"], header=header,
                                             source=_PLANES_Q3K if kind == "Q3_K" else _PLANES_IQ4XS)
    return _kernels[key](inputs=[*(pl[n] for n in names[:-1]), x], template=[("OutT", x.dtype)],
                         grid=(per_row * rows, 1, 1), threadgroup=(64, 1, 1),
                         output_shapes=[(T, rows)], output_dtypes=[x.dtype])[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--json")
    args = p.parse_args()
    g = GGUF(args.model)
    rows, cols = 10240, 2560
    result = []
    for kind in ("Q3_K", "IQ4_XS"):
        result += run_kind(g, kind, rows, cols, args.rounds)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")


def run_kind(g, kind, rows, cols, rounds):
    names = [n for n, t in g.tensors.items() if t.type == kind and tuple(t.shape) == (rows, cols)]
    layers = []
    for n in names:
        t = g[n]
        q = linear(t)
        q.set_dtype(mx.float16)
        b = linear(t, as_blocks=AS_BLOCKS)
        raw = np.asarray(t.data).reshape(-1)
        pl = planes(raw, rows, cols) if kind == "Q3_K" else iq4xs_planes(raw, rows, cols)
        mx.eval(q.parameters(), b.blocks, list(pl.values()))
        layers.append((q, b, pl))
    print(f"{len(layers)} tensors of {rows} x {cols} {kind}", flush=True)

    variants = {"mlx8": lambda L, x: L[0](x), "window": lambda L, x: L[1](x)}
    for P in (2, 4):
        variants[f"planes-{P}"] = (lambda P: lambda L, x: planes_matvec(L[2], rows, cols, x, P, kind))(P)

    def chain(fn, x0):
        x = x0
        for L in layers:
            out = fn(L, x)
            x = x + (out[:, :1] * 0).astype(x.dtype)
        return x

    rng = np.random.default_rng(61)
    result = []
    for T in (1, 2, 4, 8):
        x0 = mx.array(rng.standard_normal((T, cols)).astype(np.float16))
        ref = x0.astype(mx.float32) @ blocks.dequantize(layers[0][1].blocks, kind, (rows, cols)).T
        for name, fn in variants.items():
            got = fn(layers[0], x0).astype(mx.float32)
            err = (mx.max(mx.abs(got - ref)) / mx.max(mx.abs(ref))).item()
            if T == 1:
                print(f"  {name:<9} against the weights unpacked: largest difference {err:.1e} of the largest value")
        for fn in variants.values():
            mx.eval(chain(fn, x0))
        times = {name: [] for name in variants}
        order = list(variants) * rounds
        random.Random(T).shuffle(order)
        for name in order:
            t0 = time.perf_counter()
            mx.eval(chain(variants[name], x0))
            times[name].append(time.perf_counter() - t0)
        line = f"{T} token(s):"
        for name, ts in times.items():
            us = statistics.quantiles(ts, n=4)[0] / len(layers) * 1e6
            result.append({"kind": kind, "tokens": T, "variant": name, "us_per_tensor": round(us, 1)})
            line += f"  {name} {us:6.1f}"
        print(line + "  us a tensor", flush=True)
    floor8 = layers[0][0].weight.nbytes / 490e9 * 1e6
    floorb = layers[0][1].blocks.nbytes / 490e9 * 1e6
    print(f"bytes at 490 GB/s: 8-bit {floor8:.1f} us, the file's blocks {floorb:.1f} us a tensor")
    return result


if __name__ == "__main__":
    main()
