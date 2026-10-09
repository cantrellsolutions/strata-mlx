"""Experiment 22: Q6_K as a byte a weight and its own scales, against the blocks (e21) and against float16.

A Q6_K weight is d * scale * (q - 32): d a float16 for 256 weights, the scale a signed byte for 16, q six bits.
The blocks interleave q's bits, and undoing that for every weight is what the kernel of e20 and e21 spends its
time on; from four tokens on it loses to MLX's product over float16, which reads 2.4 times the bytes. Kept as
q - 32 in a signed byte, with the scales and d as the file has them, a weight is the same number at 8.56 bits
instead of 6.56 or 16, and a thread has nothing to undo. The kernel is laid out as MLX's own quantized product:
32 threads share R rows, each takes V weights of each row at a time and goes along the rows, and the
activations come with a position's tokens side by side, loaded once for the R rows.

Result (the machine was not quiet, single numbers are off by a third here and there): four rows and eight
weights, against MLX's 130 us on float16 and e21's blocks: about 80 for one to three tokens (blocks 72 / 83 /
101), 106 for four (106), 93 for six (146), 124 for eight (357). For a 2,560 x 6,144 tensor 53-63 up to four
tokens against MLX's 90 and the blocks' 56-94. What matters is the pass: e23.

    uv run python bench/experiments/e22_q6k_codes.py --model <IQ3_S shard 1>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from e21_q6k_tiles import q6k_matvec as blocks_matvec  # noqa: E402

from strata_mlx.gguf import GGUF  # noqa: E402
from strata_mlx.load import linear  # noqa: E402

VEC = {1: "float", 2: "float2", 3: "float3", 4: "float4"}
PACKED = {1: "float", 2: "packed_float2", 3: "packed_float3", 4: "packed_float4"}


def q6k_codes(data: np.ndarray, rows: int, cols: int):
    """A Q6_K tensor's blocks -> (q - 32 as int8 (rows, cols), scales int8 (rows, cols / 16), d float16 (rows, cols / 256))."""
    blk = np.asarray(data).reshape(-1, 210)
    ql, qh = blk[:, :128].reshape(-1, 2, 2, 32), blk[:, 128:192].reshape(-1, 2, 32)  # (block, half, low/high byte, l)
    q = np.empty((len(blk), 2, 4, 32), dtype=np.uint8)
    q[:, :, 0] = (ql[:, :, 0] & 15) | ((qh & 3) << 4)
    q[:, :, 1] = (ql[:, :, 1] & 15) | (((qh >> 2) & 3) << 4)
    q[:, :, 2] = (ql[:, :, 0] >> 4) | (((qh >> 4) & 3) << 4)
    q[:, :, 3] = (ql[:, :, 1] >> 4) | ((qh >> 6) << 4)
    codes = (q.reshape(rows, cols).astype(np.int16) - 32).astype(np.int8)
    return codes, blk[:, 192:208].copy().view(np.int8).reshape(rows, cols // 16), blk[:, 208:210].copy().view(np.float16).reshape(rows, cols // 256)


def source(tokens: int, R: int, V: int) -> str:
    chunks = [(o, min(4, tokens - o)) for o in range(0, tokens, 4)]
    decl = "".join(f"{VEC[n]} a{r}_{o} = 0.0f; " for r in range(R) for o, n in chunks)
    loads = "".join(f"{VEC[n]} x{v}_{o} = {VEC[n]}(*(const device {PACKED[n]}*)(x + (col + {v}) * T + {o})); "
                    for v in range(V) for o, n in chunks)
    rows = ""
    for r in range(R):
        rows += f"        {{ const device int8_t* c = codes + (R * rr + {r}) * COLS + col;\n"
        for g in range(V // 16 or 1):  # a scale to 16 weights; with V = 8 a thread's weights are half of one
            lo, hi = (16 * g, 16 * g + 16) if V >= 16 else (0, V)
            rows += f"          {{ float s = float(d[(R * rr + {r}) * (COLS / 256) + col / 256]) * float(sc[(R * rr + {r}) * (COLS / 16) + (col + {lo}) / 16]);\n"
            for o, n in chunks:
                rows += f"            a{r}_{o} += s * (" + " + ".join(f"float(c[{v}]) * x{v}_{o}" for v in range(lo, hi)) + ");\n"
            rows += "          }\n"
        rows += "        }\n"
    outs = ""
    for r in range(R):
        for o, n in chunks:
            outs += f"    {{ {VEC[n]} s = simd_sum(a{r}_{o}); if (lane == 0) {{ "
            outs += " ".join(f"out[{o + j} * ROWS + R * rr + {r}] = static_cast<OutT>({'s' if n == 1 else f's[{j}]'} + 0.0f * float(prev[0]));"
                             for j in range(n))
            outs += " } }\n"
    return f"""
    uint lane = thread_position_in_grid.x % 32;
    uint rr = thread_position_in_grid.x / 32;
    {decl}
    for (uint col = lane * V; col < COLS; col += 32 * V) {{
        {loads}
{rows}
    }}
{outs}
"""


_kernels = {}


def codes_matvec(codes, sc, d, xt: mx.array, prev: mx.array, R: int = 4, V: int = 8, dtype=mx.float16) -> mx.array:
    """xt (cols, T) float32 -> (T, rows) of `dtype`; `prev` only makes this product wait for another."""
    rows, cols = codes.shape
    T = xt.shape[1]
    assert cols % (32 * V) == 0 and rows % R == 0
    key = (rows, cols, T, R, V)
    if key not in _kernels:
        header = (f"constant uint ROWS = {rows};\nconstant uint COLS = {cols};\nconstant uint T = {T};\n"
                  f"constant uint R = {R};\nconstant uint V = {V};\n")
        _kernels[key] = mx.fast.metal_kernel(name=f"q6k_codes_{T}_{R}_{V}", input_names=["codes", "sc", "d", "x", "prev"],
                                             output_names=["out"], source=source(T, R, V), header=header)
    return _kernels[key](inputs=[codes, sc, d, xt, prev], template=[("OutT", dtype)], grid=(32 * rows // R, 1, 1),
                         threadgroup=(32, 1, 1), output_shapes=[(T, rows)], output_dtypes=[dtype])[0]


def best(build, n, rounds=9):
    out = float("inf")
    for _ in range(rounds):
        g = build()
        t0 = time.perf_counter()
        mx.eval(g)
        out = min(out, time.perf_counter() - t0)
    return out / n * 1e6


VARIANTS = {"4 rows, 8 w": dict(R=4, V=8), "4 rows, 16 w": dict(R=4, V=16), "2 rows, 16 w": dict(R=2, V=16),
            "8 rows, 8 w": dict(R=8, V=8), "1 row, 16 w": dict(R=1, V=16)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--json")
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--what", nargs="+", default=["attn_qkv", "ssm_out", "output layer"])
    args = p.parse_args()
    g = GGUF(args.model)
    from gguf import quants
    from gguf.constants import GGMLQuantizationType

    out_rows = []
    for what, tensors, repeat in (
            ("attn_qkv", [t for t in g.tensors.values() if t.name.endswith("attn_qkv.weight") and t.type == "Q6_K"][:12], 1),
            ("ssm_out", [t for t in g.tensors.values() if t.name.endswith("ssm_out.weight") and t.type == "Q6_K"][:12], 1),
            ("output layer", [g["output.weight"]], 3)):
        if what not in args.what:
            continue
        rows, cols = tensors[0].shape
        raw = [mx.array(np.ascontiguousarray(t.data)) for t in tensors]
        own = [linear(t) for t in tensors]
        for m in own:
            m.set_dtype(mx.float16)
        ours = [tuple(mx.array(a) for a in q6k_codes(t.data, rows, cols)) for t in tensors]
        mx.eval(raw, ours, [m.parameters() for m in own])
        n = len(tensors) * repeat
        mb = lambda arrays: sum(a.nbytes for a in arrays) / 1e6
        print(f"{what}: {len(tensors)} tensors of {rows} x {cols}: {mb([raw[0]]):.1f} MB in the file, {mb(ours[0]):.1f} as bytes "
              f"and scales, {mb([own[0].weight]):.1f} as float16", flush=True)
        some = min(rows, 512)
        w = quants.dequantize(np.asarray(tensors[0].data).reshape(rows, -1)[:some], GGMLQuantizationType.Q6_K).reshape(some, cols)
        errs = []
        for name, how in VARIANTS.items():
            worst = 0.0
            for T in (1, 3, 4, 7, 8):
                x = mx.random.normal((T, cols)).astype(mx.float16)
                want = np.array(x.astype(mx.float32)) @ w.T
                got = np.array(codes_matvec(*ours[0], mx.contiguous(x.astype(mx.float32).T), mx.zeros((1,), dtype=mx.float16),
                                            dtype=mx.float32, **how))[:, :some]
                worst = max(worst, np.abs(got - want).max() / np.abs(want).max())
            errs.append(f"{name}: {worst:.1e}")
        print("  against the dequantized weights (float32 output): " + ", ".join(errs), flush=True)
        for T in args.tokens:
            x = mx.random.normal((T, cols)).astype(mx.float16)
            mx.eval(x)

            def mlx_chain():
                prev = None
                for _ in range(repeat):
                    for m in own:
                        prev = m(x if prev is None else x + 0.0 * prev[:, :1])
                return prev

            def chain(fn):
                def build():
                    prev = mx.zeros((1,), dtype=mx.float16)
                    for _ in range(repeat):
                        for i in range(len(tensors)):
                            prev = fn(i, mx.contiguous(x.astype(mx.float32).T), prev)  # the transpose is part of the price
                    return prev
                return build
            row = {"what": what, "tokens": T, "MLX float16": best(mlx_chain, n),
                   "blocks": min(best(chain(lambda i, xt, prev: blocks_matvec(raw[i], rows, cols, xt, prev, **how)), n)
                                 for how in (dict(sub=1, R=2), dict(sub=1, R=4, scaled=True)))}
            for name, how in VARIANTS.items():
                row[name] = best(chain(lambda i, xt, prev, how=how: codes_matvec(*ours[i], xt, prev, **how)), n)
            out_rows.append({k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()})
            print(f"  {T} tokens: " + " | ".join(f"{k}: {v:7.1f}" for k, v in row.items() if isinstance(v, float)), flush=True)
        del raw, own, ours
    if args.json:
        Path(args.json).write_text(json.dumps(out_rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
