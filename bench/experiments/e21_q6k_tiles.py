"""Experiment 21: the Q6_K kernel of e20 with the tokens as a vector and several rows to a thread.

e20's kernel beats MLX's float16 product for one and two tokens (88 against 130 us) and loses from four on
(134, 335 for six, 433 for eight) while MLX's stays at 130: every weight loads, converts, multiplies and adds
an activation of every token, and from six tokens on its 4 T sums no longer fit the registers. Here the
activations come as float32 with a position's tokens side by side, so that one load brings all of them and
the sums are vectors, and a thread takes R rows, so that it loads them once for R rows.

    uv run python bench/experiments/e21_q6k_tiles.py --model <IQ3_S shard 1>
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
VEC = {1: "float", 2: "float2", 3: "float3", 4: "float4"}
PACKED = {1: "float", 2: "packed_float2", 3: "packed_float3", 4: "packed_float4"}


def source(tokens: int, R: int, groups: int, scaled: bool) -> str:
    """One thread: SUB runs of R rows, for every token. `scaled`: a weight is multiplied by its scale at once and
    a row has one sum, instead of a sum for each of the four scales of a run."""
    chunks = [(o, min(4, tokens - o)) for o in range(0, tokens, 4)]  # (first token, how many) of each vector
    K = range(4)
    decl = "".join(f"{VEC[n]} a{r}_{o} = 0.0f; " for r in range(R) for o, n in chunks)
    loads = "".join(f"{VEC[n]} x{k}_{o} = {VEC[n]}(*(const device {PACKED[n]}*)(x + (at + {32 * k}) * T + {o})); "
                    for k in K for o, n in chunks)
    rows = ""
    for r in range(R):
        rows += f"""
            {{
                const device uchar* blk = row{r} + b * 210;
                uint lo = blk[64 * h + 16 * is + i], hi = blk[64 * h + 16 * is + 32 + i], top = blk[128 + 32 * h + 16 * is + i];
                float w0 = float(int((lo & 15) | ((top & 3) << 4)) - 32);
                float w1 = float(int((hi & 15) | (((top >> 2) & 3) << 4)) - 32);
                float w2 = float(int((lo >> 4) | (((top >> 4) & 3) << 4)) - 32);
                float w3 = float(int((hi >> 4) | ((top >> 6) << 4)) - 32);
"""
        if scaled:
            rows += "".join(f"                a{r}_{o} += (w0 * s{r}_0) * x0_{o} + (w1 * s{r}_1) * x1_{o} + (w2 * s{r}_2) * x2_{o} + (w3 * s{r}_3) * x3_{o};\n"
                            for o, n in chunks)
        else:
            rows += "".join(f"                c{r}_{k}_{o} += w{k} * x{k}_{o};\n" for k in K for o, n in chunks)
        rows += "            }\n"
    scales = "".join(
        f"const device uchar* blk{r} = row{r} + b * 210; float d{r} = half_at(blk{r} + 208); "
        + "".join(f"float s{r}_{k} = d{r} * s8(blk{r}[192 + 8 * h + is + {2 * k}]); " for k in K)
        for r in range(R))
    parts = "" if scaled else "".join(f"{VEC[n]} c{r}_{k}_{o} = 0.0f; " for r in range(R) for k in K for o, n in chunks)
    folds = "" if scaled else "".join(
        f"a{r}_{o} += s{r}_0 * c{r}_0_{o} + s{r}_1 * c{r}_1_{o} + s{r}_2 * c{r}_2_{o} + s{r}_3 * c{r}_3_{o}; "
        for r in range(R) for o, n in chunks)
    outs = ""
    for r in range(R):
        for o, n in chunks:
            outs += f"{{ {VEC[n]} s = simd_sum(a{r}_{o}); "
            for j in range(n):
                value = "s" if n == 1 else f"s[{j}]"
                if groups == 1:
                    outs += f"if (lane == 0) {{ out[{o + j} * ROWS + R * rr + {r}] = static_cast<OutT>({value} + 0.0f * float(prev[0])); }} "
                else:
                    outs += f"if (lane % 32 == 0) {{ shared[(lane / 32) * (T * R) + {r} * T + {o + j}] = {value}; }} "
            outs += "} "
    tail = "" if groups == 1 else """
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < T * R) {
        float total = 0.0f;
        for (uint g = 0; g < G; ++g) { total += shared[g * (T * R) + lane]; }
        out[(lane % T) * ROWS + R * rr + lane / T] = static_cast<OutT>(total + 0.0f * float(prev[0]));
    }
"""
    return f"""
    uint lane = thread_position_in_threadgroup.x;
    uint rr = thread_position_in_grid.x / UNITS;
    {"threadgroup float shared[G * T * R];" if groups > 1 else ""}
    {decl}
    {"".join(f"const device uchar* row{r} = blocks + (R * rr + {r}) * ROW_BYTES; " for r in range(R))}
    for (uint run = lane * SUB; run < (lane + 1) * SUB; ++run) {{
        uint b = run / 4, h = (run / 2) % 2, is = run % 2;
        {scales}
        uint base = 256 * b + 128 * h + 16 * is;
        {parts}
        for (uint i = 0; i < 16; ++i) {{
            uint at = base + i;
            {loads}
            {rows}
        }}
        {folds}
    }}
    {outs}
    {tail}
"""


_kernels = {}


def q6k_matvec(blocks: mx.array, rows: int, cols: int, xt: mx.array, prev: mx.array, sub: int = 1, R: int = 1,
               scaled: bool = False, dtype=mx.float16) -> mx.array:
    """xt (cols, T) float32 -> (T, rows) of `dtype`; `prev` only makes this product wait for another."""
    T = xt.shape[1]
    units = cols // 64 // sub
    groups = -(-units // 32)
    key = (rows, cols, T, sub, R, scaled)
    if key not in _kernels:
        header = HEADER + (f"constant uint ROWS = {rows};\nconstant uint COLS = {cols};\nconstant uint ROW_BYTES = {cols // 256 * 210};\n"
                           f"constant uint UNITS = {units};\nconstant uint G = {groups};\nconstant uint T = {T};\n"
                           f"constant uint SUB = {sub};\nconstant uint R = {R};\n")
        _kernels[key] = mx.fast.metal_kernel(name=f"q6k_mv_{T}_{sub}_{R}_{int(scaled)}", input_names=["blocks", "x", "prev"],
                                             output_names=["out"], source=source(T, R, groups, scaled), header=header)
    return _kernels[key](inputs=[blocks, xt, prev], template=[("OutT", dtype)], grid=(units * rows // R, 1, 1),
                         threadgroup=(units, 1, 1), output_shapes=[(T, rows)], output_dtypes=[dtype])[0]


def best(build, n, rounds=9):
    out = float("inf")
    for _ in range(rounds):
        g = build()
        t0 = time.perf_counter()
        mx.eval(g)
        out = min(out, time.perf_counter() - t0)
    return out / n * 1e6


def check(t, raw, **how):
    from gguf import quants
    from gguf.constants import GGMLQuantizationType

    rows, cols = t.shape
    some = min(rows, 512)
    w = quants.dequantize(np.asarray(t.data).reshape(rows, -1)[:some], GGMLQuantizationType.Q6_K).reshape(some, cols)
    worst = 0.0
    for T in (1, 3, 4, 7, 8):
        x = mx.random.normal((T, cols)).astype(mx.float16)
        want = np.array(x.astype(mx.float32)) @ w.T
        xt = mx.contiguous(x.astype(mx.float32).T)
        got = np.array(q6k_matvec(raw, rows, cols, xt, mx.zeros((1,), dtype=mx.float16), dtype=mx.float32, **how))[:, :some]
        worst = max(worst, np.abs(got - want).max() / np.abs(want).max())
    return worst


VARIANTS = {
    "64 w, 1 row": dict(sub=1, R=1),
    "64 w, 1 row, scaled": dict(sub=1, R=1, scaled=True),
    "64 w, 2 rows": dict(sub=1, R=2),
    "64 w, 2 rows, scaled": dict(sub=1, R=2, scaled=True),
    "64 w, 4 rows, scaled": dict(sub=1, R=4, scaled=True),
    "128 w, 2 rows, scaled": dict(sub=2, R=2, scaled=True),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--json")
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--what", nargs="+", default=["attn_qkv", "output layer"])
    args = p.parse_args()
    g = GGUF(args.model)
    out_rows = []
    for what, tensors, repeat in (
            ("attn_qkv", [t for t in g.tensors.values() if t.name.endswith("attn_qkv.weight") and t.type == "Q6_K"][:12], 1),
            ("output layer", [g["output.weight"]], 3)):
        if what not in args.what:
            continue
        rows, cols = tensors[0].shape
        raw = [mx.array(np.ascontiguousarray(t.data)) for t in tensors]
        own = [linear(t) for t in tensors]
        for m in own:
            m.set_dtype(mx.float16)
        mx.eval(raw, [m.parameters() for m in own])
        n = len(tensors) * repeat
        print(f"{what}: {len(tensors)} tensors of {rows} x {cols}", flush=True)
        print("  against the dequantized weights (float32 output): "
              + ", ".join(f"{name}: {check(tensors[0], raw[0], **how):.1e}" for name, how in VARIANTS.items()), flush=True)
        for T in args.tokens:
            x = mx.random.normal((T, cols)).astype(mx.float16)
            mx.eval(x)

            def mlx_chain():
                prev = None
                for _ in range(repeat):
                    for m in own:
                        prev = m(x if prev is None else x + 0.0 * prev[:, :1])
                return prev

            def block_chain(how):
                def chain():
                    prev = mx.zeros((1,), dtype=mx.float16)
                    for _ in range(repeat):
                        for b in raw:
                            xt = mx.contiguous(x.astype(mx.float32).T)  # part of the price: once a product
                            prev = q6k_matvec(b, rows, cols, xt, prev, **how)
                    return prev
                return chain
            a = best(mlx_chain, n)
            ours = {name: best(block_chain(how), n) for name, how in VARIANTS.items()}
            out_rows.append({"what": what, "tokens": T, "mlx_float16_us": round(a, 1), **{k: round(v, 1) for k, v in ours.items()}})
            print(f"  {T} tokens: MLX float16 {a:8.1f} us | " + " | ".join(f"{k}: {v:7.1f}" for k, v in ours.items()), flush=True)
        del raw, own
    if args.json:
        Path(args.json).write_text(json.dumps(out_rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
