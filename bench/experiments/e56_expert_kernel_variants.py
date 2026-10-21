"""e56: what the experts' kernel waits on, by variants of one product.

The gate product of a layer's routed experts (512 experts of 640 x 2,560 Q2_0 weights), for 10, 40 and 80
(token, expert) pairs, i.e. one, four and eight tokens of ten experts each. 48 layers of their own weights,
chained (each layer's input waits on the one before), so no layer's weights are in a cache, as in a pass.

  lookup     the engine's kernel: a table of the token's activations, looked up by the weights' codes
             (blocks.lookup_table + blocks.lookup_matvec)
  bytes      the same grid reading the same bytes and doing nothing with them: the floor of that access pattern
  uniform    the lookup with the table's index the same across a SIMD group: what divergent lookups cost
  planes-L   the weights split once into a plane of codes and a plane of scales; L threads to a row read them
             as 16-byte vectors, side by side, and decode a byte into four weights through a 2 KB table in
             threadgroup memory (no table of activations)

Weights and routes are random (as e43); the planes' result is checked against the lookup's. Each variant is
timed in turn, shuffled, in one process; the lower quartile of the rounds, per layer.

    uv run python bench/experiments/e56_expert_kernel_variants.py --json bench/results/e56_expert_kernel_variants.json
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from strata_mlx import blocks  # noqa: E402

EXPERTS, ROWS, COLS, K = 512, 640, 2560, 10
ROW_BYTES = COLS // 64 * 18  # 720
LAYERS = 48

_BYTES = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    uint acc = 0;
    for (uint n = part; n < RUNS; n += P) {
        const device uchar* b = row + (n / 2) * 18;
        const device uchar* q = b + 2 + 8 * (n % 2);
        acc ^= uint(b[0]) | (uint(b[1]) << 8);
        for (uint i = 0; i < 8; ++i) acc ^= uint(q[i]) << (i % 4 * 8);
    }
    for (ushort k = 1; k < P; ++k) acc ^= simd_shuffle_xor(acc, k);
    if (part == 0) out[pair * ROWS + r] = static_cast<OutT>(float(acc & 1023) * 1e-3f);
"""

_PLANES = """
    uint lane = thread_position_in_grid.x % L;
    uint r = thread_position_in_grid.x / L;
    uint pair = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    threadgroup half4 lut[256];
    for (uint i = tid; i < 256; i += TG) {
        lut[i] = half4(half(i & 3), half((i >> 2) & 3), half((i >> 4) & 3), half(i >> 6)) - half4(1.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint e = uint(idx[pair]);
    const device uint4* c4 = (const device uint4*)codes + (e * ROWS + r) * GROUPS;  // a group: 64 weights, 16 bytes
    const device half* sc = scales + (e * ROWS + r) * GROUPS;
    const device half4* xv = (const device half4*)x + (pair / SHARE) * (COLS / 4);
    float sum = 0.0f;
    for (uint g = lane; g < GROUPS; g += L) {
        uint4 w = c4[g];
        float s = 0.0f;
        for (uint k = 0; k < 4; ++k) {
            uint word = w[k];
            for (uint b = 0; b < 4; ++b) {
                s += float(dot(lut[(word >> (8 * b)) & 255], xv[g * 16 + k * 4 + b]));
            }
        }
        sum += float(sc[g]) * s;
    }
    for (ushort k = 1; k < L; k <<= 1) sum += simd_shuffle_xor(sum, k);
    if (lane == 0) out[pair * ROWS + r] = static_cast<OutT>(sum);
"""


_PLANES_BYTES = """
    uint lane = thread_position_in_grid.x % L;
    uint r = thread_position_in_grid.x / L;
    uint pair = thread_position_in_grid.y;
    uint e = uint(idx[pair]);
    const device uint4* c4 = (const device uint4*)codes + (e * ROWS + r) * GROUPS;
    const device half* sc = scales + (e * ROWS + r) * GROUPS;
    uint4 acc = uint4(0);
    float f = 0.0f;
    for (uint g = lane; g < GROUPS; g += L) {
        acc ^= c4[g];
        f += float(sc[g]);
    }
    uint a = acc.x ^ acc.y ^ acc.z ^ acc.w;
    for (ushort k = 1; k < L; k <<= 1) { a ^= simd_shuffle_xor(a, k); f += simd_shuffle_xor(f, k); }
    if (lane == 0) out[pair * ROWS + r] = static_cast<OutT>(float(a & 1023) * 1e-3f + f);
"""

# as _PLANES, the token's activations first copied into threadgroup memory (a threadgroup is one pair's rows)
_PLANES_TG = _PLANES.replace(
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    uint e",
    "    threadgroup half4 xs[COLS / 4];\n"
    "    const device half4* xg = (const device half4*)x + (pair / SHARE) * (COLS / 4);\n"
    "    for (uint i = tid; i < COLS / 4; i += TG) xs[i] = xg[i];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    uint e").replace(
    "    const device half4* xv = (const device half4*)x + (pair / SHARE) * (COLS / 4);\n",
    "    const threadgroup half4* xv = xs;\n")


# the engine's table of the token's activations, the codes read as the planes are: side by side, 16 bytes at once
_PLANES_LOOKUP = """
    uint lane = thread_position_in_grid.x % L;
    uint r = thread_position_in_grid.x / L;
    uint pair = thread_position_in_grid.y;
    uint e = uint(idx[pair]);
    const device uint4* c4 = (const device uint4*)codes + (e * ROWS + r) * GROUPS;
    const device half* sc = scales + (e * ROWS + r) * GROUPS;
    auto t_ = table + (pair / SHARE) * (COLS / 2) * 16;
    float sum = 0.0f;
    for (uint g = lane; g < GROUPS; g += L) {
        uint4 w = c4[g];
        auto t = t_ + g * 32 * 16;
        float s = 0.0f;
        for (uint k = 0; k < 4; ++k) {
            uint word = w[k];
            for (uint b = 0; b < 4; ++b) {
                uint c = (word >> (8 * b)) & 255;
                uint at = (k * 4 + b) * 2 * 16;
                s += float(t[at + (c & 15)]) + float(t[at + 16 + (c >> 4)]);
            }
        }
        sum += float(sc[g]) * s;
    }
    for (ushort k = 1; k < L; k <<= 1) sum += simd_shuffle_xor(sum, k);
    if (lane == 0) out[pair * ROWS + r] = static_cast<OutT>(sum);
"""


def make_layer(rng):
    """One layer's gate stack: Q2_0 blocks, and the same weights as planes."""
    raw = rng.integers(0, 256, size=(EXPERTS, ROWS, COLS // 64, 18), dtype=np.uint8)
    d = (rng.uniform(0.002, 0.02, size=(EXPERTS, ROWS, COLS // 64))).astype(np.float16)
    raw[..., :2] = d.view(np.uint8).reshape(EXPERTS, ROWS, COLS // 64, 2)
    codes = np.ascontiguousarray(raw[..., 2:]).reshape(-1)
    return {"blocks": mx.array(raw.reshape(-1)), "codes": mx.array(codes.view(np.uint32)),
            "scales": mx.array(d.reshape(-1))}


_kernels = {}


def kernel(name, source, inputs, **constants):
    key = (name, tuple(sorted(constants.items())))
    if key not in _kernels:
        header = "".join(f"constant uint {k} = {v};\n" for k, v in constants.items()) + blocks._COMMON
        _kernels[key] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=["out"], source=source,
                                             header=header)
    return _kernels[key]


def lookup(layer, idx, x, uniform=False):
    lin = SimpleNamespace(blocks=layer["blocks"], output_dims=ROWS, input_dims=COLS)
    table = blocks.lookup_table(x)
    if not uniform:
        return blocks.lookup_matvec(lin, idx, table, K)
    pairs = idx.shape[0]
    per_row = min(blocks._threads_per_row(COLS // 32), 8 if pairs <= 40 else 2)
    # the codes are still read (and kept from being optimised away), but the table's index is the pair's
    src = (blocks._LOOKUP.replace("uint c = q[i];", "uint c = q[i]; s += float(c) * 1e-30f;")
           .replace("(c & 15)", "(pair & 15)").replace("(c >> 4)", "((pair >> 4) & 15)"))
    k = kernel("q2_lookup_uniform", src, ["blocks", "idx", "table"], ROWS=ROWS, COLS=COLS, RUNS=COLS // 32,
               ROW_BYTES=ROW_BYTES, P=per_row, SHARE=K)
    return k(inputs=[layer["blocks"], idx, table], template=[("OutT", mx.float16)], grid=(per_row * ROWS, pairs, 1),
             threadgroup=(32, 1, 1), output_shapes=[(pairs, ROWS)], output_dtypes=[mx.float16])[0]


def bytes_only(layer, idx, x):
    pairs = idx.shape[0]
    per_row = min(blocks._threads_per_row(COLS // 32), 8 if pairs <= 40 else 2)
    k = kernel("q2_bytes", _BYTES, ["blocks", "idx", "x"], ROWS=ROWS, COLS=COLS, RUNS=COLS // 32,
               ROW_BYTES=ROW_BYTES, P=per_row)
    return k(inputs=[layer["blocks"], idx, x], template=[("OutT", mx.float16)], grid=(per_row * ROWS, pairs, 1),
             threadgroup=(32, 1, 1), output_shapes=[(pairs, ROWS)], output_dtypes=[mx.float16])[0]


def planes(layer, idx, x, lanes):
    pairs = idx.shape[0]
    k = kernel("q2_planes", _PLANES, ["codes", "scales", "idx", "x"], ROWS=ROWS, COLS=COLS, GROUPS=COLS // 64,
               L=lanes, SHARE=K, TG=128)
    return k(inputs=[layer["codes"], layer["scales"], idx, x], template=[("OutT", mx.float16)],
             grid=(lanes * ROWS, pairs, 1), threadgroup=(128, 1, 1), output_shapes=[(pairs, ROWS)],
             output_dtypes=[mx.float16])[0]


def planes_bytes(layer, idx, x, lanes):
    pairs = idx.shape[0]
    k = kernel("q2_planes_bytes", _PLANES_BYTES, ["codes", "scales", "idx"], ROWS=ROWS, GROUPS=COLS // 64, L=lanes)
    return k(inputs=[layer["codes"], layer["scales"], idx], template=[("OutT", mx.float16)],
             grid=(lanes * ROWS, pairs, 1), threadgroup=(128, 1, 1), output_shapes=[(pairs, ROWS)],
             output_dtypes=[mx.float16])[0]


def planes_tg(layer, idx, x, lanes):
    pairs = idx.shape[0]
    k = kernel("q2_planes_tg", _PLANES_TG, ["codes", "scales", "idx", "x"], ROWS=ROWS, COLS=COLS, GROUPS=COLS // 64,
               L=lanes, SHARE=K, TG=128)
    return k(inputs=[layer["codes"], layer["scales"], idx, x], template=[("OutT", mx.float16)],
             grid=(lanes * ROWS, pairs, 1), threadgroup=(128, 1, 1), output_shapes=[(pairs, ROWS)],
             output_dtypes=[mx.float16])[0]


def planes_lookup(layer, idx, x, lanes):
    pairs = idx.shape[0]
    table = blocks.lookup_table(x)
    k = kernel("q2_planes_lookup", _PLANES_LOOKUP, ["codes", "scales", "idx", "table"], ROWS=ROWS, COLS=COLS,
               GROUPS=COLS // 64, L=lanes, SHARE=K)
    return k(inputs=[layer["codes"], layer["scales"], idx, table], template=[("OutT", mx.float16)],
             grid=(lanes * ROWS, pairs, 1), threadgroup=(128, 1, 1), output_shapes=[(pairs, ROWS)],
             output_dtypes=[mx.float16])[0]


def chain(product, layers, routes, x0):
    """Every layer's product, each layer's input waiting on the one before's output."""
    x = x0
    for layer, idx in zip(layers, routes):
        out = product(layer, idx, x)
        x = x + (out[:1, :1] * 0).astype(x.dtype)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--json")
    args = p.parse_args()
    rng = np.random.default_rng(56)
    t0 = time.perf_counter()
    layers = [make_layer(rng) for _ in range(LAYERS)]
    mx.eval([list(layer.values()) for layer in layers])
    print(f"{LAYERS} layers of weights made in {time.perf_counter() - t0:.0f} s", flush=True)

    variants = {"lookup": lambda l, i, x: lookup(l, i, x), "bytes": bytes_only,
                "uniform": lambda l, i, x: lookup(l, i, x, uniform=True),
                "glue only": lambda l, i, x: mx.zeros((1, 1), dtype=mx.float16)}
    for lanes in (4, 8, 16):
        variants[f"planes-{lanes}"] = (lambda L: lambda l, i, x: planes(l, i, x, L))(lanes)
        variants[f"pl-bytes-{lanes}"] = (lambda L: lambda l, i, x: planes_bytes(l, i, x, L))(lanes)
        variants[f"pl-tg-{lanes}"] = (lambda L: lambda l, i, x: planes_tg(l, i, x, L))(lanes)
        variants[f"pl-lookup-{lanes}"] = (lambda L: lambda l, i, x: planes_lookup(l, i, x, L))(lanes)

    result = {"rows": []}
    for tokens in (1, 4, 8):
        pairs = tokens * K
        routes = [mx.array(np.concatenate([rng.choice(EXPERTS, K, replace=False) for _ in range(tokens)])
                           .astype(np.uint32)) for _ in range(LAYERS)]
        x0 = mx.array(rng.standard_normal((tokens, COLS)).astype(np.float16))
        mx.eval(routes, x0)
        ref = lookup(layers[0], routes[0], x0)
        for name, got in (("planes-8", planes(layers[0], routes[0], x0, 8)),
                          ("pl-tg-8", planes_tg(layers[0], routes[0], x0, 8)),
                          ("pl-lookup-8", planes_lookup(layers[0], routes[0], x0, 8))):
            err = mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))) / mx.max(mx.abs(ref.astype(mx.float32)))
            print(f"{tokens} tokens: {name} against lookup, largest difference {err.item():.2e} of the largest value", flush=True)
        times = {name: [] for name in variants}
        for name, fn in variants.items():  # compile, and a first run
            mx.eval(chain(fn, layers, routes, x0))
        order = list(variants) * args.rounds
        random.Random(tokens).shuffle(order)
        for name in order:
            t = time.perf_counter()
            mx.eval(chain(variants[name], layers, routes, x0))
            times[name].append(time.perf_counter() - t)
        floor = len({(l, int(e)) for l, r in enumerate(routes) for e in np.array(r)}) * ROWS * ROW_BYTES / 490e9 / LAYERS
        glue = statistics.quantiles(times["glue only"], n=4)[0] / LAYERS
        print(f"--- {tokens} tokens ({pairs} pairs); the bytes at 490 GB/s: {floor * 1e6:.1f} us a layer; "
              f"the chain's glue {glue * 1e6:.1f} us a layer", flush=True)
        for name, ts in times.items():
            q1 = statistics.quantiles(ts, n=4)[0] / LAYERS
            row = {"tokens": tokens, "pairs": pairs, "variant": name, "us_per_layer": round(q1 * 1e6, 1),
                   "us_less_glue": round((q1 - glue) * 1e6, 1), "floor_us": round(floor * 1e6, 1)}
            result["rows"].append(row)
            print(f"  {name:<10} {row['us_per_layer']:>7} us a layer, {row['us_less_glue']:>7} less the glue "
                  f"({row['us_less_glue'] / row['floor_us']:.2f} x the floor)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
