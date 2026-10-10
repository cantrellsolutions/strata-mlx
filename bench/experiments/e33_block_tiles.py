"""Experiment 33: a prompt's experts multiplied from the file's blocks, a tile at a time.

The codebook types have no packed operand for Apple's matrix product, so a chunk of a prompt unpacks each stack
of 512 experts to 1.7 GB of floats (5.9 ms) for MLX's gather_mm (5.2 ms), three times a layer (e31). Here a
threadgroup unpacks only the tile it multiplies, 64 rows of one expert by TK columns, into threadgroup memory
as float16, and Apple's matmul2d takes that as its operand, for the tile's 16 tokens; the next TK columns
follow into the same memory. Whatever the type: `unpack32` of blocks.py does the unpacking.

On the inputs of some layers' experts taken from a real pass (as e31), one product at a time and all three:
against the way prompts go now, and checked against the kernel that multiplies a pair at a time.

(Run on the code of commit 26e78a8, when "the way prompts go now" was the 2-bit product for Q2_0 and unpacking
to floats for the rest. The kernel here is blocks.tile_matmul since, so run again it compares it with itself.)

    uv run python bench/experiments/e33_block_tiles.py --model <shard 1> [--json out.json]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
import strata_mlx.blocks as B  # noqa: E402
from spec_decode import EDIT_ME, tokenize  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--tokens", type=int, default=1024)
p.add_argument("--layers", nargs="+", type=int, default=[1, 13, 25, 37, 46])
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--columns", nargs="+", type=int, default=[64, 128], help="TK: columns unpacked at a time")
p.add_argument("--tile-tokens", nargs="+", type=int, default=[16, 32], help="TM: tokens to a tile")
p.add_argument("--json")
args = p.parse_args()

TN, SIMD_GROUPS = 64, 2

BODY = """
    uint tx = threadgroup_position_in_grid.x, ty = threadgroup_position_in_grid.y;
    uint e = uint(tile_expert[ty]);
    if (e >= EXPERTS) {
        return;
    }
    threadgroup half wbuf[TN * TK];
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false);
    matmul2d<desc, execution_simdgroups<SIMD_GROUPS>> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A((device half *)a, dextents<int32_t, 2>(COLS, int(a_shape[0])));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> Bt((threadgroup half *)wbuf, dextents<int32_t, 2>(TK, TN));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(ROWS, int(a_shape[0])));
    auto tA = A.slice<TK, TM>(0, ty * TM);
    auto acc = op.get_destination_cooperative_tensor<decltype(tA), decltype(Bt), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0;
        }
    }
    uint lid = thread_index_in_threadgroup;
    const device uchar* base = blocks + (e * ROWS + tx * TN) * ROW_BYTES;
    float w[32];
    for (uint kt = 0; kt < COLS / TK; ++kt) {
        for (uint u = lid; u < TN * (TK / 32); u += 32 * SIMD_GROUPS) {
            uint r = u / (TK / 32), n = u % (TK / 32);
            unpack32(base + r * ROW_BYTES, kt * (TK / 32) + n, w);
            for (uint j = 0; j < 32; ++j) {
                wbuf[r * TK + 32 * n + j] = half(w[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tA = A.slice<TK, TM>(kt * TK, ty * TM);
        auto part = op.get_destination_cooperative_tensor<decltype(tA), decltype(Bt), float>();
        op.run(tA, Bt, part);  // sets, does not add
        for (uint16_t i = 0; i < part.get_capacity(); ++i) {
            if (part.is_valid_element(i)) {
                acc[i] += part[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tC = C.slice<TN, TM>(tx * TN, ty * TM);
    acc.store(tC);
"""

_kernels = {}


def tile_product(lin, a, tiles, tk, tm):
    """out[slot] = W[expert of the slot's tile] . a[slot], from lin's blocks: (slots, rows) float32."""
    key = (lin.kind, lin.num_experts, lin.output_dims, lin.input_dims, tk, tm)
    if key not in _kernels:
        header = ("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"
                  + B._header(lin.kind, lin.output_dims, lin.input_dims, TM=tm, TN=TN, TK=tk, SIMD_GROUPS=SIMD_GROUPS,
                              EXPERTS=lin.num_experts))
        _kernels[key] = mx.fast.metal_kernel(name=f"tiles_{lin.kind.lower()}", input_names=["blocks", "a", "tile_expert"],
                                             output_names=["c"], source=BODY, header=header)
    width = 32 * SIMD_GROUPS
    (out,) = _kernels[key](inputs=[lin.blocks, a, tiles], grid=(width * (lin.output_dims // TN), a.shape[0] // tm, 1),
                           threadgroup=(width, 1, 1), output_shapes=[(a.shape[0], lin.output_dims)], output_dtypes=[mx.float32])
    return out


def tile_glu(x, indices, up, gate, down, tk, tm):
    before, B.TOKENS_TILE = B.TOKENS_TILE, tm
    tiles, token, where = B.tiles_of(indices, gate.num_experts)
    B.TOKENS_TILE = before
    a = x.reshape(-1, x.shape[-1])[token]
    h = B.swiglu(tile_product(gate, a, tiles, tk, tm), tile_product(up, a, tiles, tk, tm)).astype(x.dtype)
    out = tile_product(down, h, tiles, tk, tm)[where].astype(x.dtype)
    return out.reshape(*indices.shape, down.output_dims)


model, _ = load(args.model)
text = "<|im_start|>user\nSay in two sentences what this module is for.\n\n```python\n" + EDIT_ME * 9 + "```<|im_end|>\n<|im_start|>assistant\n"
prompt = tokenize(args.model, text)[: args.tokens]
taken, calls = {}, [0]
switch_glu = B.switch_glu


def taking(x, indices, up, gate, down):
    if calls[0] in args.layers:
        mx.eval(x, indices)
        taken[calls[0]] = (x, indices, up, gate, down)
    calls[0] += 1
    return switch_glu(x, indices, up, gate, down)


engine = Engine(model)
B.switch_glu = taking
engine.read(prompt[:-1])
B.switch_glu = switch_glu
engine.reset()
mx.clear_cache()


def ms(f, *a):
    out, ts = None, []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        out = f(*a)
        mx.eval(*(out if isinstance(out, (tuple, list)) else [out]))
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2], out


by_kind, whole, worst = defaultdict(lambda: defaultdict(list)), defaultdict(list), defaultdict(float)
for layer, (x, indices, up, gate, down) in sorted(taken.items()):
    pairs, k = indices.size, indices.shape[-1]
    x2 = x.reshape(-1, x.shape[-1])
    # the reference: a pair at a time, sums in float32 over the weights as the file has them (64 tokens' pairs)
    sub = slice(0, 64)
    limit, B.PACKED_FROM, B.LOOKUP_FROM = (B.PACKED_FROM, B.LOOKUP_FROM), 1 << 62, 1 << 62
    ref = np.array(switch_glu(x[:, sub], indices[:, sub], up, gate, down).astype(mx.float32))
    B.PACKED_FROM, B.LOOKUP_FROM = limit
    t_now, now = ms(switch_glu, x, indices, up, gate, down)
    whole["as prompts go now"].append(t_now)
    worst["as prompts go now"] = max(worst["as prompts go now"],
                                     float(np.abs(np.array(now[:, sub].astype(mx.float32)) - ref).max() / np.abs(ref).max()))
    for tm in args.tile_tokens:
        before, B.TOKENS_TILE = B.TOKENS_TILE, tm
        tiles, token, where = B.tiles_of(indices, gate.num_experts)
        B.TOKENS_TILE = before
        a = x2[token]
        mx.eval(tiles, token, where, a)
        for tk in args.columns:
            if TN * tk * 2 >= 32768:  # all the memory a threadgroup has: wrong results on the real shapes, and slow
                continue
            outs = {}
            for name, lin in (("gate", gate), ("up", up)):
                t, outs[name] = ms(tile_product, lin, a, tiles, tk, tm)
                by_kind[(lin.kind, lin.output_dims, lin.input_dims)][(tm, tk)].append((t, 2.0 * pairs * lin.output_dims * lin.input_dims))
            h = B.swiglu(outs["gate"], outs["up"]).astype(x.dtype)
            mx.eval(h)
            t, _ = ms(tile_product, down, h, tiles, tk, tm)
            by_kind[(down.kind, down.output_dims, down.input_dims)][(tm, tk)].append((t, 2.0 * pairs * down.output_dims * down.input_dims))
            t, got = ms(tile_glu, x, indices, up, gate, down, tk, tm)
            whole[f"tiles of {tm} tokens, {tk} columns at a time"].append(t)
            err = np.abs(np.array(got[:, sub].astype(mx.float32)) - ref).max() / np.abs(ref).max()
            name = f"tiles of {tm} tokens, {tk} columns at a time"
            worst[name] = max(worst[name], float(err))
    mx.clear_cache()


def mean(xs):
    return sum(xs) / len(xs)


print(f"\n{args.tokens - 1} tokens, layers {sorted(taken)}; one product from the blocks (ms, T operations a second):")
for (kind, rows, cols), ways in sorted(by_kind.items()):
    cells = "   ".join(f"TM {tm} TK {tk}: {mean([r[0] for r in v]):5.2f} ms {mean([r[1] for r in v]) / 1e12 / (mean([r[0] for r in v]) / 1e3):4.1f}"
                       for (tm, tk), v in sorted(ways.items()))
    print(f"  {kind:8} {cols:4} -> {rows:4}   {cells}")
print("\na layer's ten chosen, all three products (ms, the mean of the layers):")
print(f"  {'':46} {'ms':>7}   largest difference from a pair at a time, of the largest value")
for name, ts in whole.items():
    print(f"  {name:46} {mean(ts):7.2f}   {worst[name]:.1e}")
if args.json:
    Path(args.json).write_text(json.dumps({
        "tokens": args.tokens - 1,
        "products": [{"kind": kind, "rows": rows, "cols": cols, "tile_tokens": tm, "columns": tk,
                      "ms": mean([r[0] for r in v]), "operations": mean([r[1] for r in v])}
                     for (kind, rows, cols), ways in sorted(by_kind.items()) for (tm, tk), v in sorted(ways.items())],
        "whole_ms": {k: mean(v) for k, v in whole.items()}, "largest_difference": dict(worst)}, indent=1) + "\n")
