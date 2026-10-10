"""Experiment 31: inside the ten chosen experts of a chunk of a prompt, and what a good kernel would reach.

e30 found them to be 71% of the time a prompt takes, as one number. Here the inputs of some layers' experts
are taken from a real pass over a prompt (their activations, and what the routers chose), and on those:

    the steps of the way prompts go now (blocks._packed_glu), each made to finish: the wait for the router's
    choice and the sorting on the CPU, the gathers, and for each of the three products the split of the stack
    into codes and scales, the sums, the product itself

    one product (gate: 2560 -> 640) the other ways there are: the table of sums a pair at a time; MLX's own
    grouped products, over the stack requantized at 2 bits (gather_qmm) and over the stack as float16
    (gather_mm), which need 0.2 and 1.7 GB the file does not have but say what a well-made kernel gets out of
    this GPU when an expert has thirty tokens

    uv run python bench/experiments/e31_expert_parts.py --model <shard 1> [--json out.json]
(Run on the code of commit 4d72e24. Since then the 2-bit product this takes apart, blocks.packed_matmul, has been
replaced by blocks.tile_matmul, and these names are gone from blocks.py.)
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
p.add_argument("--json")
args = p.parse_args()

model, _ = load(args.model)
m = model.model
text = "<|im_start|>user\nSay in two sentences what this module is for.\n\n```python\n" + EDIT_ME * 9 + "```<|im_end|>\n<|im_start|>assistant\n"
prompt = tokenize(args.model, text)[: args.tokens]

# ---- the inputs of some layers' experts, from a real pass
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
engine.read(prompt[:-1])  # one piece
B.switch_glu = switch_glu
engine.reset()
mx.clear_cache()


def ms(f, *a):
    """f(*a) made to finish, in ms (the median of some), and what it gave."""
    out, ts = None, []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        out = f(*a)
        mx.eval(*(out if isinstance(out, (tuple, list)) else [out]))
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2], out


steps, ways, spread = defaultdict(list), defaultdict(list), []
for layer, (x, indices, up, gate, down) in sorted(taken.items()):
    kinds = {lin.kind for lin in (up, gate, down)}
    idx = np.array(indices)
    k, flat = idx.shape[-1], idx.reshape(-1)
    pairs = len(flat)
    counts = np.unique(flat, return_counts=True)[1]
    spread.append({"layer": layer, "pairs": pairs, "experts": len(counts), "most": int(counts.max()),
                   "median": float(np.median(counts)), "kinds": sorted(kinds)})
    x2 = x.reshape(-1, x.shape[-1])
    ops = {lin: 2.0 * pairs * lin.output_dims * lin.input_dims for lin in (up, gate, down)}

    if kinds == {"Q2_0"} and x.dtype == mx.float16:
        # the way prompts go now, step by step (the body of blocks._packed_glu)
        def sort_on_cpu():
            idx = np.array(indices)
            flat = idx.reshape(-1)
            order = np.argsort(flat, kind="stable")
            used, counts = np.unique(flat, return_counts=True)
            padded = -(-counts // B.TOKENS_TILE) * B.TOKENS_TILE
            slot = np.repeat(np.cumsum(padded) - padded, counts) + np.arange(len(flat)) - np.repeat(np.cumsum(counts) - counts, counts)
            token = np.zeros(int(padded.sum()), dtype=np.uint32)
            token[slot] = order // k
            where = np.empty(len(flat), dtype=np.uint32)
            where[order] = slot
            return mx.array(np.repeat(used, padded // B.TOKENS_TILE).astype(np.uint32)), mx.array(token), mx.array(where)

        t, (tiles, token, where) = ms(sort_on_cpu)
        steps["sorting the pairs by expert on the CPU"].append(t)
        t, a = ms(lambda: x2[token])
        steps["gathering the activations"].append(t)
        B.packed_matmul(gate.blocks, (gate.num_experts, gate.output_dims, gate.input_dims), a, tiles)  # the kernels made
        outs = {}
        for name, lin, inp in (("gate", gate, a), ("up", up, a), ("down", down, None)):
            if inp is None:
                t, inp = ms(lambda: B.swiglu(outs["gate"], outs["up"]).astype(x.dtype))
                steps["silu and the product of the two"].append(t)
            split, product = B._kernels[("packed", lin.output_dims, lin.input_dims)]
            nblk, rows, n = lin.input_dims // 64, lin.output_dims, lin.num_experts
            t, (codes, scales) = ms(lambda: split(
                inputs=[lin.blocks], grid=(nblk, n * rows, 1), threadgroup=(min(nblk, 32), 1, 1),
                output_shapes=[(n * rows, lin.input_dims // 4), (n * rows, nblk)], output_dtypes=[mx.uint8, mx.float16]))
            steps["splitting a stack into codes and scales (x3)"].append(t)
            t, sums = ms(lambda: inp.reshape(inp.shape[0], nblk, 64).astype(mx.float32).sum(axis=-1))
            steps["sums of the activations (x3)"].append(t)
            width = 32 * B._SIMD_GROUPS
            t, out = ms(lambda: product(
                inputs=[inp, sums, codes, scales, tiles], grid=(width * (rows // B.ROWS_TILE), inp.shape[0] // B.TOKENS_TILE, 1),
                threadgroup=(width, 1, 1), output_shapes=[(inp.shape[0], rows)], output_dtypes=[mx.float32])[0])
            steps[f"the product: {name}"].append(t)
            ways[f"2-bit product in tiles, the product alone: {name}"].append((t, ops[lin]))
            outs[name] = out
            del codes, scales
        t, _ = ms(lambda: outs["down"][where].astype(x.dtype))
        steps["gathering the outputs back"].append(t)
        t, _ = ms(B._packed_glu, x, indices, up, gate, down)
        steps["all of it, as the engine runs it"].append(t)

        table_ms, table = ms(B.lookup_table, x2)
        t, _ = ms(lambda: B.lookup_matvec(gate, indices.reshape(-1).astype(mx.uint32), table, k))
        ways["table of sums a pair at a time: gate"].append((t, ops[gate]))
        t, _ = ms(B._lookup_glu, x, indices, up, gate, down)
        steps["(the table of sums instead, all three products)"].append(t)

    else:
        # the way prompts go with the codebook types: each stack unpacked to floats for MLX's gather_mm
        t, (xs_, idx_, inv_) = ms(lambda: B._gather_sort(mx.expand_dims(x, (-2, -3)), indices))
        steps["sorting the pairs by expert (MLX)"].append(t)
        outs = {}
        for name, lin, inp in (("gate", gate, xs_), ("up", up, xs_), ("down", down, None)):
            if inp is None:
                t, inp = ms(lambda: B.swiglu(outs["gate"], outs["up"]))
                steps["silu and the product of the two"].append(t)
            shape_ = (lin.num_experts, lin.output_dims, lin.input_dims)
            t, w_ = ms(lambda: B.dequantize(lin.blocks, lin.kind, shape_, x.dtype))
            steps[f"unpacking the stack to floats: {name} ({lin.kind})"].append(t)
            t, out = ms(lambda: mx.gather_mm(inp, w_.swapaxes(-1, -2), rhs_indices=idx_, sorted_indices=True))
            steps[f"gather_mm: {name}"].append(t)
            ways[f"MLX gather_mm over the stack as floats, the product alone: {name}"].append((t, ops[lin]))
            outs[name] = out
            del w_
        t, _ = ms(lambda: B._scatter_unsort(outs["down"], inv_, indices.shape))
        steps["putting the outputs back in order"].append(t)
        mx.clear_cache()
        t, _ = ms(B.switch_glu, x, indices, up, gate, down)
        steps["all of it, as the engine runs it"].append(t)

    # what MLX's own grouped products get from the same pairs: gate only, over forms the file does not have
    shape = (gate.num_experts, gate.output_dims, gate.input_dims)
    xs, idx_sorted, _ = B._gather_sort(mx.expand_dims(x, (-2, -3)), indices)
    mx.eval(xs, idx_sorted)
    w16 = B.dequantize(gate.blocks, gate.kind, shape, mx.float16)
    mx.eval(w16)
    wt = mx.contiguous(w16.swapaxes(-1, -2))
    mx.eval(wt)
    t, ref = ms(lambda: mx.gather_mm(xs, wt, rhs_indices=idx_sorted, sorted_indices=True))
    ways["MLX gather_mm over the stack as float16: gate"].append((t, ops[gate]))
    del wt
    if gate.kind == "Q2_0":
        q, s, b = mx.quantize(w16.astype(mx.float32), group_size=64, bits=2)
        mx.eval(q, s, b)
        t, _ = ms(lambda: mx.gather_qmm(xs, q, s, b, rhs_indices=idx_sorted, transpose=True, group_size=64, bits=2,
                                        sorted_indices=True))
        ways["MLX gather_qmm over the stack requantized at 2 bits: gate"].append((t, ops[gate]))
        del q, s, b
    t, _ = ms(lambda: B.dequantize(gate.blocks, gate.kind, shape, mx.float16))
    ways["(unpacking the stack to float16: gate)"].append((t, 0.0))
    del w16, ref
    mx.clear_cache()

print(f"\n{args.tokens - 1} tokens; the pairs of the layers taken:")
for s in spread:
    print(f"  layer {s['layer']:2}: {s['pairs']} pairs for {s['experts']} experts, {s['median']:.0f} at the median, {s['most']} at most"
          f"   {', '.join(s['kinds'])}")


def mean(xs):
    return sum(xs) / len(xs)


if steps:
    whole = mean(steps["all of it, as the engine runs it"])
    print(f"\nthe way prompts go now, a layer (ms, the mean of {len(taken)} layers):")
    for name, ts in steps.items():
        n = 3 if "(x3)" in name else 1
        print(f"  {name:52} {mean(ts) * n:7.2f}   {mean(ts) * n / whole:5.0%}" + (f"   ({len(ts)} layers)" if len(ts) != len(taken) else ""))
print("\none product, each way (ms a layer, T operations a second):")
for name, rows in ways.items():
    t = mean([r[0] for r in rows])
    o = mean([r[1] for r in rows])
    print(f"  {name:60} {t:7.2f}" + (f"   {o / 1e12 / (t / 1e3):5.1f}" if o else ""))
if args.json:
    Path(args.json).write_text(json.dumps({
        "tokens": args.tokens - 1, "layers": spread,
        "steps_ms": {k: mean(v) * (3 if "(x3)" in k else 1) for k, v in steps.items()},
        "ways": {k: {"ms": mean([r[0] for r in v]), "operations": mean([r[1] for r in v])} for k, v in ways.items()}}, indent=1) + "\n")
