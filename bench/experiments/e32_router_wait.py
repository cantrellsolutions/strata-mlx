"""Experiment 32: what a chunk of a prompt pays for waiting, at every layer, for the routers' choice.

e31: a layer's ten chosen experts, their inputs given, take 20 ms (48 layers: 1.0 s of a chunk of 1,024
tokens), and the product in them already runs at what MLX's own grouped products reach. e30: in a pass they
cost 1.7-2.1 s. The way prompts went sorted the pairs by expert on the CPU, so the pass stopped at every layer
until the GPU had the routers' choice. Here the same chunk is read four ways, in turns:

    sorting on the CPU, as the engine read it then
    the same with the experts' output made to finish too: two stops a layer, and the time between them says
        what the experts and what the rest of a layer take inside a real pass
    with the choices known beforehand (taken from a pass before: the same prompt chooses the same), so that
        nothing stops and the pass is one graph: what sorting on the GPU could be worth at most
    sorting on the GPU (blocks.tiles_of), as the engine reads it since

    uv run python bench/experiments/e32_router_wait.py --model <shard 1> [--json out.json]
(Run on the code of commit 26e78a8. Since then the 2-bit product this takes apart, blocks.packed_matmul, has been
replaced by blocks.tile_matmul, and these names are gone from blocks.py.)
"""

import argparse
import json
import sys
import time
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
p.add_argument("--rounds", type=int, default=4)
p.add_argument("--json")
args = p.parse_args()

model, _ = load(args.model)
text = "<|im_start|>user\nSay in two sentences what this module is for.\n\n```python\n" + EDIT_ME * 9 + "```<|im_end|>\n<|im_start|>assistant\n"
prompt = tokenize(args.model, text)[: args.tokens]
engine = Engine(model)
packed_glu = B._packed_glu
known, at = [], [0]
marks = []


def on_cpu(x, indices, up, gate, down, keep=None):
    """blocks._packed_glu as it was: the pass waits here for the routers' choice."""
    idx = np.array(indices)
    k, flat = idx.shape[-1], idx.reshape(-1)
    order = np.argsort(flat, kind="stable")
    used, counts = np.unique(flat, return_counts=True)
    padded = -(-counts // B.TOKENS_TILE) * B.TOKENS_TILE
    slot = np.repeat(np.cumsum(padded) - padded, counts) + np.arange(len(flat)) - np.repeat(np.cumsum(counts) - counts, counts)
    token = np.zeros(int(padded.sum()), dtype=np.uint32)
    token[slot] = order // k
    where = np.empty(len(flat), dtype=np.uint32)
    where[order] = slot
    tiles, token, where = mx.array(np.repeat(used, padded // B.TOKENS_TILE).astype(np.uint32)), mx.array(token), mx.array(where)
    if keep is not None:
        keep.append((tiles, token, where))
    return _products(x, indices, up, gate, down, tiles, token, where)


def recording(x, indices, up, gate, down):
    return on_cpu(x, indices, up, gate, down, keep=known)


def _products(x, indices, up, gate, down, tiles, token, where):
    a = x.reshape(-1, x.shape[-1])[token]

    def product(lin, a):
        return B.packed_matmul(lin.blocks, (lin.num_experts, lin.output_dims, lin.input_dims), a, tiles)

    h = B.swiglu(product(gate, a), product(up, a)).astype(x.dtype)
    out = product(down, h)[where].astype(x.dtype)
    return out.reshape(*indices.shape, down.output_dims)


def knowing(x, indices, up, gate, down):
    """blocks._packed_glu with the sorting done before: nothing here waits for the GPU."""
    tiles, token, where = known[at[0]]
    at[0] += 1
    return _products(x, indices, up, gate, down, tiles, token, where)


def stopping(x, indices, up, gate, down):
    mx.eval(x, indices)
    marks.append(time.perf_counter())
    out = on_cpu(x, indices, up, gate, down)
    mx.eval(out)
    marks.append(time.perf_counter())
    return out


def read(way):
    engine.reset()
    mx.eval(engine.m.embed_tokens(mx.array([prompt[:4]])))
    at[0] = 0
    marks.clear()
    B._packed_glu = way
    t0 = time.perf_counter()
    logits, w = engine._forward(mx.array([prompt]), last_only=True)
    mx.eval(logits)
    engine._commit(w, len(prompt))
    mx.eval(*[a for v in engine.gdn.values() for a in v])
    dt = (time.perf_counter() - t0) * 1e3
    B._packed_glu = packed_glu
    return dt, int(mx.argmax(logits[0, -1]).item()), t0


def median(xs):
    return sorted(xs)[len(xs) // 2]


_, first, _ = read(recording)
assert len(known) == len(model.model.layers), "this file's experts do not go the 2-bit way"
WAYS = (("sorting on the CPU", on_cpu), ("sorting on the CPU, two stops a layer", stopping),
        ("the choices known beforehand", knowing), ("sorting on the GPU", packed_glu))
for _, way in WAYS:
    read(way)
times = {name: [] for name, _ in WAYS}
inside, between = [], []
mx.reset_peak_memory()
for _ in range(args.rounds):
    for name, way in WAYS:
        time.sleep(6)
        dt, token, t0 = read(way)
        assert token == first, (name, token, first)
        times[name].append(dt)
        if way is stopping:
            inside.append(sum(b - a for a, b in zip(marks[::2], marks[1::2])) * 1e3)
            between.append(dt - inside[-1])
n = len(prompt)
print(f"\n{n} tokens of a prompt, one piece (ms, the median of {args.rounds}; the same first token of the answer each way):")
for name, ts in times.items():
    print(f"  {name:40} {median(ts):7.0f} ms  {n / median(ts) * 1e3:5.0f} tok/s")
print(f"  with two stops a layer: the ten chosen experts {median(inside):.0f} ms, all the rest {median(between):.0f} ms")
print(f"  peak memory {mx.get_peak_memory() / 1e9:.1f} GB")
if args.json:
    Path(args.json).write_text(json.dumps({"tokens": n, "ms": {k: median(v) for k, v in times.items()},
                                           "experts_ms": median(inside), "rest_ms": median(between)}, indent=1) + "\n")
