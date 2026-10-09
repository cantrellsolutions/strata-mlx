"""Experiment 6: what each part of a pass really costs, by leaving it out of a real pass.

Timed alone, parts of a pass do not add up to the pass: alone a kernel pays for the CPU's work on it (a commit
of the command buffer for every large input), in a pass the GPU is what is waited for. So here each part in turn
is replaced by something that reads no weights, and the pass is timed without it, for one token and for four.

    uv run python bench/experiments/e06_ablation.py --model <shard 1>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--passes", type=int, default=80)
p.add_argument("--windows", nargs="+", type=int, default=[1, 4])
p.add_argument("--json")
args = p.parse_args()
model, _ = load(args.model)
m = model.model
d, hc = m.args.hidden_size, m.hc
layers = m.layers


def zeros_like_output(mod):
    n = mod.weight.shape[0]
    return lambda x, n=n: mx.zeros((*x.shape[:-1], n), dtype=x.dtype)


def swap(owner, name, new):
    old = getattr(owner, name)
    setattr(owner, name, new)
    return lambda: setattr(owner, name, old)


def linears(owners, names):
    def patch():
        undo = [swap(o, n, zeros_like_output(getattr(o, n))) for o in owners for n in names]
        return lambda: [u() for u in undo]
    return patch


def routed():
    undo = [swap(l.mlp, "switch_mlp", lambda x, idx: mx.zeros((*idx.shape, d), dtype=x.dtype)) for l in layers]
    return lambda: [u() for u in undo]


def shared():
    undo = [swap(l.mlp, "shared_expert", lambda x: mx.zeros_like(x)) for l in layers]
    return lambda: [u() for u in undo]


def no_residuals(engine):
    def stub(before, out, inject):
        return before[..., :d] + out, before, mx.zeros((*before.shape[:-1], hc), dtype=before.dtype)
    engine.hc = {key: (stub, stub) for key in engine.hc}


gdn = [l.linear_attn for l in layers if l.layer_type == "linear_attention"]
att = [l.self_attn for l in layers if l.layer_type != "linear_attention"]
ABLATIONS = {
    "nothing left out": (None, None),
    "routed experts": (routed, None),
    "shared experts": (shared, None),
    "hyper-connections": (None, no_residuals),
    "delta-net projections": (linears(gdn, ["in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b"]), None),
    "attention projections": (linears(att, ["q_proj", "k_proj", "v_proj", "o_proj"]), None),
    "routers": (linears([l.mlp for l in layers], ["gate"]), None),
    "output layer": (linears([model], ["lm_head"]), None),
    "nothing left out, again": (None, None),
}


def measure(engine, T):
    times = []
    engine.reset()
    engine.read(list(range(1000, 1200)))
    for i in range(args.passes):
        ids = mx.array([[1000 + (i * 7 + j) % 500 for j in range(T)]])
        t0 = time.perf_counter()
        logits, w = engine._forward(ids)
        mx.eval(logits)
        engine._commit(w, T)
        times.append((time.perf_counter() - t0) * 1e3)
    return sorted(times)[len(times) // 4]


rows, base = [], None
for name, (patch, tweak) in ABLATIONS.items():
    undo = patch() if patch else None
    engine = Engine(model)
    if tweak:
        tweak(engine)
    engine.warm()
    ms = {T: measure(engine, T) for T in args.windows}
    if undo:
        undo()
    base = base or ms
    rows.append({"left_out": name, "ms": ms})
    print(f"{name:26}" + "".join(f"  {T} token{'s' if T > 1 else ' '} {ms[T]:6.2f} ms ({ms[T] - base[T]:+6.2f})" for T in args.windows),
          flush=True)
if args.json:
    Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
