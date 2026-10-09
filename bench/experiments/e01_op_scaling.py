"""Experiment 1: which operations make a window of four tokens cost 1.6 times a single one?

Every kind of weight a pass reads, over all the layers that have it, in one evaluation, for windows of 1, 2, 4
and 8 tokens: milliseconds, and the rate at which its bytes are read. A kind whose time grows with the window
although its weights are read once is where a window loses.

    uv run python bench/experiments/e01_op_scaling.py --model <shard 1>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--windows", nargs="+", type=int, default=[1, 2, 4, 8])
p.add_argument("--rounds", type=int, default=7)
p.add_argument("--json")
args = p.parse_args()
model, _ = load(args.model)
engine = Engine(model)
engine.warm()
m = engine.m
d, hc = m.args.hidden_size, m.hc
dtype = mx.float16
rng = np.random.default_rng(0)


def width(mod) -> int:
    return mod.scales.shape[1] * mod.group_size if isinstance(mod, nn.QuantizedLinear) else mod.weight.shape[1]


def nbytes(mod) -> int:
    return sum(a.nbytes for a in (mod.weight, getattr(mod, "scales", None), getattr(mod, "biases", None)) if a is not None)


def kind(mod) -> str:
    return f"{mod.bits}-bit" if isinstance(mod, nn.QuantizedLinear) else str(mod.weight.dtype).split(".")[-1]


groups = {}  # name -> (function of T giving the arrays to evaluate, bytes of weights)


def linears(name, mods):
    mods = [x for x in mods if x is not None]
    widths = {width(x) for x in mods}
    assert len(widths) == 1, (name, widths)
    w = widths.pop()
    kinds = sorted({kind(x) for x in mods})

    def run(T):
        x = mx.random.normal((1, T, w)).astype(dtype)
        mx.eval(x)
        return lambda: [mod(x) for mod in mods]

    groups[f"{name} [{', '.join(kinds)}]"] = (run, sum(nbytes(x) for x in mods))


gdn = [m.layers[i].linear_attn for i in engine.gdn_layers]
att = [l.self_attn for l in m.layers if l.layer_type != "linear_attention"]
for part in ("in_proj_qkv", "in_proj_z", "out_proj", "in_proj_b", "in_proj_a"):
    linears(f"delta-net {part}", [getattr(x, part) for x in gdn])
for part in ("q_proj", "k_proj", "v_proj", "o_proj"):
    linears(f"attention {part}", [getattr(x, part) for x in att])
linears("router", [l.mlp.gate for l in m.layers])
for part in ("gate_proj", "up_proj", "down_proj"):
    linears(f"shared expert {part}", [getattr(l.mlp.shared_expert, part) for l in m.layers])
linears("output layer", [model.lm_head])


def residuals(which):
    def run(T):
        before = mx.random.normal((1, T, hc * d)).astype(dtype)
        out, inject = mx.random.normal((1, T, d)).astype(dtype), mx.random.normal((1, T, hc)).astype(dtype)
        mx.eval(before, out, inject)
        return lambda: [engine.hc[key][which](before, out, inject) for key in engine.hc]
    return run


hc_bytes = sum(getattr(gr, n).weight.nbytes for gr in [*(l.attn_hyper_connection for l in m.layers),
                                                 *(l.mlp_hyper_connection for l in m.layers), m.hyper_connection_mixer]
               for n in ("input_mix_weight_down", "input_mix_weight_up") if hasattr(gr, n))
groups["hyper-connections, fused kernels [float16]"] = (residuals(0), hc_bytes)
groups["hyper-connections, MLX's own [float16]"] = (residuals(1), hc_bytes)


def routed(T):
    x = mx.random.normal((1, T, d)).astype(dtype)
    idx = mx.array(np.stack([rng.choice(m.args.num_experts, m.args.num_experts_per_tok, replace=False) for _ in range(T)])[None])
    mx.eval(x, idx)
    return lambda: [l.mlp.switch_mlp(x, idx) for l in m.layers]


one_expert = sum(getattr(m.layers[0].mlp.switch_mlp, n).blocks.nbytes for n in ("gate_proj", "up_proj", "down_proj")) // m.args.num_experts
groups["routed experts [blocks]"] = (routed, None)


def best(fn, rounds):
    out = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        mx.eval(fn())
        out = min(out, time.perf_counter() - t0)
    return out * 1e3


def a_pass(T):
    ids = mx.zeros((1, T), dtype=mx.int32)

    def run():
        logits, w = engine._forward(ids)
        mx.eval(logits)
        for cache in engine.attn.values():
            cache.trim(T)
        return logits
    return run


rows = []
print(f"{'':46}{'weights':>9}" + "".join(f"{f'{T} tokens':>18}" for T in args.windows))
total = {T: 0.0 for T in args.windows}
for name, (make, size) in groups.items():
    row = {"what": name, "ms": {}}
    line = ""
    for T in args.windows:
        ms = best(make(T), args.rounds)
        b = size if size is not None else len(m.layers) * m.args.num_experts_per_tok * T * one_expert
        row["ms"][T], row["mb"] = round(ms, 2), round(b / 1e6)
        total[T] += ms if "MLX's own" not in name else 0
        line += f"{ms:8.2f} ms {b / ms / 1e6:4.0f} GB/s"
    rows.append(row)
    shown = f"{size / 1e6:6.0f} MB" if size is not None else " 14/token"
    print(f"{name:46}{shown:>9}{line}", flush=True)
print(f"{'sum (fused hyper-connections)':55}" + "".join(f"{total[T]:8.2f} ms         " for T in args.windows))
passes = {T: best(a_pass(T), args.rounds * 3) for T in args.windows}
print(f"{'a whole pass, GPU and all':55}" + "".join(f"{passes[T]:8.2f} ms         " for T in args.windows))
if args.json:
    Path(args.json).write_text(json.dumps({"rows": rows, "pass_ms": passes}, indent=1) + "\n")
