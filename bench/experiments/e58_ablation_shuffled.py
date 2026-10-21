"""e58: what each part of a pass costs now, by leaving it out (e06 again, as e57 measures).

Each part in turn is replaced by something that reads no weights, in an engine of its own sharing the weights;
every (engine, size) pass is shuffled with every other, each engine's stand-in put in for its pass and taken out
after (parts inside compiled functions keep the stand-in they were traced with), and the lower quartile taken.
The machine slows under long load; shuffled single passes compare at speed (e57).

    uv run python bench/experiments/e58_ablation_shuffled.py --model <Q2_0 shard 1> --json out.json
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from strata_mlx import engine as engine_mod  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402


def swap(owner, name, new):
    old = getattr(owner, name)
    setattr(owner, name, new)
    return lambda: setattr(owner, name, old)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rounds", type=int, default=120)
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model)
    m = model.model
    d, layers = m.args.hidden_size, m.layers
    gdn = [l.linear_attn for l in layers if l.layer_type == "linear_attention"]
    att = [l for l in layers if l.layer_type != "linear_attention"]

    def zeros_out(mod):
        n = getattr(mod, "output_dims", None) or mod.weight.shape[0]
        return lambda x, n=n: mx.zeros((*x.shape[:-1], n), dtype=x.dtype)

    def linears(owners, names):
        return lambda: [swap(o, n, zeros_out(getattr(o, n))) for o in owners for n in names]

    def routed():
        return [swap(l.mlp, "switch_mlp", lambda x, idx: mx.zeros((*idx.shape, d), dtype=x.dtype)) for l in layers]

    def shared():
        return [swap(l.mlp, "shared_expert", lambda x: mx.zeros_like(x)) for l in layers]

    def recurrence():
        return [swap(engine_mod, "gated_delta_update",
                     lambda q, k, v, a, b, A, dt, rec, mask: (mx.zeros_like(v), rec))]

    def attention():
        return [swap(l, "self_attn", lambda x, *a, **k: mx.zeros_like(x)) for l in att]

    parts = {
        "nothing left out": None,
        "nothing left out, control": None,
        "routed experts": routed,
        "shared experts": shared,
        "routers": linears([l.mlp for l in layers], ["gate"]),
        "delta-net projections": linears(gdn, ["in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b"]),
        "delta-net recurrence": recurrence,
        "attention, all of it": attention,
        "output layer": linears([model], ["lm_head"]),
    }
    engines = {}
    for name, patch in parts.items():
        undo = patch() if patch else []
        e = Engine(model)
        e.warm()
        e.read(list(range(1000, 1300)))
        for u in reversed(undo):
            u()
        engines[name] = e

    def a_pass(name, T, i):
        e, patch = engines[name], parts[name]
        undo = patch() if patch else []
        ids = mx.array([[1000 + (i * 7 + j) % 500 for j in range(T)]])
        t0 = time.perf_counter()
        logits, w = e._forward(ids)
        mx.eval(logits)
        e._commit(w, T)
        mx.eval(*[c.keys for c in e.attn.values() if c.keys is not None])
        dt = time.perf_counter() - t0
        for u in reversed(undo):
            u()
        return dt

    times = {(name, T): [] for name in parts for T in (1, 4)}
    order = list(times) * args.rounds
    random.Random(58).shuffle(order)
    for i, key in enumerate(order):
        times[key].append(a_pass(*key, i))
    q = {k: statistics.quantiles(v, n=4)[0] * 1e3 for k, v in times.items()}
    rows = []
    base = {T: q[("nothing left out", T)] for T in (1, 4)}
    for name in parts:
        row = {"left_out": name, "ms": {T: round(q[(name, T)], 2) for T in (1, 4)},
               "saved": {T: round(base[T] - q[(name, T)], 2) for T in (1, 4)}}
        rows.append(row)
        print(f"{name:<26} 1 token {row['ms'][1]:6.2f} ms ({-row['saved'][1]:+5.2f})   "
              f"4 tokens {row['ms'][4]:6.2f} ms ({-row['saved'][4]:+5.2f})", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
