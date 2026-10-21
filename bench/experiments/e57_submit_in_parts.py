"""e57: a pass handed to the GPU in parts while the rest is built (Engine(submit_every=N)).

One process, engines sharing the weights: submit_every 0 (as now) twice, the second the control, and 6, 12, 16.
Each round decodes every prompt with every engine in a shuffled order; a token a pass and with the draft head
separately. Reported: the median over rounds of each decode's mean step time, and whether its tokens are the
reference's (they must be: the graph is the same).

    uv run python bench/experiments/e57_submit_in_parts.py --model <Q2_0 shard 1> --mtp <folder> --json out.json
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spec_decode import PROMPTS, STOP, tokenize  # noqa: E402

import mlx.core as mx  # noqa: E402

from strata_mlx import blocks  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402
from strata_mlx.sampler import Sampling  # noqa: E402

# submit_every, gate and up in one kernel
SETTINGS = {"now": (0, False), "control": (0, False), "every 6": (6, False), "every 12": (12, False),
            "every 16": (16, False), "gate+up": (0, True)}
# the second round of passes: around the best of the first, and the two together
SETTINGS_2 = {"now": (0, False), "control": (0, False), "every 3": (3, False), "every 4": (4, False),
              "every 6": (6, False), "every 8": (8, False), "every 6 + gate+up": (6, True)}


def decode(engine, prompt, tokens):
    engine.reset()
    engine.sampling = Sampling()
    out = [engine.read(prompt)]
    steps, t0 = 0, time.perf_counter()
    while len(out) < tokens and out[-1] not in STOP:
        out += engine.step(tokens - len(out), STOP)
        steps += 1
    dt = time.perf_counter() - t0
    return out, dt / steps * 1e3, (len(out) - 1) / dt


def passes(engines, rounds, sizes=(1, 4)):
    """Single passes, every (engine, size) in a shuffled order, each timed from building to the GPU done; the
    attention caches cut back after each, as warm() does. The lower quartile per (engine, size): the machine
    slows under long load (bench/quiet.py), and a shuffled order with the lower quartile compares at speed."""
    from strata_mlx.engine import _arrays
    times = {(name, T): [] for name in engines for T in sizes}
    order = list(times) * rounds
    random.Random(570).shuffle(order)
    for name, T in order:
        e = engines[name]
        t0 = time.perf_counter()
        logits, w = e._forward(mx.zeros((1, T), dtype=mx.int32) + 1000)
        mx.eval(logits, *_arrays([list(w.gdn), w.ple_conv, w.residual]))
        times[(name, T)].append(time.perf_counter() - t0)
        for cache in e.attn.values():
            cache.trim(T)
    return {k: statistics.quantiles(v, n=4)[0] * 1e3 for k, v in times.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--tokens", type=int, default=160)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--passes", type=int, default=0, help="time single passes this many rounds, not answers")
    p.add_argument("--second", action="store_true", help="the second round's settings")
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model)
    mtp = load_mtp(args.mtp, model.args.text, mx.float16)
    prompts = {n: tokenize(args.model, f"<|im_start|>user\n{PROMPTS[n]}<|im_end|>\n<|im_start|>assistant\n")
               for n in PROMPTS}
    rows = []
    for kind in (("a token a pass",) if args.passes else ("a token a pass", "draft head")):
        engines = {}
        for name, (k, one) in (SETTINGS_2 if args.second else SETTINGS).items():
            blocks.GATE_UP_ONE_KERNEL = one  # read when an engine's compiled halves are traced, in warm()
            engines[name] = (Engine(model, submit_every=k) if kind == "a token a pass"
                             else Engine(model, mtp, submit_every=k))
            engines[name].warm()
        blocks.GATE_UP_ONE_KERNEL = False
        if args.passes:
            for e in engines.values():
                e.reset()
                e.read(list(range(1000, 1300)))  # a short conversation for the passes to follow
            q = passes(engines, args.passes)
            for name in engines:
                rows.append({"kind": kind, "setting": name, "pass_ms": {T: round(q[(name, T)], 2) for T in (1, 4)}})
                print(f"{kind:<15} {name:<9} pass, lower quartile: 1 token {q[(name, 1)]:6.2f} ms, "
                      f"4 tokens {q[(name, 4)]:6.2f} ms", flush=True)
            continue
        ref = {n: decode(engines["now"], pr, args.tokens)[0] for n, pr in prompts.items()}
        times = {(name, n): [] for name in engines for n in prompts}
        same = {name: True for name in engines}
        order = [(name, n) for name in engines for n in prompts] * args.rounds
        random.Random(57).shuffle(order)
        for name, n in order:
            out, step_ms, tps = decode(engines[name], prompts[n], args.tokens)
            times[(name, n)].append((step_ms, tps))
            same[name] = same[name] and out == ref[n]
        for name in engines:
            step = statistics.median([statistics.median([t[0] for t in times[(name, n)]]) for n in prompts])
            tps = [round(statistics.median([t[1] for t in times[(name, n)]]), 1) for n in prompts]
            rows.append({"kind": kind, "setting": name, "step_ms": round(step, 2), "tok_s": tps, "same": same[name]})
            print(f"{kind:<15} {name:<9} step {step:6.2f} ms  tok/s {tps}  same tokens: {same[name]}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
