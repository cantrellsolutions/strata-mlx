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

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402
from strata_mlx.sampler import Sampling  # noqa: E402

SETTINGS = {"now": 0, "control": 0, "every 6": 6, "every 12": 12, "every 16": 16}


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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--tokens", type=int, default=160)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model)
    mtp = load_mtp(args.mtp, model.args.text, mx.float16)
    prompts = {n: tokenize(args.model, f"<|im_start|>user\n{PROMPTS[n]}<|im_end|>\n<|im_start|>assistant\n")
               for n in PROMPTS}
    rows = []
    for kind in ("a token a pass", "draft head"):
        engines = {name: (Engine(model, submit_every=k) if kind == "a token a pass"
                          else Engine(model, mtp, submit_every=k)) for name, k in SETTINGS.items()}
        for e in engines.values():
            e.warm()
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
