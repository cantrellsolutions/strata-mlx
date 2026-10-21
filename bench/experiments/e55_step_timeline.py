"""e55: where a decoding step's time goes, and what the memory bandwidth really is.

Nothing in the engine changes: its methods are wrapped for timing in this process only. A step is cut into
  build    building the trunk's graph (`_forward`; MLX is lazy, so the GPU has nothing yet), the n-gram rows
           read on the CPU inside it counted apart as `ngram`
  choose   `_choose`: the wait for the trunk's logits (the GPU runs the pass) and picking the tokens
  commit   `_commit`
  draft    `_draft`, all of it; of it, the time inside mx.eval is `draft_wait`
and every mx.eval in the step is summed as `wait`. The CPU's time with nothing submitted is step - wait: an
upper bound on the GPU idling (work started by async_eval may run in it).

The bandwidth is measured twice: mx.sum over 5.5 GB (the old ruler, 13.9 ms) and a kernel that only reads, vectors
of 16 bytes, at a few grid sizes.

    uv run python bench/experiments/e55_step_timeline.py --model <Q2_0 shard 1> --mtp <folder> --json out.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spec_decode import PROMPTS, STOP, tokenize  # noqa: E402

import mlx.core as mx  # noqa: E402

from strata_mlx import engine as engine_mod  # noqa: E402
from strata_mlx import load as load_mod  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402
from strata_mlx.sampler import Sampling  # noqa: E402

GB = 1e9


class Clock:
    """Sums of time per phase within the step under way."""

    def __init__(self):
        self.now = None

    def add(self, key, dt):
        if self.now is not None:
            self.now[key] = self.now.get(key, 0.0) + dt


CLOCK = Clock()


def timed(fn, key):
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            CLOCK.add(key, time.perf_counter() - t0)
    return wrapper


def instrument():
    """Wrap what a step does, in this process only."""
    real_eval = mx.eval
    in_draft = [False]

    def eval_(*a, **k):
        t0 = time.perf_counter()
        try:
            return real_eval(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            CLOCK.add("wait", dt)
            if in_draft[0]:
                CLOCK.add("draft_wait", dt)

    engine_mod.mx.eval = eval_  # the engine's module reads mx.eval at call time

    def draft(fn):
        def wrapper(*a, **k):
            in_draft[0] = True
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                in_draft[0] = False
                CLOCK.add("draft", time.perf_counter() - t0)
        return wrapper

    Engine._forward = timed(Engine._forward, "build")
    Engine._choose = timed(Engine._choose, "choose")
    Engine._commit = timed(Engine._commit, "commit")
    Engine._draft = draft(Engine._draft)
    load_mod.PleTable.read = timed(load_mod.PleTable.read, "ngram")
    return real_eval


def decode_timed(engine, prompt, tokens):
    engine.reset()
    engine.sampling = Sampling()
    out = [engine.read(prompt)]
    steps = []
    while len(out) < tokens and out[-1] not in STOP:
        CLOCK.now = {}
        t0 = time.perf_counter()
        settled = engine.step(tokens - len(out), STOP)
        CLOCK.now["step"] = time.perf_counter() - t0
        CLOCK.now["settled"] = len(settled)
        steps.append(CLOCK.now)
        CLOCK.now = None
        out += settled
    return out, steps


def summary(steps):
    keys = ["step", "build", "ngram", "choose", "commit", "draft", "draft_wait", "wait"]
    s = {k: round(statistics.median([x.get(k, 0.0) for x in steps]) * 1e3, 2) for k in keys}
    idle = [x["step"] - x.get("wait", 0.0) for x in steps]
    s["cpu_no_wait"] = round(statistics.median(idle) * 1e3, 2)
    s["tokens_per_step"] = round(sum(x["settled"] for x in steps) / len(steps), 2)
    s["tok_s"] = round(sum(x["settled"] for x in steps) / sum(x["step"] for x in steps), 1)
    s["steps"] = len(steps)
    return s


def bandwidth(real_eval, total=5.5 * GB, part=512 << 20):
    n = int(total // part)
    arrays = [mx.random.randint(0, 1 << 30, (part // 4,), dtype=mx.uint32) for _ in range(n)]
    real_eval(*arrays)
    nbytes = n * part
    kernel = mx.fast.metal_kernel(
        name="read_only",
        input_names=["inp"],
        output_names=["out"],
        source="""
            uint gid = thread_position_in_grid.x;
            uint stride = threads_per_grid.x;
            const device uint4* p = (const device uint4*)inp;
            uint n4 = inp_shape[0] / 4;
            uint4 acc = uint4(0);
            for (uint i = gid; i < n4; i += stride) acc ^= p[i];
            if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) out[0] = 1;
        """,
    )

    def read_all(threads):
        outs = [kernel(inputs=[a], grid=(threads, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(1,)], output_dtypes=[mx.uint32])[0] for a in arrays]
        real_eval(*outs)

    def best(fn, rounds=12):
        fn()
        times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times), statistics.median(times)

    rows = {}
    lo, med = best(lambda: real_eval(*[mx.sum(a) for a in arrays]))
    rows["mx.sum"] = {"ms_min": round(lo * 1e3, 2), "ms_median": round(med * 1e3, 2), "GB_s": round(nbytes / lo / GB, 1)}
    for threads in (40 * 1024, 40 * 4096, 40 * 16384, 40 * 65536):
        lo, med = best(lambda: read_all(threads))
        rows[f"read kernel, {threads} threads"] = {"ms_min": round(lo * 1e3, 2), "ms_median": round(med * 1e3, 2),
                                                   "GB_s": round(nbytes / lo / GB, 1)}
    del arrays
    return {"bytes": nbytes, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--tokens", type=int, default=160)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    p.add_argument("--json")
    args = p.parse_args()

    real_eval = instrument()
    result = {"bandwidth": bandwidth(real_eval)}
    for name, row in result["bandwidth"]["rows"].items():
        print(f"{name:<30} {row['ms_min']:>7} ms  {row['GB_s']:>6} GB/s", flush=True)

    model, _ = load(args.model)
    mtp = load_mtp(args.mtp, model.args.text, mx.float16)
    engines = {"one token a pass": Engine(model), "draft head": Engine(model, mtp)}
    for e in engines.values():
        e.warm()
    prompts = {n: tokenize(args.model, f"<|im_start|>user\n{PROMPTS[n]}<|im_end|>\n<|im_start|>assistant\n")
               for n in args.prompts}
    rows = []
    for r in range(args.rounds):
        for label, e in engines.items():
            for name, prompt in prompts.items():
                _, steps = decode_timed(e, prompt, args.tokens)
                s = summary(steps[2:])  # the first steps may still compile
                s.update(round=r, engine=label, prompt=name)
                rows.append(s)
                print(f"r{r} {label:<17} {name:<9} step {s['step']:>6} | build {s['build']:>5} ngram {s['ngram']:>4} "
                      f"choose {s['choose']:>6} commit {s['commit']:>5} draft {s['draft']:>5} (wait {s['draft_wait']:>5}) "
                      f"| all waits {s['wait']:>6} CPU-only {s['cpu_no_wait']:>5} | {s['tokens_per_step']} tok/step "
                      f"{s['tok_s']} tok/s", flush=True)
    result["steps"] = rows
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
