"""Experiment 40: a prompt read a layer at a time (bylayer.py), against the engine's read chunk by chunk.

On the real model as a machine of some memory, the experts its answers used most in memory as the server has
them, the files dropped from the page cache before every read, the ways taken in turns:

    chunks      the engine as it was: each chunk through all the layers (`piece=0`)
    layers      a layer at a time, a layer's experts read once and held (`ahead=False`): stage one
    ahead       and the next layer's experts read during this one: stage two

For each read: seconds, bytes read, the arrays at the peak above rest, and a digest of the whole state the prompt
leaves (every array of the conversation, its checkpoints, the first token, the guesses, some tokens of answer),
which must be the same for the three ways.

    uv run python bench/experiments/e40_prompt_by_layer.py --model <shard 1> --mtp <folder> [--tokens 8192]
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

from strata_mlx import paged  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402

WAYS = {"chunks": dict(piece=0), "layers": dict(ahead=False), "ahead": dict()}


def digest(e: Engine, first: int, answer: int) -> str:
    h = hashlib.sha256()

    def add(x):
        if x is not None:
            h.update(np.asarray(x).tobytes())

    add(e.context), add(e.ple_conv)
    for i in sorted(e.gdn):
        add(e.gdn[i][0]), add(e.gdn[i][1])
    for c in [*[e.attn[i] for i in sorted(e.attn)], e.mtp_cache]:
        for a in c.state:
            add(a)
    for cp in e.checkpoints:
        h.update(repr((cp.at, cp.pinned, cp.tail)).encode())
        add(cp.context), add(cp.ple_conv)
        for i in sorted(cp.gdn):
            add(cp.gdn[i][0]), add(cp.gdn[i][1])
    out = [first]
    h.update(repr((first, list(e.guesses))).encode())
    while len(out) < answer:
        out += e.step(answer - len(out))
    h.update(repr(out).encode())
    return h.hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--machine-memory", type=float, default=32)
    p.add_argument("--tokens", type=int, nargs="*", default=[8192])
    p.add_argument("--ways", nargs="*", default=list(WAYS), choices=list(WAYS))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--answer", type=int, default=16, help="tokens of answer in the digest; 0 for the state alone")
    p.add_argument("--rest", type=float, default=3, help="seconds between reads")
    p.add_argument("--json")
    args = p.parse_args()
    from drop_cache import in_memory
    from spec_decode import tokenize

    from strata_mlx.load import load
    from strata_mlx.mtp import load_mtp

    dtype = getattr(mx, args.dtype)
    model, _ = load(args.model, dtype, expert_memory=0.0)
    store = model.expert_store
    mtp = load_mtp(args.mtp, model.args.text, dtype)
    text = "\n\n".join((ROOT / name).read_text() for name in ("ROADMAP.md", "docs/OPTIMIZATION_RECORD.md", "README.md"))
    ids = tokenize(args.model, text)
    while len(ids) < max(args.tokens):
        ids = ids + ids[1:]
    Engine(model, mtp, 3, lookup=True).warm()
    files = sorted({t.path for tensors in store.layers.values() for t in tensors})
    profile = paged.load_profile(None)
    rows = []
    for n in args.tokens:
        print(f"\n-- {n} tokens, {args.dtype}, as a machine of {args.machine_memory:g} GiB", flush=True)
        for round_ in range(args.rounds):
            for way in args.ways:
                store.clear()
                budget = store.fit(int(args.machine_memory * 2**30))
                mx.clear_cache()
                store.preload(profile)
                for path in files:
                    in_memory(path, drop=True)
                e = Engine(model, mtp, 3, lookup=True, **WAYS[way])
                store.reset_stats()
                rest = mx.get_active_memory()
                mx.reset_peak_memory()
                steps, t0 = [], time.perf_counter()
                first = e.read(ids[:n], on_step=lambda: steps.append(time.perf_counter()))
                dt = time.perf_counter() - t0
                peak = mx.get_peak_memory() - rest
                gaps = [b - a for a, b in zip([t0, *steps[:-1]], steps)]
                row = {"tokens": n, "way": way, "round": round_ + 1, "s": round(dt, 2), "tok_s": round(n / dt, 1),
                       "read_gb": round(store.bytes_read / 1e9, 1), "disk_s": round(store.read_s, 1),
                       "waited_s": round(store.waited_s, 2),
                       "experts_gb": round(budget / 1e9, 1), "rest_gb": round(rest / 1e9, 2),
                       "peak_above_rest_gb": round(peak / 1e9, 2),
                       "longest_step_ms": round(max(gaps) * 1e3) if gaps else None,
                       "state": digest(e, first, args.answer) if args.answer else None}
                rows.append(row)
                print(f"  {way:<7} {dt:7.1f} s {row['tok_s']:6.0f} tok/s  read {row['read_gb']:6.1f} GB in {row['disk_s']:4.1f} s, waited {row['waited_s']:4.1f}  peak {row['peak_above_rest_gb']:5.2f} GB above "
                      f"{row['rest_gb']:.1f}  " + (f"longest step {row['longest_step_ms']} ms  " if gaps else "") + f"state {row['state']}", flush=True)
                del e
                time.sleep(args.rest)
        mine = [r for r in rows if r["tokens"] == n]
        states = {r["state"] for r in mine}
        print(f"  the state: {'the same in every read' if len(states) == 1 else f'DIFFERS: {sorted(states)}'}", flush=True)
        med = {w: statistics.median(r["s"] for r in mine if r["way"] == w) for w in args.ways}
        for w in args.ways:
            print(f"  {w:<7} median {med[w]:6.1f} s" + (f", {med['chunks'] / med[w]:.2f} of chunk by chunk" if "chunks" in med and w != "chunks" else ""), flush=True)
    print(f"{sum(in_memory(path) for path in files):.1f} GB of the experts' files in the page cache at the end", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
