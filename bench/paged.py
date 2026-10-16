"""Answer speed with only part of the experts in memory, the others read from the SSD (strata_mlx/paged.py).

For each memory budget: a few conversations one after the other in one process (the store starts empty and is
never emptied, as in a server), with and without the draft head. The OS is asked to forget the file before the
first conversation, and the last line says how much of it is back in memory: it must stay near nothing, or the
page cache of this machine has been answering for the disk of a smaller one.

`--machine-memory 16` gives the experts what a machine of that many GiB has left for them once everything else
is loaded, as the server does; `--dense arrays` holds the dense tensors as MLX's own quantized arrays, as a
machine that has all experts in memory does, where the default here is the file's blocks (1.7 GB less).
`--read <file>` reads that prompt as well, twice, and says how fast.

`--hold auto` (or a number of GiB) makes the machine smaller in fact and not only in the budget: another process
holds what the machine has beyond `--machine-memory` in random bytes for the whole run (bench/memory.py), so
swapping shows as it would on the small machine. Every row says this process's footprint as the kernel counts it;
the last says its peak, what the machine swapped out meanwhile, and whether the hold stayed whole, without which
the run says nothing about pressure. `--hold quiet` holds less by what this machine's other programs hold now
beyond a quiet macOS (6.1 GB, issue 6's Air with nothing open), so that the small machine has a quiet macOS in it
and not this one's desktop (e48 had 38 GB of it inside a 48 GB machine).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drop_cache import in_memory  # noqa: E402
from memory import Hold, footprint, machine_gib, own, swapped  # noqa: E402
from spec_decode import PROMPTS, STOP, tokenize  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402


QUIET_MACOS = 6.1e9  # what macOS held on issue 6's 32 GB Air with nothing else open


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--budget", type=float, help="GB of experts in memory")
    p.add_argument("--machine-memory", type=float, help="GiB of a machine: its experts get what the rest leaves")
    p.add_argument("--available", action="store_true", help="with --machine-memory, on a machine of that size: bound "
                   "the experts by what it has free or in its file cache now (the server's --expert-memory available)")
    p.add_argument("--margin", type=float, help="with --available: GB of what is available left to other programs")
    p.add_argument("--dense", choices=["blocks", "arrays"], default="blocks")
    p.add_argument("--engines", nargs="+", choices=["one", "head"], default=["one", "head"])
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--prompts", nargs="+", default=["prose", "code", "reasoning", "edit"])
    p.add_argument("--read", help="a long prompt in a file, read twice")
    p.add_argument("--hold", help="GiB held by another process for the run; 'auto': all but --machine-memory; "
                   "'quiet': less what this machine's other programs hold beyond a quiet macOS")
    p.add_argument("--json")
    args = p.parse_args()
    if (args.budget is None) == (args.machine_memory is None):
        p.error("one of --budget and --machine-memory")
    hold = None
    if args.hold:
        if args.hold in ("auto", "quiet") and not args.machine_memory:
            p.error(f"--hold {args.hold} needs --machine-memory")
        hold = round(machine_gib() - args.machine_memory, 1) if args.hold in ("auto", "quiet") else float(args.hold)
        if args.hold == "quiet":
            from strata_mlx.paged import available_memory

            free, cached = available_memory()
            others = machine_gib() * 2**30 - free - cached  # what nothing frees for the asking
            extra = max(0.0, (others - QUIET_MACOS) / 2**30)
            hold = round(hold - extra, 1)
            print(f"other programs hold {others / 1e9:.1f} GB beyond the free memory and the file cache; a quiet macOS "
                  f"{QUIET_MACOS / 1e9:.1f}: {extra:.1f} GiB less held", flush=True)
    swapped_before, own_before = swapped()[0], own()
    if hold:
        print(f"holding {hold} GiB in another process", flush=True)
        with Hold(hold) as h:
            rows = run(args)
        print(f"the hold: {h.report()}", flush=True)
    else:
        h, rows = None, run(args)
    now, peak = footprint()
    mine = own()
    summary = {"summary": True, "footprint_peak_gb": round(peak / 1e9, 2),
               "swapped_out_gb": round((swapped()[0] - swapped_before) / 1e9, 2),
               "own_compressed_peak_gb": round(mine["compressed_peak"] / 1e9, 2),
               "own_compressed_during_gb": round((mine["compressed_lifetime"] - own_before["compressed_lifetime"]) / 1e9, 2),
               **(h.report() if h else {})}
    rows.append(summary)
    print(f"this process: footprint peak {summary['footprint_peak_gb']} GB, {summary['own_compressed_during_gb']} GB of it "
          f"compressed or swapped out during the run (at most {summary['own_compressed_peak_gb']} GB at once); the machine "
          f"swapped out {summary['swapped_out_gb']} GB meanwhile" + ("" if not h else "; the hold " +
          ("stayed whole" if h.whole else "did NOT stay whole: the run says nothing about pressure")), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


def run(args) -> list:
    if args.available:  # counted before the model's files are read: their pages are file cache for a while
        from strata_mlx.paged import available_memory

        free, cached = available_memory()
    model, _ = load(args.model, expert_memory=args.budget or 0.0, dense_blocks=None if args.dense == "blocks" else ())
    store = model.expert_store
    mtp = load_mtp(args.mtp, model.args.text)
    engines = {}
    if "one" in args.engines:
        engines["one token a pass"] = Engine(model)
    if "head" in args.engines:
        engines["draft head"] = Engine(model, mtp, 3, lookup=True)
    if args.machine_memory:
        ruled = store.fit(int(args.machine_memory * 2**30))
        if args.available:  # on the machine itself: no more than it had free or in its file cache (#6)
            store.fit(int(args.machine_memory * 2**30), free + cached, None if args.margin is None else int(args.margin * 1e9))
            print(f"{free / 1e9:.1f} GB free and {cached / 1e9:.1f} GB of file cache before loading; by the rule alone the experts "
                  f"would get {ruled / 1e9:.2f} GB", flush=True)
        args.budget = round(store.budget / 1e9, 2)
    for e in engines.values():
        e.warm()
    in_memory(args.model, drop=True)
    rest = mx.get_active_memory() / 1e9 - store.held / 1e9
    print(f"budget {args.budget} GB of experts; everything else {rest:.2f} GB; dense tensors as {args.dense}", flush=True)
    rows = []
    for round_ in ("first", "again"):  # again: the same conversations with the store as the first round left it
        for name in args.prompts:
            prompt = tokenize(args.model, f"<|im_start|>user\n{PROMPTS[name]}<|im_end|>\n<|im_start|>assistant\n")
            for label, e in engines.items():
                e.reset()
                store.reset_stats()
                t0 = time.perf_counter()
                out = [e.read(prompt)]
                t1 = time.perf_counter()
                read_prompt = store.bytes_read
                store.reset_stats()
                while len(out) < args.tokens and out[-1] not in STOP:
                    out += e.step(args.tokens - len(out), STOP)
                dt, n = time.perf_counter() - t1, len(out) - 1
                row = {"budget_gb": args.budget, "machine_gib": args.machine_memory, "dense": args.dense,
                       "everything_else_gb": round(rest, 2), "round": round_, "prompt": name, "engine": label,
                       "prompt_tokens": len(prompt), "prompt_tok_s": round(len(prompt) / (t1 - t0), 1),
                       "prompt_read_gb": round(read_prompt / 1e9, 2), "tok_s": round(n / dt, 1),
                       "misses_per_token": round(store.misses / n, 1), "read_mb_per_token": round(store.bytes_read / n / 1e6, 1),
                       "read_share": round(store.read_s / dt, 2), "tokens_per_pass": round(e.stats.tokens / max(e.stats.steps, 1), 2),
                       "held_gb": round(store.held / 1e9, 2), "active_gb": round(mx.get_active_memory() / 1e9, 1),
                       "footprint_gb": round(footprint()[0] / 1e9, 1)}
                rows.append(row)
                print(f"{round_:5} {name:9} {label:16} {row['tok_s']:5.1f} tok/s  {row['misses_per_token']:6.1f} misses/token  "
                      f"{row['read_mb_per_token']:6.1f} MB/token  {row['read_share']:4.0%} of the time reading  "
                      f"{row['tokens_per_pass']:4.2f} tokens/pass | prompt {row['prompt_tok_s']:5.1f} tok/s", flush=True)
    if args.read:
        ids = tokenize(args.model, Path(args.read).read_text())
        e = list(engines.values())[-1]
        for _ in range(2):
            e.reset()
            mx.reset_peak_memory()
            store.reset_stats()
            t0 = time.perf_counter()
            first = e.read(ids)
            dt = time.perf_counter() - t0
            row = {"budget_gb": args.budget, "machine_gib": args.machine_memory, "dense": args.dense,
                   "read_tokens": len(ids), "read_tok_s": round(len(ids) / dt, 1),
                   "read_gb": round(store.bytes_read / 1e9, 2), "read_share": round(store.read_s / dt, 2),
                   "first_token": int(first),
                   "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "active_gb": round(mx.get_active_memory() / 1e9, 2),
                   "footprint_peak_gb": round(footprint()[1] / 1e9, 2)}
            rows.append(row)
            print(f"read {len(ids)} tokens at {row['read_tok_s']:.0f} tok/s, {row['read_gb']} GB from the disk in "
                  f"{row['read_share']:.0%} of the time; arrays at the "
                  f"peak {row['peak_gb']} GB, after {row['active_gb']} GB; first token of the answer {first}", flush=True)
            time.sleep(4)
    print(f"held {store.held / 1e9:.2f} GB of experts; all arrays {mx.get_active_memory() / 1e9:.1f} GB, "
          f"peak {mx.get_peak_memory() / 1e9:.1f} GB; {in_memory(args.model):.1f} GB of the file in the page cache", flush=True)
    return rows


if __name__ == "__main__":
    main()
