"""Answer speed with only part of the experts in memory, the others read from the SSD (strata_mlx/paged.py).

For each memory budget: a few conversations one after the other in one process (the store starts empty and is
never emptied, as in a server), with and without the draft head. The OS is asked to forget the file before the
first conversation, and the last line says how much of it is back in memory: it must stay near nothing, or the
page cache of this machine has been answering for the disk of a smaller one.

`--machine-memory 16` gives the experts what a machine of that many GiB has left for them once everything else
is loaded, as the server does; `--dense arrays` holds the dense tensors as MLX's own quantized arrays, as a
machine that has all experts in memory does, where the default here is the file's blocks (1.7 GB less).
`--read <file>` reads that prompt as well, twice, and says how fast.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drop_cache import in_memory  # noqa: E402
from spec_decode import PROMPTS, STOP, tokenize  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--budget", type=float, help="GB of experts in memory")
    p.add_argument("--machine-memory", type=float, help="GiB of a machine: its experts get what the rest leaves")
    p.add_argument("--dense", choices=["blocks", "arrays"], default="blocks")
    p.add_argument("--engines", nargs="+", choices=["one", "head"], default=["one", "head"])
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--prompts", nargs="+", default=["prose", "code", "reasoning", "edit"])
    p.add_argument("--read", help="a long prompt in a file, read twice")
    p.add_argument("--json")
    args = p.parse_args()
    if (args.budget is None) == (args.machine_memory is None):
        p.error("one of --budget and --machine-memory")
    model, _ = load(args.model, expert_memory=args.budget or 0.0, dense_blocks=None if args.dense == "blocks" else ())
    store = model.expert_store
    mtp = load_mtp(args.mtp, model.args.text)
    engines = {}
    if "one" in args.engines:
        engines["one token a pass"] = Engine(model)
    if "head" in args.engines:
        engines["draft head"] = Engine(model, mtp, 3, lookup=True)
    if args.machine_memory:
        args.budget = round(store.fit(int(args.machine_memory * 2**30)) / 1e9, 2)
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
                       "held_gb": round(store.held / 1e9, 2), "active_gb": round(mx.get_active_memory() / 1e9, 1)}
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
                   "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "active_gb": round(mx.get_active_memory() / 1e9, 2)}
            rows.append(row)
            print(f"read {len(ids)} tokens at {row['read_tok_s']:.0f} tok/s, {row['read_gb']} GB from the disk in "
                  f"{row['read_share']:.0%} of the time; arrays at the "
                  f"peak {row['peak_gb']} GB, after {row['active_gb']} GB; first token of the answer {first}", flush=True)
            time.sleep(4)
    print(f"held {store.held / 1e9:.2f} GB of experts; all arrays {mx.get_active_memory() / 1e9:.1f} GB, "
          f"peak {mx.get_peak_memory() / 1e9:.1f} GB; {in_memory(args.model):.1f} GB of the file in the page cache", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
