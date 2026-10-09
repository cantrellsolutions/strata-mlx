"""Answer speed with only part of the experts in memory, the others read from the SSD (strata_mlx/paged.py).

For each memory budget: a few conversations one after the other in one process (the store starts empty and is
never emptied, as in a server), with and without the draft head. The OS is asked to forget the file before the
first conversation, and the last line says how much of it is back in memory: it must stay near nothing, or the
page cache of this machine has been answering for the disk of a smaller one.
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
    p.add_argument("--budget", type=float, required=True, help="GB of experts in memory")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--prompts", nargs="+", default=["prose", "code", "reasoning", "edit"])
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model, expert_memory=args.budget)
    store = model.expert_store
    mtp = load_mtp(args.mtp, model.args.text)
    engines = {"one token a pass": Engine(model), "draft head": Engine(model, mtp, 3, lookup=True)}
    for e in engines.values():
        e.warm()
    in_memory(args.model, drop=True)
    print(f"budget {args.budget} GB of experts; everything else {mx.get_active_memory() / 1e9 - store.held / 1e9:.1f} GB", flush=True)
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
                row = {"budget_gb": args.budget, "round": round_, "prompt": name, "engine": label,
                       "prompt_tokens": len(prompt), "prompt_tok_s": round(len(prompt) / (t1 - t0), 1),
                       "prompt_read_gb": round(read_prompt / 1e9, 2), "tok_s": round(n / dt, 1),
                       "misses_per_token": round(store.misses / n, 1), "read_mb_per_token": round(store.bytes_read / n / 1e6, 1),
                       "read_share": round(store.read_s / dt, 2), "tokens_per_pass": round(e.stats.tokens / max(e.stats.steps, 1), 2),
                       "held_gb": round(store.held / 1e9, 2), "active_gb": round(mx.get_active_memory() / 1e9, 1)}
                rows.append(row)
                print(f"{round_:5} {name:9} {label:16} {row['tok_s']:5.1f} tok/s  {row['misses_per_token']:6.1f} misses/token  "
                      f"{row['read_mb_per_token']:6.1f} MB/token  {row['read_share']:4.0%} of the time reading  "
                      f"{row['tokens_per_pass']:4.2f} tokens/pass | prompt {row['prompt_tok_s']:5.1f} tok/s", flush=True)
    print(f"held {store.held / 1e9:.2f} GB of experts; all arrays {mx.get_active_memory() / 1e9:.1f} GB, "
          f"peak {mx.get_peak_memory() / 1e9:.1f} GB; {in_memory(args.model):.1f} GB of the file in the page cache", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
