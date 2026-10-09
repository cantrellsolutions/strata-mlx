"""Experiment 28: guesses over the listed rows (strata_mlx/rows.py) against guesses over all rows, in turns.

Two engines of the same kind in one process, one with the list and one without, answer each prompt one after
the other, `--rounds` times. The list is kept from prompt to prompt, as a server keeps it.

    uv run python bench/experiments/e28_guess_rows.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp [--fixed 3]
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from spec_decode import PROMPTS, decode, tokenize  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--mtp", required=True)
p.add_argument("--tokens", type=int, default=256)
p.add_argument("--fixed", type=int, default=0, help="this many guesses every pass; 0: as many as the head is sure of")
p.add_argument("--rounds", type=int, default=2)
p.add_argument("--json")
args = p.parse_args()
model, _ = load(args.model)
mtp = load_mtp(args.mtp, model.args.text, mx.float16)
kind = dict(drafts=args.fixed, as_sure=False) if args.fixed else {}
plain, listed = Engine(model, mtp, guess_rows=False, **kind), Engine(model, mtp, **kind)
plain.warm()
listed.warm()
rows = []
for name, text in PROMPTS.items():
    prompt = tokenize(args.model, f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
    for _ in range(args.rounds):
        a, _, a_tps = decode(plain, prompt, args.tokens, {})
        a_pass, a_kept = plain.stats.tokens / plain.stats.steps, plain.stats.accepted / max(plain.stats.drafted, 1)
        b, _, b_tps = decode(listed, prompt, args.tokens, {})
        b_pass, b_kept = listed.stats.tokens / listed.stats.steps, listed.stats.accepted / max(listed.stats.drafted, 1)
        same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        rows.append({"prompt": name, "all_rows": round(a_tps, 1), "listed": round(b_tps, 1), "tokens_per_pass": [round(a_pass, 2), round(b_pass, 2)],
                     "guesses_kept": [round(a_kept, 3), round(b_kept, 3)], "rows": listed.rows.count, "same_until": same, "tokens": len(a)})
        print(f"{name:10} all rows {a_tps:5.1f} tok/s ({a_pass:.2f} tokens/pass, {a_kept:.0%} kept)   listed {b_tps:5.1f} tok/s "
              f"({b_pass:.2f}, {b_kept:.0%})  {b_tps / a_tps - 1:+6.1%}   {listed.rows.count:6} rows"
              + ("" if same == len(a) == len(b) else f" | answers part at token {same}"), flush=True)
if args.json:
    Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
