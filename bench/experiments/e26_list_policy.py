"""Experiment 26: ways of keeping the list of rows a guess is worked out over, replayed on e25's recordings.

The list: the tokens of the conversation, and of every row of logits the trunk works out the tokens within
`near` of the largest (they are there for nothing). While the list is shorter than `full_until` rows a guess is
worked out over all rows as before, and then the head's own rows feed the list the same way. The list is kept
from one conversation to the next, or not.

For each way: how many of the guesses the trunk kept would still have been found, how many guesses were worked
out over all rows, the list's length at the end, and what it would do to the speed: a guess over all rows costs
`--all-ms`, over the list `--list-ms` and its share of `--all-ms` by its rows; the head's own pass `--head-ms`;
the rest of a step `--step-ms`; `--guesses` of a draft's recorded guesses are counted (the recording made
five). A guess that is no longer found loses the tokens kept after it in
its pass, which are taken to cost what tokens cost on average.

Result: the ways differ little. The 256 largest of every row the trunk kept, all rows until the list has 1,000:
98.0% of the kept guesses still found, 2% of the guesses over all rows, 13,000 rows; kept between conversations
98.4% and 30,000. Only the last row of a pass: 96%. The 64 largest: 97%. Rows within some distance of the largest
instead of a fixed number: worse for the same length up to 20, and (e27) a row with no clear winner then lists
tens of thousands. Expected speed with three guesses a pass: 6% with the Q2_0 file's costs, 8% with IQ3_S's,
13% with five guesses.

    uv run python bench/experiments/e26_list_policy.py recordings.pkl --all-ms 0.96 --step-ms 31
"""

import argparse
import pickle
from itertools import product
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("recordings")
p.add_argument("--all-ms", type=float, default=0.96)
p.add_argument("--list-ms", type=float, default=0.05)
p.add_argument("--guesses", type=int, default=3)
p.add_argument("--rows", choices=["all", "last"], default="all", help="of a pass's rows, list from all or from the last")
p.add_argument("--top", type=int, default=256, help="at most this many of a row")
p.add_argument("--head-ms", type=float, default=1.1)
p.add_argument("--step-ms", type=float, default=31.0)
args = p.parse_args()
runs = pickle.loads(Path(args.recordings).read_bytes())


def near_rows(ids, values, near, trunk=False):
    out = set()
    ids, values = np.atleast_2d(ids)[:, : args.top], np.atleast_2d(values)[:, : args.top]
    if trunk and args.rows == "last":
        ids, values = ids[-1:], values[-1:]
    for row_ids, row_values in zip(ids, values):
        out.update(row_ids[row_values >= row_values[0] - near].tolist())
    return out


def replay(near, full_until, kept_between, head_rows=True):
    listed = set()
    base_ms = new_ms = 0.0
    tokens = lost = found = was_kept = all_rows = made = 0
    for name, (guesses, lists) in runs.items():
        if not kept_between:
            listed = set()
        by_step = {}
        for g in guesses:
            by_step.setdefault(g[4], []).append(g)
        for i, (toks, ids, values) in enumerate(lists):
            listed.update(toks)
            listed |= near_rows(ids, values, near, trunk=True)
            draft = [g for g in by_step.get(i + 1, []) if g[0] < args.guesses]
            if not draft:
                continue
            kept = sum(g[3] == "kept" for g in draft)
            tokens += kept + 1
            base_ms += args.step_ms + len(draft) * (args.head_ms + args.all_ms)
            new_ms += args.step_ms
            still = True  # the guesses so far in this draft are the ones the trunk will see
            for j, x, t, what, _, h_ids, h_values in sorted(draft, key=lambda g: g[0]):
                made += 1
                if len(listed) < full_until:
                    all_rows += 1
                    new_ms += args.head_ms + args.all_ms
                    if head_rows:
                        listed |= near_rows(h_ids, h_values, near)
                    hit = True
                else:
                    new_ms += args.head_ms + args.list_ms + args.all_ms * len(listed) / 248320
                    hit = t in listed
                listed.add(t)
                if what == "kept":
                    was_kept += 1
                    if still and hit:
                        found += 1
                    elif still:
                        still = False
                        lost += kept - j
    per_token = new_ms / max(tokens - lost, 1)
    return {"found": found / was_kept, "all_rows": all_rows / made, "rows": len(listed),
            "speed": (tokens / (new_ms + lost * per_token)) / (tokens / base_ms)}


print(f"{len(runs)} answers, the first {args.guesses} guesses of a draft; a guess over all rows {args.all_ms} ms, over the list "
      f"{args.list_ms} ms and its share of that; the rest of a step {args.step_ms} ms")
print("near   all rows until   kept between   kept guesses still found   guesses over all rows   rows at the end   speed")
for near, full_until, between in product((1e9,), (0, 1000), (False, True)):
    r = replay(near, full_until, between)
    print(f"{near if near < 1e9 else 'all 256':>7}  {full_until:8}        {'yes' if between else 'no ':3}            {r['found']:6.1%}"
          f"                   {r['all_rows']:6.1%}             {r['rows']:6}        {r['speed']:5.3f}")
