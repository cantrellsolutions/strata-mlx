"""Experiment 25: which rows of the output layer does a guess of the draft head need?

A guess is the largest of 248,320 logits, and working them all out is 1.0 ms a guess with the Q2_0 file and 1.4
with IQ3_S, of a 37 to 43 ms step with three guesses (e24). A guess that comes out wrong costs speed, never the
answer, so the rows could be cut down, if the row that wins is seldom cut. Recorded here in real answers, for
every guess: the vector the head hands the output layer, the token that won, and whether the trunk kept it.
Then, with nothing running, how often the winner would have been among the rows of

  a list    the tokens of the conversation so far, with the K largest of every row of logits the trunk has
            worked out so far (those are there for nothing), and with what other conversations left behind
            (here: the other three prompts' lists, standing in for a list kept between runs);
  a sketch  the output layer times the activations, both squeezed to the r directions the layer's rows vary
            most in (r numbers a row instead of 2,560): the rows whose sketch comes out among the M largest.

Result (Q2_0, 1,190 guesses of four answers, five guesses a pass; of the guesses the trunk kept):

    the conversation's tokens alone                       72%      130 rows at the end
    with the 16 / 64 / 256 largest of every trunk row     92% / 96% / 98%    1,200 / 3,900 / 12,100 rows
    and what the other three answers listed               97% / 99% / 99%    3,600 / 10,900 / 30,100 rows

The sketch does not pay: the layer's rows vary in all directions alike (its 256 strongest hold a third of it),
and for 99% of the winners it takes 512 numbers a row and the 4,096 rows it puts first, or 128 and 16,384:
a fifth of the layer either way, where the list is a twentieth.

    uv run python bench/experiments/e25_guess_rows.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from spec_decode import PROMPTS, STOP, tokenize  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402

TOP = 256


def record(model, mtp, prompt, drafts, tokens):
    """One answer -> its guesses [(position in the draft, vector, token, kept / refused / not checked, how many
    lists were there when it was made)] and the lists [(tokens added, top rows added)] in the order they came."""
    engine = Engine(model, mtp, drafts, as_sure=False)
    engine.warm()
    head = model.lm_head
    seen, lists, guesses, pending = [], [], [], []

    def largest(logits):
        rows = np.array(logits.astype(mx.float32)).reshape(-1, logits.shape[-1])
        top = np.argpartition(-rows, TOP, axis=1)[:, :TOP]
        order = np.take_along_axis(rows, top, axis=1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)  # each row's largest first
        return top, np.take_along_axis(rows, top, axis=1)

    def tap(x):
        out = head(x)
        seen.append((np.array(x.reshape(-1).astype(mx.float32)), *(a[0] for a in largest(out))))
        return out

    real_draft, real_choose = engine._draft, engine._choose

    def draft(residual, following):
        seen.clear()
        model.lm_head = tap
        out = real_draft(residual, following)
        model.lm_head = head
        pending[:] = [(j, x, t, ids, values) for j, ((x, ids, values), t) in enumerate(zip(list(seen), out))]
        return out

    def choose(logits, window, history=None):
        choose.top = largest(logits)
        return real_choose(logits, window, history)

    engine._draft, engine._choose = draft, choose
    out = [engine.read(prompt)]
    lists.append((list(prompt) + out, *choose.top))
    while len(out) < tokens and out[-1] not in STOP:
        made = list(pending)
        settled = engine.step(tokens - len(out), STOP)
        kept = len(settled) - 1
        for j, x, t, ids, values in made:
            guesses.append((j, x, t, "kept" if j < kept else "refused" if j == kept else "not checked", len(lists),
                            ids, values))
        out += settled
        lists.append((settled, choose.top[0][: len(settled)], choose.top[1][: len(settled)]))
    model.lm_head = head
    return guesses, lists


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--drafts", type=int, default=5)
    p.add_argument("--save", help="keep what was recorded (a pickle), for e26 to try ways of listing on")
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model)
    mtp = load_mtp(args.mtp, model.args.text, mx.float16)
    runs = {}
    for name, text in PROMPTS.items():
        prompt = tokenize(args.model, f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        runs[name] = record(model, mtp, prompt, args.drafts, args.tokens)
        g = runs[name][0]
        print(f"{name}: {len(g)} guesses, {sum(k[3] == 'kept' for k in g)} kept, {sum(k[3] == 'refused' for k in g)} refused",
              flush=True)
    if args.save:
        import pickle

        Path(args.save).write_bytes(pickle.dumps(runs))

    # ---- a list
    def members(lists, upto, K):
        s = set()
        for toks, top, _ in lists[:upto]:
            s.update(toks)
            if K:
                s.update(top[:, :K].reshape(-1).tolist())
        return s

    print("\nthe winner is on the list (kept guesses / all checked ones; and the list's size at the end):")
    out = {"list": [], "sketch": []}
    for K in (0, 4, 16, 64, 256):
        for others in (False, True):
            hit = {"kept": [0, 0], "refused": [0, 0]}
            size = []
            for name, (guesses, lists) in runs.items():
                prior = set()
                if others:
                    for other, (_, ol) in runs.items():
                        if other != name:
                            prior |= members(ol, len(ol), K)
                cache = {}
                for j, x, t, what, upto, *_ in guesses:
                    if what == "not checked":
                        continue
                    if upto not in cache:
                        cache[upto] = members(lists, upto, K)
                    # the guesses before this one in the same draft are tokens too
                    hit[what][0] += t in cache[upto] or t in prior
                    hit[what][1] += 1
                size.append(len(members(lists, len(lists), K) | prior))
            kept, ref = hit["kept"], hit["refused"]
            row = {"K": K, "with_other_conversations": others, "kept": kept[0] / kept[1],
                   "checked": (kept[0] + ref[0]) / (kept[1] + ref[1]), "rows": int(np.mean(size))}
            out["list"].append(row)
            print(f"  the conversation{' and the other three' if others else '':21} with the {K:3} largest of every row: "
                  f"{row['kept']:6.1%} / {row['checked']:6.1%}   {row['rows']:6} rows", flush=True)

    # ---- a sketch
    lm = model.lm_head
    W = lm.floats(mx.float32) if hasattr(lm, "floats") else mx.dequantize(
        lm.weight, lm.scales, lm.biases, lm.group_size, lm.bits).astype(mx.float32)
    mx.eval(W)
    t0 = time.perf_counter()
    gram = np.array(W.T @ W).astype(np.float64)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    print(f"\nthe layer's directions worked out in {time.perf_counter() - t0:.1f} s; of what its rows vary by, the first "
          + ", ".join(f"{r}: {values[:r].sum() / values.sum():.0%}" for r in (64, 128, 256, 512, 1024)), flush=True)
    checked = [(x, t, what) for g, _ in runs.values() for _, x, t, what, *_ in g if what != "not checked"]
    X = mx.array(np.stack([x for x, _, _ in checked]))
    tok = mx.array([t for _, t, _ in checked])
    is_kept = np.array([what == "kept" for *_, what in checked])
    exact_top = []
    for i in range(0, len(checked), 128):
        exact_top.append(mx.argmax(X[i : i + 128] @ W.T, axis=1))
    exact_top = mx.concatenate(exact_top)
    agree = (exact_top == tok).astype(mx.float32).mean().item()
    print(f"the recorded winner is the largest of the float32 logits for {agree:.1%} of the guesses")
    print("the winner is among the M rows the sketch puts first (kept guesses / all checked ones):")
    for r in (64, 128, 256, 512):
        P = mx.array(vectors[:, :r].astype(np.float32))
        Wr = W @ P
        mx.eval(Wr)
        ranks = []
        for i in range(0, len(checked), 128):
            A = (X[i : i + 128] @ P) @ Wr.T
            mine = mx.take_along_axis(A, tok[i : i + 128, None], axis=1)
            ranks.append((A > mine).sum(axis=1))
        ranks = np.array(mx.concatenate(ranks))
        line = f"  {r:4} numbers a row:"
        for M in (1, 16, 256, 1024, 4096, 16384):
            row = {"r": r, "M": M, "kept": float((ranks[is_kept] < M).mean()), "checked": float((ranks < M).mean())}
            out["sketch"].append(row)
            line += f"  M {M:5}: {row['kept']:6.1%} / {row['checked']:6.1%}"
        print(line, flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
