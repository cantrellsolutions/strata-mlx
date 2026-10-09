"""Experiment 13: how many guesses should a pass try, given how sure the draft head is of each?

The engine tries three guesses every pass. A guess costs a call of the head and a longer pass whether or not
it is kept. This records, for the four benchmark prompts with seven guesses a pass, the probability the head
gave each guess and how many of them the trunk kept; then, from the measured time of a pass of each size, what
rule would have written the most tokens a second:

    fixed k        always k guesses
    while sure     guess on while the head's probabilities so far multiply to at least a threshold

    uv run python bench/experiments/e13_draft_confidence.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from spec_decode import PROMPTS, STOP, tokenize  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402
from strata_mlx.sampler import Sampling, pick, uniform  # noqa: E402


class Recording(Engine):
    """An engine whose head also says how probable it found each of its guesses."""

    def _draft(self, residual, following):
        m, cache, s = self.m, self.mtp_cache, self.sampling
        first = len(self.tokens) + 1
        guesses, probs = [], []

        def guess(x, j):
            logits = self.model.lm_head(x)[0].astype(mx.float32)
            g = pick(logits, mx.array([uniform(s.seed, first + j)]), s)[None]
            p = mx.softmax(logits, axis=-1)
            probs.append(mx.take_along_axis(p, g.reshape(-1, 1), axis=-1).reshape(-1))
            return g

        x, r = self._cells(residual, following)
        g = guess(x[:, -1:], 0)
        guesses, r = [g], r[:, -1:]
        for j in range(1, self.drafts):
            x, r = self.mtp(r, m.embed_tokens(g), m.rope, cache)
            g = guess(x, j)
            guesses.append(g)
        out = mx.concatenate(guesses, axis=1)
        pr = mx.concatenate(probs)
        mx.eval(out, pr)
        cache.trim(self.drafts - 1)
        mx.eval(cache.keys, cache.values, cache.indexer.keys)
        self.last_probs = pr.tolist()
        return out[0].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model)
    mtp = load_mtp(args.mtp, model.args.text)
    engine = Recording(model, mtp, drafts=7)
    engine.warm()
    sampling = dict(temperature=args.temperature, top_p=0.95, top_k=20, seed=1) if args.temperature else {}

    # the time of a pass by its size and of one guess of the head, in this run
    engine.read(list(range(1000, 1200)))
    pass_ms = {}
    for T in list(range(1, 9)) * 12:
        ids = mx.array([[1000 + (T * 7 + j) % 500 for j in range(T)]])
        t0 = time.perf_counter()
        logits, w = engine._forward(ids)
        mx.eval(logits)
        engine._commit(w, T)
        pass_ms.setdefault(T, []).append((time.perf_counter() - t0) * 1e3)
    pass_ms = {T: sorted(v)[len(v) // 4] for T, v in pass_ms.items()}

    steps = []  # (probabilities of the 7 guesses, how many were kept)
    draft_ms = []
    for name in PROMPTS:
        prompt = tokenize(args.model, f"<|im_start|>user\n{PROMPTS[name]}<|im_end|>\n<|im_start|>assistant\n")
        engine.reset()
        engine.sampling = Sampling(**sampling)
        out = [engine.read(prompt)]
        probs = engine.last_probs
        while len(out) < args.tokens and out[-1] not in STOP:
            before = engine.stats.accepted
            n_guesses = len(engine.guesses)
            t0 = time.perf_counter()
            got = engine.step(args.tokens - len(out), STOP)
            if n_guesses == 7 and len(got) <= 8:
                steps.append({"prompt": name, "probs": probs, "kept": engine.stats.accepted - before})
            probs = engine.last_probs
            out += got
    # one guess of the head: the time of drafting seven, over seven
    t = []
    for _ in range(20):
        r = mx.zeros((1, 1, model.args.text.hidden_size * model.args.text.hc_count), dtype=mx.float16)
        t0 = time.perf_counter()
        engine._draft(r, mx.array([[1000]]))
        t.append((time.perf_counter() - t0) * 1e3)
        engine.mtp_cache.trim(1)
    guess_ms = sorted(t)[len(t) // 4] / 7
    fixed_ms = 3.0  # building the graph, choosing, committing

    def step_ms(k):  # k guesses tried
        return fixed_ms + pass_ms[k + 1] + guess_ms * k

    print("a pass by its size, ms: " + " ".join(f"{T}:{ms:.1f}" for T, ms in pass_ms.items()) + f"; a guess {guess_ms:.2f} ms")
    by_depth = [sum(s["kept"] > j for s in steps) / len(steps) for j in range(7)]
    print(f"{len(steps)} passes; a guess is kept, by its place: " + " ".join(f"{x:.2f}" for x in by_depth))
    rows = []
    for k in range(0, 8):
        tokens = sum(1 + min(s["kept"], k) for s in steps) / len(steps)
        rows.append({"rule": f"fixed {k}", "tokens": tokens, "ms": step_ms(k)})
    for threshold in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
        tokens = ms = 0.0
        for s in steps:
            k, sure = 0, 1.0
            # the head has to make a guess to know how sure it is of it: a guess below the threshold is made
            # and paid for, then left out of the window
            made = 0
            while made < 7:
                made += 1
                sure *= s["probs"][made - 1]
                if sure < threshold:
                    break
                k = made
            tokens += 1 + min(s["kept"], k)
            ms += fixed_ms + pass_ms[k + 1] + guess_ms * made
        rows.append({"rule": f"while sure >= {threshold}", "tokens": tokens / len(steps), "ms": ms / len(steps)})
    for r in rows:
        r["tok_s"] = r["tokens"] / r["ms"] * 1e3
        print(f"{r['rule']:22} {r['tokens']:.2f} tokens a pass, {r['ms']:5.1f} ms a pass, {r['tok_s']:5.1f} tok/s")
    if args.json:
        Path(args.json).write_text(json.dumps({"pass_ms": pass_ms, "guess_ms": guess_ms, "rows": rows, "steps": steps}, indent=1) + "\n")


if __name__ == "__main__":
    main()
