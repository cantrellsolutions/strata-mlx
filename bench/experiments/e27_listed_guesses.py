"""Experiment 27: the draft head's guesses worked out over a list of rows, in the engine.

e25 and e26 replayed recordings; here the thing itself runs. The list: the prompt's tokens and, of every row of
logits the trunk works out, the 256 largest (or those within `--near` of the largest). Rows new to the list
have their weights taken out of the output layer once, at the step that lists them; a guess multiplies the
pieces taken so far and puts the logits where they belong among 248,320 that are otherwise minus infinity, so
that everything after it is as before. Until the list has `--full-until` rows a guess goes over all rows.

Two engines with three guesses a pass, one as it is and one with the list, answer each prompt in turns.

Result (Q2_0): the tokens a pass settles stay what they were (3.07-3.49); 2 to 7% more tokens a second, with
1.1 ms of every step spent keeping the list: picking the 256 largest of four rows is 0.5 ms of GPU
(`argpartition` sorts), the rest is waiting for it and for the rows taken out. With `--near 20` the list grew
to 38,000-225,000 rows and the answers got slower. Hence rows.py: only the rows the trunk kept, picked out
without being waited for, and listed a step later.

    uv run python bench/experiments/e27_listed_guesses.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp
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

from strata_mlx.dense import ScaledLinear  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402
from strata_mlx.sampler import Sampling  # noqa: E402


def rows_of(head, ids: np.ndarray):
    """The output layer's rows `ids` as a layer of their own: x (1, in) -> (1, len(ids))."""
    n = len(ids)
    if isinstance(head, ScaledLinear):
        padded = mx.array(np.concatenate([ids, np.repeat(ids[-1:], -n % 4)]))
        part = ScaledLinear(*(mx.take(a, padded, axis=0) for a in (head.codes, head.scales, head.d)))
        mx.eval(part.codes, part.scales, part.d)
        return lambda x: part(x)[:, :n]
    at = mx.array(ids)
    if hasattr(head, "scales"):
        w, s, b = (mx.take(a, at, axis=0) for a in (head.weight, head.scales, head.biases))
        mx.eval(w, s, b)
        return lambda x: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=head.group_size, bits=head.bits)
    w = mx.take(head.weight, at, axis=0)
    mx.eval(w)
    return lambda x: x @ w.T


class Listed:
    def __init__(self, head, vocab: int, pieces: int = 4):
        self.head, self.vocab, self.most = head, vocab, pieces
        self.reset()

    def reset(self):
        self.have = np.zeros(self.vocab, dtype=bool)
        self.pieces = []  # (ids, their rows as a layer)
        self.count = 0

    def add(self, ids):
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        new = ids[~self.have[ids]]
        if not len(new):
            return
        self.have[new] = True
        self.count += len(new)
        self.pieces.append((new, rows_of(self.head, new)))
        if len(self.pieces) > self.most:
            everything = np.concatenate([i for i, _ in self.pieces])
            self.pieces = [(everything, rows_of(self.head, everything))]

    def __call__(self, x: mx.array) -> mx.array:
        """Logits (..., vocab): of the listed rows, and minus infinity elsewhere."""
        flat = x.reshape(1, -1)
        logits = mx.concatenate([layer(flat) for _, layer in self.pieces], axis=1)[0]
        out = mx.full((self.vocab,), -mx.inf, dtype=logits.dtype)
        out[mx.array(np.concatenate([i for i, _ in self.pieces]))] = logits
        return out.reshape(*x.shape[:-1], self.vocab)


def with_list(engine, model, args):
    """Wrap an engine's choosing and drafting so that its guesses go over the list."""
    head = model.lm_head
    listed = Listed(head, engine.vocab)
    stats = {"maintain_ms": 0.0, "steps": 0, "over_all": 0, "drafts": 0}
    real_choose, real_draft = engine._choose, engine._draft

    def choose(logits, window, history=None):
        out = real_choose(logits, window, history)
        t0 = time.perf_counter()
        if args.near:
            near = (logits >= logits.max(axis=-1, keepdims=True) - args.near).any(axis=0)
            ids = np.flatnonzero(np.array(near))
        else:
            ids = np.array(mx.argpartition(-logits, args.top, axis=-1)[:, : args.top]).reshape(-1)
        listed.add(ids)
        stats["maintain_ms"] += (time.perf_counter() - t0) * 1e3
        stats["steps"] += 1
        return out

    def draft(residual, following):
        stats["drafts"] += 1
        if listed.count < args.full_until:
            stats["over_all"] += 1
            return real_draft(residual, following)
        model.lm_head = listed
        try:
            return real_draft(residual, following)
        finally:
            model.lm_head = head

    engine._choose, engine._draft = choose, draft
    return listed, stats


def decode(engine, prompt, tokens, listed=None):
    engine.reset()
    engine.sampling = Sampling()
    if listed is not None:
        listed.reset()
        listed.add(prompt)
    out = [engine.read(prompt)]
    t0 = time.perf_counter()
    while len(out) < tokens and out[-1] not in STOP:
        out += engine.step(tokens - len(out), STOP)
    return out, (len(out) - 1) / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--drafts", type=int, default=3)
    p.add_argument("--as-sure", action="store_true")
    p.add_argument("--top", type=int, default=256)
    p.add_argument("--near", type=float, default=0.0, help="list the rows within this of the largest instead of the top ones")
    p.add_argument("--full-until", type=int, default=1000)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model)
    mtp = load_mtp(args.mtp, model.args.text, mx.float16)
    kind = dict(as_sure=True) if args.as_sure else dict(drafts=args.drafts, as_sure=False)
    plain, ours = Engine(model, mtp, **kind), Engine(model, mtp, **kind)
    plain.warm()
    ours.warm()
    listed, stats = with_list(ours, model, args)
    rows = []
    for name, text in PROMPTS.items():
        prompt = tokenize(args.model, f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        for _ in range(args.rounds):
            a, a_tps = decode(plain, prompt, args.tokens)
            a_pass = plain.stats.tokens / plain.stats.steps
            stats.update(maintain_ms=0.0, steps=0, over_all=0, drafts=0)
            b, b_tps = decode(ours, prompt, args.tokens, listed)
            b_pass = ours.stats.tokens / ours.stats.steps
            same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            rows.append({"prompt": name, "as_it_is": round(a_tps, 1), "listed": round(b_tps, 1), "tokens_per_pass": [round(a_pass, 2), round(b_pass, 2)],
                         "rows": listed.count, "maintain_ms": round(stats["maintain_ms"] / max(stats["steps"], 1), 3),
                         "drafts_over_all": stats["over_all"], "same_until": same, "tokens": len(a)})
            print(f"{name:10} as it is {a_tps:5.1f} tok/s ({a_pass:.2f} tokens/pass)   listed {b_tps:5.1f} tok/s ({b_pass:.2f})  {b_tps / a_tps - 1:+6.1%}"
                  f"   {listed.count:6} rows, {stats['maintain_ms'] / max(stats['steps'], 1):.2f} ms a step keeping the list, "
                  f"{stats['over_all']} of {stats['drafts']} drafts over all rows"
                  + ("" if same == len(a) == len(b) else f" | answers part at token {same}"), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
