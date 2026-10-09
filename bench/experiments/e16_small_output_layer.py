"""Experiment 16: could the draft head guess through a smaller copy of the output layer?

A guess of the head is one call of its layer and one of the output layer: 477 MB read for one token id, three
to seven times a pass. A copy at fewer bits is only for guessing; the trunk decides every token through the
real one. This collects what the head hands to the output layer over the four benchmark prompts, then for
copies at 4, 3 and 2 bits: how often the copy's guess is the real layer's, and what a call costs.

    uv run python bench/experiments/e16_small_output_layer.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from spec_decode import PROMPTS, STOP, tokenize  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--mtp", required=True)
p.add_argument("--tokens", type=int, default=200)
args = p.parse_args()
model, _ = load(args.model)
mtp = load_mtp(args.mtp, model.args.text)
head = model.lm_head
seen = []


class Spy:
    def __call__(self, x):
        if x.shape[1] == 1:
            seen.append(x)
        return head(x)


engine = Engine(model, mtp)
engine.warm()
model.lm_head = Spy()
for name in PROMPTS:
    prompt = tokenize(args.model, f"<|im_start|>user\n{PROMPTS[name]}<|im_end|>\n<|im_start|>assistant\n")
    engine.reset()
    out = [engine.read(prompt)]
    while len(out) < args.tokens and out[-1] not in STOP:
        out += engine.step(args.tokens - len(out), STOP)
model.lm_head = head
xs = mx.concatenate(seen, axis=0)  # the trunk's own single-token passes are among them: all are states of the model
mx.eval(xs)
print(f"{xs.shape[0]} states")
real = mx.argmax(head(xs), axis=-1).reshape(-1)
logp = nn.log_softmax(head(xs).astype(mx.float32), axis=-1)
sure = mx.exp(logp.max(axis=-1)).reshape(-1)
mx.eval(real, sure)

w = mx.dequantize(head.weight, head.scales, head.biases, group_size=head.group_size, bits=head.bits)
mx.eval(w)


def timed(layer):
    x = xs[:1]
    best = float("inf")
    for _ in range(30):
        t0 = time.perf_counter()
        mx.eval(mx.argmax(layer(x), axis=-1))
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


size = lambda l: sum(a.nbytes for a in (l.weight, l.scales, l.biases)) / 1e6
print(f"the real one, {head.bits} bits: {size(head):.0f} MB, {timed(head):.2f} ms a call")
for bits, group in ((4, 64), (3, 64), (2, 64), (2, 32)):
    q = nn.QuantizedLinear(group, 1, bias=False, group_size=group, bits=bits)
    q.weight, q.scales, q.biases = mx.quantize(w, group_size=group, bits=bits)
    q.set_dtype(mx.float16)
    mx.eval(q.parameters())
    got = mx.argmax(q(xs), axis=-1).reshape(-1)
    same = (got == real)
    mx.eval(same)
    hi = sure > 0.5
    print(f"{bits} bits, groups of {group}: {size(q):.0f} MB, {timed(q):.2f} ms a call; the same guess in {same.astype(mx.float32).mean().item():.1%} "
          f"of all, {(same & hi).sum().item() / max(hi.sum().item(), 1):.1%} of those the real one gives more than half", flush=True)
    del q
