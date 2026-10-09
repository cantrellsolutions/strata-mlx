"""Experiment 18: what would holding the hyper-connections at 8 bits do to the answers?

Their weights are 16-bit floats in the file, 1.27 GB read by every token, a quarter of what a pass reads. At 8
bits a pass would read 0.6 GB less (about 1.6 ms of 20.8 for one token, of 30.3 for four). That changes
weights, which nothing in this engine does so far; this only measures what it would change: the projections
are rounded to 8 (6, 4) bits and put back as floats, so the kernels and their speed stay as they are, and the
engine is compared with llama.cpp's saved answers as `compare_llama.py` does.

    uv run python bench/experiments/e18_hc_bits.py --model <shard 1> --ref <saved answers> [--ref ...]
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from compare_llama import END_TOKENS  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--ref", action="append", required=True)
args = p.parse_args()
model, _ = load(args.model)
m = model.model
grs = [gr for l in m.layers for gr in (l.attn_hyper_connection, l.mlp_hyper_connection)] + [m.hyper_connection_mixer]
original = [(gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight) for gr in grs]


def set_bits(bits):
    worst = 0.0
    for gr, (down, up) in zip(grs, original):
        for lin, w in ((gr.input_mix_weight_down, down), (gr.input_mix_weight_up, up)):
            if bits is None:
                lin.weight = w
                continue
            q, scales, biases = mx.quantize(w.astype(mx.float32), group_size=64, bits=bits)
            back = mx.dequantize(q, scales, biases, group_size=64, bits=bits).astype(w.dtype)
            worst = max(worst, (mx.abs(back.astype(mx.float32) - w.astype(mx.float32)).max() / mx.abs(w).max()).item())
            lin.weight = back
            mx.eval(back)
    return worst


def logp_rows(prompt, gen):
    engine = Engine(model)  # built after the weights are set: its kernels hold the arrays they were made with
    engine.read(prompt[:-1])
    rows = []
    for t in prompt[-1:] + gen[:-1]:
        logits, w = engine._forward(mx.array([[t]]))
        mx.eval(logits)
        engine._commit(w, 1)
        rows.append(logits[0])
    logp = mx.concatenate(rows).astype(mx.float32)
    logp[:, list(END_TOKENS)] = -mx.inf
    return logp - mx.logsumexp(logp, axis=-1, keepdims=True)


refs = [json.loads(Path(r).read_text()) for r in args.ref]
base = {}
for bits in (None, 8, 6, 4):
    worst = set_bits(bits)
    name = "16-bit floats, as in the file" if bits is None else f"{bits} bits (largest change of a weight {worst:.1e} of the largest weight)"
    for path, saved in zip(args.ref, refs):
        prompt, ref = saved["prompt"], saved["ref"]
        logp = logp_rows(prompt, [s["id"] for s in ref])
        top = mx.argmax(logp, axis=-1).tolist()
        same = sum(t == s["id"] for t, s in zip(top, ref))
        gaps = sorted(abs(logp[i, s["id"]].item() - s["top_logprobs"][0]["logprob"]) for i, s in enumerate(ref))
        line = (f"{name}: {Path(path).stem}: llama.cpp's token at {same}/{len(ref)} steps; its log-probability "
                f"off by {gaps[len(gaps) // 2]:.3f} (median), {gaps[-1]:.3f} (most)")
        if bits is None:
            base[path] = (top, logp)
        else:
            b_top, b_logp = base[path]
            moved = mx.abs(logp - b_logp)[mx.arange(len(top)), mx.array(b_top)]
            line += f"; against 16 bits: the same token at {sum(a == b for a, b in zip(top, b_top))}/{len(top)}, log-probability moved {moved.max().item():.3f} at most"
        print(line, flush=True)
set_bits(None)
