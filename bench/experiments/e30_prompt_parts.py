"""Experiment 30: where the time goes when a chunk of a prompt is read.

A window of an answer is thousands of small kernels, and timing its parts alone says little (e06). A chunk of
a prompt is the other case: a thousand tokens through each part, every kernel milliseconds long. So here the
parts are timed in place: each is made to finish (its inputs before the clock starts, its outputs before it
stops), and the time between two parts goes to the part they are inside of. The parts add up to the pass by
construction; the pass without the stops is timed too, to show what the stops cost.

For the parts that are matrix products the operations are counted (2 x rows x columns x tokens), so that each
can be set against what this GPU does in one large product: 14.3 T operations a second (ROADMAP).

Then the two ways the ten chosen experts are multiplied (the 2-bit product in tiles of 16 tokens an expert,
blocks.packed_matmul, and the table of sums a pair at a time, blocks.lookup_matvec), each forced at every
length, and how much of the tiles is filling.

    uv run python bench/experiments/e30_prompt_parts.py --model <shard 1> [--mtp DIR] [--json out.json]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
import numpy as np  # noqa: E402

import strata_mlx.blocks as B  # noqa: E402
import strata_mlx.engine as E  # noqa: E402
from spec_decode import EDIT_ME, tokenize  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--mtp")
p.add_argument("--tokens", type=int, default=1024)
p.add_argument("--rounds", type=int, default=4)
p.add_argument("--lengths", nargs="*", type=int, default=[128, 256, 512, 1024, 2048])
p.add_argument("--json")
args = p.parse_args()

model, _ = load(args.model)
m = model.model
text = "<|im_start|>user\nSay in two sentences what this module is for.\n\n```python\n" + EDIT_ME * 9 + "```<|im_end|>\n<|im_start|>assistant\n"
prompt = tokenize(args.model, text)
assert len(prompt) > max([args.tokens, 1180, *args.lengths]), len(prompt)


def arrays(x, out=None):
    out = [] if out is None else out
    if isinstance(x, mx.array):
        out.append(x)
    elif isinstance(x, (tuple, list)):
        for v in x:
            arrays(v, out)
    return out


class Clock:
    """Time goes to the innermost part that is running; parts stop the GPU at their edges."""

    def __init__(self):
        self.on = False
        self.reset()

    def reset(self):
        self.ms, self.ops, self.stack, self.last = defaultdict(float), defaultdict(float), ["glue"], time.perf_counter()

    def enter(self, name, inputs):
        mx.eval(*arrays(inputs))
        now = time.perf_counter()
        self.ms[self.stack[-1]] += (now - self.last) * 1e3
        self.stack.append(name)
        self.last = now

    def leave(self, outputs):
        mx.eval(*arrays(outputs))
        now = time.perf_counter()
        self.ms[self.stack.pop()] += (now - self.last) * 1e3
        self.last = now


clock = Clock()


def part(owner, name, label, linear=False):
    """Have owner.name charge its time to `label`; a matrix product counts its operations too."""
    inner = getattr(owner, name)

    def timed(*a, **kw):
        if not clock.on:
            return inner(*a, **kw)
        clock.enter(label, a)
        out = inner(*a, **kw)
        clock.leave(out)
        if linear:
            x = a[0]
            clock.ops[label] += 2.0 * x.size * out.shape[-1]
        return out

    setattr(owner, name, timed)


tiles = {"pairs": 0, "slots": 0, "experts": 0, "calls": 0}
_packed_glu = B._packed_glu


def counting(x, indices, up, gate, down):
    counts = np.unique(np.array(indices).reshape(-1), return_counts=True)[1]
    tiles["pairs"] += int(counts.sum())
    tiles["slots"] += int((-(-counts // B.TOKENS_TILE) * B.TOKENS_TILE).sum())
    tiles["experts"] += len(counts)
    tiles["calls"] += 1
    return _packed_glu(x, indices, up, gate, down)


mlps = [layer.mlp for layer in m.layers]
engine = Engine(model)
with_head = None  # made before the parts are wrapped: an engine reaches into the layers when it is set up
if args.mtp:
    from strata_mlx.mtp import load_mtp

    with_head = Engine(model, load_mtp(args.mtp, model.args.text, mx.float16))
    with_head.warm()
for layer in m.layers:
    if layer.ple is not None:
        part(layer, "ple", "n-gram table: rows from the file, their convolution")
    if layer.layer_type == "linear_attention":
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
            part(layer.linear_attn, name, "delta-net: projections", linear=True)
    else:  # the parts inside first: once a part is wrapped it is a function, with nothing to reach into
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            part(layer.self_attn, name, "attention: projections", linear=True)
        part(layer, "self_attn", "attention: picking the cells, the attention itself")
    part(layer.mlp, "gate", "experts: the routers' product", linear=True)
    part(layer.mlp, "switch_mlp", "experts: the ten chosen")
    for name in ("gate_proj", "up_proj", "down_proj"):
        part(layer.mlp.shared_expert, name, "experts: the shared one", linear=True)
    part(layer, "mlp", "experts: choosing ten of 512, weighting and adding them up")
part(E, "gated_delta_update", "delta-net: the recurrence, token after token")
part(engine, "_gdn", "delta-net: convolution, norms")
part(model, "lm_head", "output layer, the last token only")
for key, pair in engine.hc.items():

    def timed(*a, inner=pair[1]):
        if not clock.on:
            return inner(*a)
        clock.enter("hyper-connections", a)
        out = inner(*a)
        clock.leave(out)
        return out

    engine.hc[key] = (pair[0], timed)

d = m.args.hidden_size
ROUTED_OPS = 2.0 * 10 * 3 * 640 * d  # a token, a layer: ten experts of three 640 x 2560 matrices


def read(T, stops):
    engine.reset()
    mx.eval(engine.m.embed_tokens(mx.array([prompt[:4]])))
    clock.on = stops
    clock.reset()
    t0 = time.perf_counter()
    logits, w = engine._forward(mx.array([prompt[:T]]), last_only=True)
    mx.eval(logits)
    if stops:
        now = time.perf_counter()
        clock.ms["glue"] += (now - clock.last) * 1e3
        clock.last = now
    engine._commit(w, T)
    mx.eval(*arrays([list(v) for v in engine.gdn.values()]))
    dt = (time.perf_counter() - t0) * 1e3
    if stops:
        clock.ms["keeping the states"] += (time.perf_counter() - clock.last) * 1e3
    clock.on = False
    return dt


def median(xs):
    return sorted(xs)[len(xs) // 2]


T = args.tokens
read(T, False), read(T, True)  # kernels compiled, floats made
plain, stopped, parts, ops = [], [], defaultdict(list), {}
for _ in range(args.rounds):
    time.sleep(6)
    plain.append(read(T, False))
    time.sleep(6)
    stopped.append(read(T, True))
    for k, v in clock.ms.items():
        parts[k].append(v)
    ops = dict(clock.ops)
ops["experts: the ten chosen"] = ROUTED_OPS * T * len(m.layers)
total = median(stopped)
print(f"\n{T} tokens of a prompt, one piece: {median(plain):.0f} ms ({T / median(plain) * 1e3:.0f} tok/s); "
      f"with a stop after every part {total:.0f} ms\n")
print(f"{'part':62} {'ms':>7} {'share':>6}   {'T operations':>12} {'T ops/s':>8}")
rows = []
for name, v in sorted(parts.items(), key=lambda kv: -median(kv[1])):
    ms = median(v)
    o = ops.get(name)
    rows.append({"part": name, "ms": ms, "share": ms / total, "operations": o})
    print(f"{name:62} {ms:7.0f} {ms / total:6.1%}   " + (f"{o / 1e12:12.2f} {o / 1e12 / (ms / 1e3):8.1f}" if o else ""))
print(f"{'sum':62} {sum(median(v) for v in parts.values()):7.0f}")

# The stops are not free (a fifth more time with the Q2_0 file, twice the time with IQ3_S, whose experts go
# through 1.7 GB of floats a stack). So the largest part is measured a second way: the pass without it.
whole_ms, without_ms = [], []
for _ in range(max(args.rounds, 3)):
    time.sleep(6)
    whole_ms.append(read(T, False))
    kept = [mlp.switch_mlp for mlp in mlps]
    for mlp in mlps:
        mlp.switch_mlp = lambda x, idx: mx.zeros((*idx.shape, d), dtype=x.dtype)
    time.sleep(6)
    without_ms.append(read(T, False))
    for mlp, f in zip(mlps, kept):
        mlp.switch_mlp = f
left_out = {"whole": median(whole_ms), "without": median(without_ms)}
print(f"\nthe pass without the ten chosen (they answer zeros), no stops: {left_out['without']:.0f} ms against "
      f"{left_out['whole']:.0f}: they are {left_out['whole'] - left_out['without']:.0f} ms, "
      f"{1 - left_out['without'] / left_out['whole']:.0%} of the pass, "
      f"{ROUTED_OPS * T * len(m.layers) / 1e12 / ((left_out['whole'] - left_out['without']) / 1e3):.1f} T operations a second")

B._packed_glu = counting
read(T, False)
B._packed_glu = _packed_glu
filling = None
if tiles["calls"]:
    filling = {k: v / tiles["calls"] for k, v in tiles.items() if k != "calls"}
    print(f"\nthe ten chosen, a layer of {T} tokens: {filling['pairs']:.0f} pairs for {filling['experts']:.0f} experts "
          f"({filling['pairs'] / filling['experts']:.1f} each), in tiles of {B.TOKENS_TILE}: {filling['slots']:.0f} slots, "
          f"{1 - filling['pairs'] / filling['slots']:.0%} of them filling")

print("\none piece of a prompt, by its length (no stops), the ten chosen multiplied either way:")
lengths = []
default = B.PACKED_FROM
for n in args.lengths:
    row = {"tokens": n}
    for way, at in (("tiles", 0), ("table", 10**9)):
        B.PACKED_FROM = at
        read(n, False)
        ts = []
        for _ in range(3):
            time.sleep(4)
            ts.append(read(n, False))
        row[way] = median(ts)
    B.PACKED_FROM = default
    lengths.append(row)
    now = "tiles" if n * 10 >= default else "table"
    print(f"  {n:5} tokens   2-bit product in tiles {row['tiles']:6.0f} ms {n / row['tiles'] * 1e3:5.0f} tok/s   "
          f"table of sums {row['table']:6.0f} ms {n / row['table'] * 1e3:5.0f} tok/s   (now: {now})")

print(f"  peak memory so far {mx.get_peak_memory() / 1e9:.1f} GB")

print("\na prompt as the engine reads it (cut for its checkpoints), 1,180 tokens:")
whole = []
long = prompt[:1180]
for label, e in (("no draft head", engine),) + ((("with the draft head", with_head),) if with_head else ()):
    e.reset()
    e.read(long)
    ts, pieces = [], []
    for _ in range(3):
        time.sleep(4)
        e.reset()
        pieces = []
        t0 = time.perf_counter()
        e.read(long, on_chunk=lambda b, n: pieces.append(b))
        ts.append((time.perf_counter() - t0) * 1e3)
    whole.append({"engine": label, "ms": median(ts), "pieces_end_at": pieces})
    print(f"  {label:20} {median(ts):7.0f} ms  {len(long) / median(ts) * 1e3:6.0f} tok/s   pieces end at {pieces}")

if args.json:
    Path(args.json).write_text(json.dumps({"tokens": T, "ms": median(plain), "ms_with_stops": total, "parts": rows,
                                           "without_the_ten_chosen": left_out, "filling": filling, "by_length": lengths, "as_read": whole}, indent=1) + "\n")
