"""Experiment 23: the Q6_K kernels of e21 and e22 in a real pass of the IQ3_S model.

Alone, against MLX's product over float16 (130 us for a 10,240 x 2,560 tensor whatever the tokens): the blocks
72 / 83 / 106 / 146 / 357 us for 1 / 2 / 4 / 6 / 8 tokens, a byte a weight 80 / 81 / 105 / 93 / 124. Parts timed
alone do not add up to a pass (e04, e06), so here the model's Q6_K layers are swapped for them and whole passes
are timed, in turns with the pass as it is.

    uv run python bench/experiments/e23_q6k_in_pass.py --model <IQ3_S shard 1>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from e21_q6k_tiles import q6k_matvec as blocks_matvec  # noqa: E402
from e22_q6k_codes import codes_matvec, q6k_codes  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.gguf import GGUF  # noqa: E402
from strata_mlx.load import load, v_heads_grouped  # noqa: E402

ZERO = mx.zeros((1,), dtype=mx.float16)


class Alt:
    """A Q6_K layer three ways: as the model has it (float16), on the file's blocks, as a byte a weight."""

    policy = "float"

    def __init__(self, t, original, rows=None, cols=None):
        self.rows, self.cols = t.shape
        data = np.asarray(t.data).reshape(self.rows, -1)
        if rows is not None:
            data = data[rows]
        data = np.ascontiguousarray(data)
        self.original = original
        self.inv = mx.array(np.argsort(cols)) if cols is not None else None
        self.raw = mx.array(data.reshape(-1))
        self.codes = tuple(mx.array(a) for a in q6k_codes(data, self.rows, self.cols))
        mx.eval(self.raw, self.codes)

    def __call__(self, x):
        T = int(np.prod(x.shape[:-1]))
        how = {"float": "float", "codes": "codes", "blocks": "blocks", "codes16": "codes16",
               "blocks, codes from 3": "blocks" if T <= 2 else "codes",
               "blocks, float from 5": "blocks" if T <= 4 else "float",
               "blocks, codes from 3, float from 7": "blocks" if T <= 2 else "codes" if T <= 6 else "float"}[Alt.policy]
        if how == "float" or T > 8:
            return self.original(x)
        x2 = x.reshape(T, self.cols)
        if self.inv is not None:
            x2 = mx.take(x2, self.inv, axis=-1)
        xt = mx.contiguous(x2.astype(mx.float32).T)
        if how == "blocks":
            opts = dict(sub=1, R=2) if T <= 2 else dict(sub=1, R=4, scaled=True)
            y = blocks_matvec(self.raw, self.rows, self.cols, xt, ZERO, dtype=x.dtype, **opts)
        else:
            y = codes_matvec(*self.codes, xt, ZERO, R=4, V=16 if how == "codes16" else 8, dtype=x.dtype)
        return y.reshape(*x.shape[:-1], self.rows)


def a_pass(engine, T, i):
    ids = mx.array([[1000 + (i * 7 + j) % 500 for j in range(T)]])
    t0 = time.perf_counter()
    logits, w = engine._forward(ids)
    mx.eval(logits)
    engine._commit(w, T)
    return (time.perf_counter() - t0) * 1e3, logits


def quartile(xs):
    return sorted(xs)[len(xs) // 4]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--passes", type=int, default=40)
    p.add_argument("--windows", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model)
    g = GGUF(args.model)
    a = model.args.text
    n_k, n_v, d_k, d_v = a.linear_num_key_heads, a.linear_num_value_heads, a.linear_key_head_dim, a.linear_value_head_dim
    qk = 2 * n_k * d_k
    v_rows = v_heads_grouped(n_k, n_v // n_k, d_v)
    qkv_rows = np.concatenate([np.arange(qk), qk + v_rows])
    places = []  # (owner, attribute, tensor, rows, cols, in the output layer?)
    for i, layer in enumerate(model.model.layers):
        b = f"blk.{i}."
        if layer.layer_type == "linear_attention":
            la = layer.linear_attn
            places += [(la, "in_proj_qkv", b + "attn_qkv.weight", qkv_rows, None), (la, "in_proj_z", b + "attn_gate.weight", v_rows, None),
                       (la, "out_proj", b + "ssm_out.weight", None, v_rows)]
        else:
            sa = layer.self_attn
            places += [(sa, "q_proj", b + "attn_q.weight", None, None), (sa, "k_proj", b + "attn_k.weight", None, None),
                       (sa, "v_proj", b + "attn_v.weight", None, None), (sa, "o_proj", b + "attn_output.weight", None, None)]
        se = layer.mlp.shared_expert
        places += [(se, "gate_proj", b + "ffn_gate_shexp.weight", None, None), (se, "up_proj", b + "ffn_up_shexp.weight", None, None)]
    places.append((model, "lm_head", "output.weight", None, None))
    alts = []
    for owner, name, tensor, rows, cols in places:
        t = g[tensor]
        if t.type == "Q6_K":
            alts.append((owner, name, Alt(t, getattr(owner, name), rows, cols), tensor == "output.weight"))
    print(f"{len(alts)} Q6_K layers; as float16 {sum(al.original.weight.nbytes for *_, al, _ in alts) / 1e9:.2f} GB, in the file "
          f"{sum(al.raw.nbytes for *_, al, _ in alts) / 1e9:.2f} GB, a byte a weight {sum(sum(c.nbytes for c in al.codes) for *_, al, _ in alts) / 1e9:.2f} GB",
          flush=True)

    def patch(which):
        undo = []
        for owner, name, alt, is_head in alts:
            if which == "all" or is_head:
                undo.append((owner, name, getattr(owner, name)))
                setattr(owner, name, alt)
        return lambda: [setattr(o, n, v) for o, n, v in undo]

    whole = Engine(model)
    whole.warm()
    whole.read(list(range(1000, 1200)))
    SETS = [("output layer", "blocks"), ("output layer", "codes"), ("all", "codes"), ("all", "blocks"),
            ("all", "blocks, codes from 3"), ("all", "blocks, float from 5")]
    out = []
    for which, policy in SETS:
        Alt.policy = policy
        whole.reset()  # both from the same 200 tokens: the logits are compared, and a longer conversation costs more
        whole.read(list(range(1000, 1200)))
        undo = patch(which)
        engine = Engine(model)
        engine.warm()
        engine.read(list(range(1000, 1200)))
        undo()
        line, row = f"{which:12} {policy:36}", {"layers": which, "policy": policy, "ms": {}}
        for T in args.windows:
            base, ours, far = [], [], 0.0
            for i in range(args.passes):
                ms, ref = a_pass(whole, T, i)
                base.append(ms)
                undo = patch(which)
                ms, got = a_pass(engine, T, i)
                undo()
                ours.append(ms)
                if i < 4:
                    far = max(far, mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item())
            row["ms"][T] = {"as it is": round(quartile(base), 2), "swapped": round(quartile(ours), 2), "logits apart": round(far, 3)}
            line += f"  {T}: {quartile(base):6.2f} -> {quartile(ours):6.2f} ({quartile(ours) - quartile(base):+5.2f})"
        out.append(row)
        print(line + f"   logits apart by at most {max(v['logits apart'] for v in row['ms'].values())}", flush=True)
        del engine
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
