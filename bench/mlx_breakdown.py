"""Where a step of the Qwen3.8-Flash-Next graph spends its time in MLX, by component.

Same random-weight model as mlx_overhead.py. Each component is run for all of its layers inside one graph and
evaluated once, so it pays the per-evaluation cost once, as it does inside a real step. The rows do not have to
add up to the full step: the full graph can overlap work that the isolated ones cannot.

    uv run bench/mlx_breakdown.py --tokens 1 4 --context 1024
    uv run bench/mlx_breakdown.py --tokens 2048 --context 0 --reps 3
"""

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

from mlx_overhead import arrays, build, fresh, run


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*arrays(fn()))
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out)


def residual(gr, h, x=None):
    """A hyper-connection read and write around `x` (the read's own output when no mixer is given)."""
    mixed, hyper, inject = gr(h)
    x = mixed if x is None else x
    return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


def breakdown(model, T, context, reps):
    m = model.model
    vocab = model.args.text.vocab_size
    d, hc = m.args.hidden_size, m.hc
    gdn = [l for l in m.layers if l.layer_type == "linear_attention"]
    qsa = [l for l in m.layers if l.layer_type == "full_attention"]

    cache = fresh(model, vocab, context)
    by_layer = {id(l): c for l, c in zip(m.layers, cache)}
    x = mx.random.normal((1, T, d)).astype(mx.float16)
    h = mx.tile(x, (1, 1, hc))
    idx = mx.random.randint(0, m.args.num_experts, (1, T, m.args.num_experts_per_tok))
    mx.eval(x, h, idx)

    def chain(fn, layers, start):
        def go():
            y = start
            for layer in layers:
                y = fn(layer, y)
            return y
        return go

    def all_layers(fn, layers, start):
        return lambda: [fn(layer, start) for layer in layers]

    def qsa_step(layer, y):
        c = by_layer[id(layer)]
        return layer.self_attn(y, m.rope, create_attention_mask(y, c), c, c.indexer)

    rows = {
        "hyper-connections (96 read+write)": chain(
            lambda l, y: residual(l.mlp_hyper_connection, residual(l.attn_hyper_connection, y)), m.layers, h),
        "GDN mixers (36)": chain(lambda l, y: l.linear_attn(y, None, by_layer[id(l)]), gdn, x),
        "QSA attention (12)": chain(qsa_step, qsa, x),
        "MoE router + top-10 (48)": all_layers(
            lambda l, y: mx.argpartition(-l.mlp.gate(y.astype(mx.float32)), 9, axis=-1)[..., :10], m.layers, x),
        "MoE routed experts (48)": chain(lambda l, y: l.mlp.switch_mlp(y, idx).sum(axis=-2), m.layers, x),
        "MoE shared expert (48)": chain(lambda l, y: l.mlp.shared_expert(y), m.layers, x),
        "MoE block, whole (48)": chain(lambda l, y: l.mlp(y), m.layers, x),
        "final mixer + head + argmax": lambda: mx.argmax(model.lm_head(m.hyper_connection_mixer(h)), axis=-1),
    }
    print(f"\nT = {T} tokens at a {context}-token context, medians of {reps}")
    for name, fn in rows.items():
        fn()  # the first call compiles kernels
        print(f"  {name:<36} {timed(fn, reps):>9.2f} ms")

    cache = fresh(model, vocab, context)
    ids = mx.random.randint(0, vocab, (1, T))
    run(model, cache, ids)
    full = statistics.median(sum(run(model, cache, ids)) for _ in range(reps)) * 1e3
    print(f"  {'the full step':<36} {full:>9.2f} ms")
    empty = timed(lambda: x + 1, reps)
    print(f"  {'one empty evaluation':<36} {empty:>9.2f} ms")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 4])
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--reps", type=int, default=20)
    args = p.parse_args()
    mx.random.seed(0)
    model = build(2, 4)
    for T in args.tokens:
        breakdown(model, T, args.context, args.reps)


if __name__ == "__main__":
    main()
