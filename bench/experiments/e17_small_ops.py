"""Experiment 17: which of MLX's own small operations in a pass are slow?

The output layer's product takes 0.7 ms of a 1.1 ms call; picking the largest of its 248,320 logits takes the
rest. A layer's router is a small product, a partial sort of 512 logits and a softmax over ten. Each is timed
in a chain of 48, every step waiting for the one before, for one token and for four.

    uv run python bench/experiments/e17_small_ops.py
"""

import time

import mlx.core as mx

N = 48


def best(build, rounds=15):
    out = float("inf")
    for _ in range(rounds):
        g = build()
        t0 = time.perf_counter()
        mx.eval(g)
        out = min(out, time.perf_counter() - t0)
    return out / N * 1e6


def chain(step, x):
    def build():
        y, dep = x, None
        for _ in range(N):
            dep = step(y if dep is None else y + 0.0 * dep.reshape(-1)[:1].astype(y.dtype))
        return dep
    return build


for T in (1, 4):
    logits = (mx.random.normal((T, 248320)) * 4).astype(mx.float16)
    r = mx.random.normal((T, 512))
    x = mx.random.normal((T, 2560)).astype(mx.float16)
    w = mx.random.normal((512, 2560)).astype(mx.float16)
    mx.eval(logits, r, x, w)

    def router(v):
        l = v.astype(mx.float32) @ w.T
        idx = mx.argpartition(-l, 9, axis=-1)[..., :10]
        return mx.softmax(mx.take_along_axis(l, idx, axis=-1), axis=-1, precise=True) + idx

    steps = {
        "an add on 248,320 (the chain's own cost)": (lambda v: v + 1, logits),
        "argmax of 248,320": (lambda v: mx.argmax(v, axis=-1), logits),
        "max of 248,320": (lambda v: mx.max(v, axis=-1), logits),
        "logsumexp of 248,320": (lambda v: mx.logsumexp(v.astype(mx.float32), axis=-1), logits),
        "an add on 512": (lambda v: v + 1, r),
        "argpartition of 512, ten kept": (lambda v: mx.argpartition(-v, 9, axis=-1)[..., :10], r),
        "argsort of 512": (lambda v: mx.argsort(-v, axis=-1)[..., :10], r),
        "topk of 512": (lambda v: mx.topk(v, 10, axis=-1), r),
        "softmax (precise) of ten": (lambda v: mx.softmax(v[..., :10], axis=-1, precise=True), r),
        "a router: product, partial sort, softmax": (router, x),
        "the router's product alone": (lambda v: v.astype(mx.float32) @ w.T, x),
    }
    for name, (step, start) in steps.items():
        print(f"{T} token{'s' if T > 1 else ' '}  {name:42} {best(chain(step, start)):7.1f} us", flush=True)
