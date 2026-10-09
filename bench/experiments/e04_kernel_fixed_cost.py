"""Experiment 4: what does one more kernel cost, MLX's own against a custom one?

Experiment 3: an expert kernel that computes nothing still takes 29 microseconds, where MLX's small products take
8. Chains of kernels each feeding the next, as a pass is, and sets of independent ones, timed for:

    add        MLX's own elementwise add on 2,560 floats
    custom     a custom kernel doing the same
    threads    the same custom kernel started on a grid of 25,600 or 102,400 threads (most return at once)
    big input  the same, with a 236 MB array bound as an input it reads one byte of

    uv run python bench/experiments/e04_kernel_fixed_cost.py
"""

import time

import mlx.core as mx

N = 480
x = mx.random.normal((2560,)).astype(mx.float16)
big = mx.zeros((236 << 20,), dtype=mx.uint8)
mx.eval(x, big)

plain = mx.fast.metal_kernel(name="e04_plain", input_names=["x"], output_names=["out"], source="""
    uint i = thread_position_in_grid.x;
    if (i < 2560 && thread_position_in_grid.y == 0) { out[i] = x[i] + T(1); }
""")
with_big = mx.fast.metal_kernel(name="e04_big", input_names=["x", "big"], output_names=["out"], source="""
    uint i = thread_position_in_grid.x;
    if (i < 2560 && thread_position_in_grid.y == 0) { out[i] = x[i] + T(big[i * 4096] + 1); }
""")


def custom(y, grid=(2560, 1, 1), group=(32, 1, 1)):
    return plain(inputs=[y], template=[("T", y.dtype)], grid=grid, threadgroup=group, output_shapes=[(2560,)],
                 output_dtypes=[y.dtype])[0]


def custom_big(y):
    return with_big(inputs=[y, big], template=[("T", y.dtype)], grid=(2560, 1, 1), threadgroup=(32, 1, 1),
                    output_shapes=[(2560,)], output_dtypes=[y.dtype])[0]


VARIANTS = {
    "add (MLX's own)": lambda y: y + 1,
    "custom, 2,560 threads": custom,
    "custom, 25,600 threads": lambda y: custom(y, (2560, 10, 1)),
    "custom, 102,400 threads": lambda y: custom(y, (10240, 10, 1)),
    "custom, threadgroups of 256": lambda y: custom(y, (2560, 1, 1), (256, 1, 1)),
    "custom, a 236 MB input": custom_big,
}


def chain(step):
    y = x
    for _ in range(N):
        y = step(y)
    return y


def independent(step):
    return [step(x) for _ in range(N)]


def measure(build, rounds=9):
    """Best time to build the graph, and to evaluate it, in microseconds a kernel."""
    b = e = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        g = build()
        t1 = time.perf_counter()
        mx.eval(g)
        b, e = min(b, t1 - t0), min(e, time.perf_counter() - t1)
    return b / N * 1e6, e / N * 1e6


print(f"{'microseconds a kernel':30}{'chain: build':>14}{'evaluate':>10}{'  independent: build':>20}{'evaluate':>10}")
for name, step in VARIANTS.items():
    cb, ce = measure(lambda: chain(step))
    ib, ie = measure(lambda: independent(step))
    print(f"{name:30}{cb:14.1f}{ce:10.1f}{ib:20.1f}{ie:10.1f}", flush=True)
for name in ("add (MLX's own)", "custom, 2,560 threads", "custom, 25,600 threads"):
    step = VARIANTS[name]
    f = mx.compile(lambda y: chain(step))
    mx.eval(f(x))
    best = float("inf")
    for _ in range(9):
        t0 = time.perf_counter()
        mx.eval(f(x))
        best = min(best, time.perf_counter() - t0)
    print(f"{name + ', compiled chain':45}{best / N * 1e6:10.1f} in all", flush=True)
