"""e59: the delta-net recurrence of a decoding pass, by its parts and by kernel.

36 layers of this model's shapes (16 key heads, 48 value heads of 128 x 128, a float32 state), chained (each
layer's v waits on the one before's output), for 1 and 4 tokens:

  now          mlx-lm's gated_delta_update as the engine calls it (g and beta, then the packed kernel)
  unpacked     g and beta, then mlx-lm's one-row-per-SIMD-group kernel
  packed only, unpacked only   the kernels alone, g and beta made beforehand
  g, beta only                 what comes before the kernel

Shuffled in one process, the lower quartile; whether unpacked gives the packed kernel's bits is checked.

    uv run python bench/experiments/e59_recurrence_kernels.py --json bench/results/e59_recurrence_kernels.json
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models import gated_delta as gd

LAYERS, HK, HV, D = 36, 16, 48, 128


def layer_inputs(rng, T):
    f = lambda *s: mx.array(rng.standard_normal(s).astype(np.float16))
    q, k = f(1, T, HK, D), f(1, T, HK, D)
    q = q * (D ** -0.5) / mx.sqrt(mx.mean(q.astype(mx.float32) ** 2, -1, keepdims=True)).astype(q.dtype)
    k = k / mx.sqrt(mx.mean(k.astype(mx.float32) ** 2, -1, keepdims=True) * D).astype(k.dtype)
    return {"q": q, "k": k, "v": f(1, T, HV, D), "a": f(1, T, HV), "b": f(1, T, HV),
            "A_log": mx.array(rng.uniform(-1, 1, HV).astype(np.float32)), "dt_bias": f(HV),
            "state": mx.array((0.05 * rng.standard_normal((1, HV, D, D))).astype(np.float32))}


def gb(x):
    return gd.compute_g(x["A_log"], x["a"], x["dt_bias"]), mx.sigmoid(x["b"])


VARIANTS = {
    "now": lambda x, v: gd.gated_delta_update(x["q"], x["k"], v, x["a"], x["b"], x["A_log"], x["dt_bias"], x["state"]),
    "unpacked": lambda x, v: gd.gated_delta_kernel_unpacked(x["q"], x["k"], v, *gb(x), x["state"]),
    "packed only": lambda x, v: gd.gated_delta_kernel(x["q"], x["k"], v, x["g"], x["beta"], x["state"]),
    "unpacked only": lambda x, v: gd.gated_delta_kernel_unpacked(x["q"], x["k"], v, x["g"], x["beta"], x["state"]),
    "g, beta only": lambda x, v: (v + (gb(x)[0][..., None] * 0).astype(v.dtype), x["state"]),
}


def chain(fn, layers):
    y = None
    outs = []
    for x in layers:
        v = x["v"] if y is None else x["v"] + (y[..., :1, :1] * 0).astype(x["v"].dtype)
        y, s = fn(x, v)
        outs.append(s)
    return [y] + outs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=150)
    p.add_argument("--json")
    args = p.parse_args()
    rng = np.random.default_rng(59)
    rows = []
    for T in (1, 4):
        layers = [layer_inputs(rng, T) for _ in range(LAYERS)]
        for x in layers:
            x["g"], x["beta"] = gb(x)
        mx.eval(layers)
        y1, s1 = VARIANTS["now"](layers[0], layers[0]["v"])
        y2, s2 = VARIANTS["unpacked"](layers[0], layers[0]["v"])
        same = mx.array_equal(y1, y2).item() and mx.array_equal(s1, s2).item()
        print(f"{T} token(s): unpacked gives the packed kernel's bits: {same}", flush=True)
        for fn in VARIANTS.values():
            mx.eval(chain(fn, layers))
        times = {name: [] for name in VARIANTS}
        order = list(VARIANTS) * args.rounds
        random.Random(T).shuffle(order)
        for name in order:
            t0 = time.perf_counter()
            mx.eval(chain(VARIANTS[name], layers))
            times[name].append(time.perf_counter() - t0)
        for name, ts in times.items():
            ms = statistics.quantiles(ts, n=4)[0] * 1e3
            rows.append({"tokens": T, "variant": name, "ms_36_layers": round(ms, 3), "same_bits": same})
            print(f"  {name:<14} {ms:6.3f} ms for 36 layers ({ms / LAYERS * 1e3:5.1f} us a layer)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
