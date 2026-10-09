"""How fast experts kept as the file's blocks multiply, against MLX's own 2-bit gather, at the model's shapes.

Random blocks: 512 experts, ten to a token. Each timing is a chain of calls evaluated together, as in a pass.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strata_mlx.blocks import BLOCK, BlockSwitchLinear

E, K, HIDDEN, INNER = 512, 10, 2560, 640
p = argparse.ArgumentParser()
p.add_argument("--kinds", nargs="+", default=["IQ4_NL", "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_S"])
p.add_argument("--windows", nargs="+", type=int, default=[1, 2, 4, 8])
p.add_argument("--per-row", nargs="+", type=int, default=[0])
p.add_argument("--calls", type=int, default=48)
p.add_argument("--dtype", default="float16")
p.add_argument("--out")
args = p.parse_args()
dtype = getattr(mx, args.dtype)
rng = np.random.default_rng(0)


def blocks(kind, rows, cols):
    per_block, size = BLOCK[kind] if kind in BLOCK else (64, 18)
    b = rng.integers(0, 256, (E * rows * cols // per_block, size), dtype=np.uint8)
    if kind != "IQ1_M":
        b[:, :2] = np.frombuffer(np.float16(0.01).tobytes(), np.uint8)
    return b


def mlx_q2(rows, cols):
    b = blocks("Q2_0", rows, cols)
    words = mx.array(np.ascontiguousarray(b[:, 2:]).view("<u4").reshape(E, rows, cols // 16))
    d = mx.array(np.ascontiguousarray(b[:, :2]).view("<f2").reshape(E, rows, cols // 64).astype(args.dtype))
    return lambda x, idx: mx.gather_qmm(x, words, d, -d, rhs_indices=idx, transpose=True, group_size=64, bits=2)


def time_chain(fn, T, cols, share):
    xs = [mx.random.normal((T, 1 if share else K, 1, cols)).astype(dtype) for _ in range(args.calls)]
    idx = [mx.array(rng.integers(0, E, (T, K))) for _ in range(args.calls)]
    mx.eval(xs, idx)
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval([fn(x, i) for x, i in zip(xs, idx)])
        best = min(best, time.perf_counter() - t0)
    return best / args.calls * 1e6  # microseconds a call


rows_out = []
for name, rows, cols, share in (("gate/up", INNER, HIDDEN, True), ("down", HIDDEN, INNER, False)):
    ref = mlx_q2(rows, cols)
    base = {T: time_chain(ref, T, cols, share) for T in args.windows}
    print(f"{name:8} {'MLX Q2_0':10}        " + "  ".join(f"T={T}: {base[T]:6.0f} us" for T in args.windows), flush=True)
    for kind in args.kinds:
        if cols % BLOCK[kind][0]:
            continue
        data = mx.array(blocks(kind, rows, cols).reshape(-1))
        mx.eval(data)
        for per_row in args.per_row:
            layer = BlockSwitchLinear(data, kind, (E, rows, cols), per_row=per_row or None)
            us = {T: time_chain(layer, T, cols, share) for T in args.windows}
            nbytes = K * rows * cols // BLOCK[kind][0] * BLOCK[kind][1]
            print(f"{name:8} {kind:10} P={layer.per_row:<3}  " + "  ".join(f"T={T}: {us[T]:6.0f} us" for T in args.windows)
                  + f"   {nbytes / us[1] / 1e3:5.0f} GB/s at T=1", flush=True)
            rows_out.append({"tensor": name, "type": kind, "threads_per_row": layer.per_row,
                             "us": us, "mlx_q2_0_us": base})
        del data
if args.out:
    json.dump(rows_out, open(args.out, "w"), indent=1)
