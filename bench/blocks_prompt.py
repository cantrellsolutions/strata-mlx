"""Reading a prompt through experts kept as blocks: the per-pair kernel against unpacking the stack to floats.

Pairs arrive sorted by expert, one activation row each, as mlx-lm's SwitchGLU sends them from 64 pairs on.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strata_mlx.blocks import BLOCK, BlockSwitchLinear  # noqa: E402

E, K = 512, 10
p = argparse.ArgumentParser()
p.add_argument("--tokens", nargs="+", type=int, default=[8, 64, 256, 1024, 2048])
p.add_argument("--dtype", default="float16")
args = p.parse_args()
dtype = getattr(mx, args.dtype)
rng = np.random.default_rng(0)


def blocks(kind, rows, cols):
    per_block, size = BLOCK.get(kind, (64, 18))
    b = rng.integers(0, 256, (E * rows * cols // per_block, size), dtype=np.uint8)
    b[:, :2] = np.frombuffer(np.float16(0.01).tobytes(), np.uint8)
    return b


def best(fn, x, idx, n=3):
    out = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fn(x, idx))
        out = min(out, time.perf_counter() - t0)
    return out * 1e3


for name, kind, rows, cols in (("gate/up", "IQ3_XXS", 640, 2560), ("gate/up", "IQ2_S", 640, 2560),
                               ("down", "IQ4_NL", 2560, 640)):
    b = blocks("Q2_0", rows, cols)
    words = mx.array(np.ascontiguousarray(b[:, 2:]).view("<u4").reshape(E, rows, cols // 16))
    d = mx.array(np.ascontiguousarray(b[:, :2]).view("<f2").reshape(E, rows, cols // 64).astype(args.dtype))
    data = mx.array(blocks(kind, rows, cols).reshape(-1))
    layer = BlockSwitchLinear(data, kind, (E, rows, cols))
    mx.eval(words, d, data)
    for T in args.tokens:
        idx = mx.array(np.sort(rng.integers(0, E, T * K)))
        x = mx.random.normal((T * K, 1, cols)).astype(dtype)
        mx.eval(idx, x)
        q2 = best(lambda x, i: mx.gather_qmm(x, words, d, -d, rhs_indices=i, transpose=True, group_size=64, bits=2,
                                             sorted_indices=True), x, idx)
        BlockSwitchLinear.unpack_from = 1 << 62
        per_pair = best(layer, x, idx)
        BlockSwitchLinear.unpack_from = 0
        floats = best(lambda x, i: layer(x, i, sorted_indices=True), x, idx)
        print(f"{name:8} {kind:8} {T:5} tokens   MLX Q2_0 {q2:8.2f} ms   per pair {per_pair:8.2f} ms   "
              f"unpacked {floats:8.2f} ms", flush=True)
    del data, words, d
