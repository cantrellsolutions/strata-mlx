"""Experiment 9: the three ways of a hyper-connection as a chain of 97, outside a pass.

Experiment 7's new kernels did nothing inside a pass (one token +1.2 ms, four -0.5, eight +10) although
experiment 8 measured their slowest part twice as fast alone. Here whole connections, each fed by the one
before as in a pass, with nothing else between them: if the new kernels win here and not in a pass, it is not
the GPU that a pass waits for at that point.

    uv run python bench/experiments/e09_hc_chain.py --model <shard 1>
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from e07_hc_multi import MultiResidual  # noqa: E402

from strata_mlx.fused import FusedResidual, PlainResidual  # noqa: E402
from strata_mlx.load import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
args = p.parse_args()
model, _ = load(args.model)
m = model.model
grs = [gr for l in m.layers for gr in (l.attn_hyper_connection, l.mlp_hyper_connection)]
d, hc = m.args.hidden_size, m.hc
WAYS = {
    "two kernels a token (fused.py)": lambda gr: FusedResidual(gr),
    "MLX's own operations": lambda gr: PlainResidual(gr),
    "read once, 1024/32": lambda gr: MultiResidual(gr, 32, 32),
    "read once, 1024/8": lambda gr: MultiResidual(gr, 32, 8),
    "read once, 256/8": lambda gr: MultiResidual(gr, 8, 8),
    "read once, 32/8": lambda gr: MultiResidual(gr, 1, 8),
}
print(f"{'microseconds a connection':34}" + "".join(f"{f'{T} tokens':>12}" for T in (1, 2, 4, 8)))
for name, make in WAYS.items():
    mods = [make(gr) for gr in grs]
    line = ""
    for T in (1, 2, 4, 8):
        before = mx.random.normal((1, T, hc * d)).astype(mx.float16)
        out = mx.random.normal((1, T, d)).astype(mx.float16)
        inject = mx.random.uniform(shape=(1, T, hc)).astype(mx.float16)
        mx.eval(before, out, inject)

        def chain():
            b, o, i = before, out, inject
            for mod in mods:
                o, b, i = mod(b, o, i)
            return o, b, i
        best = float("inf")
        for _ in range(9):
            g = chain()
            t0 = time.perf_counter()
            mx.eval(*g)
            best = min(best, time.perf_counter() - t0)
        line += f"{best / len(mods) * 1e6:12.1f}"
    print(f"{name:34}{line}", flush=True)
