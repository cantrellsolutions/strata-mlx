"""Experiment 36: the Q3_K and IQ4_XS dense tensors as the file's blocks, against MLX's 8-bit form of them.

On a machine that reads experts from the SSD, memory is experts: the Q2_0 file's dense tensors take 5.5 GB as
arrays for 3.6 GB of file, 1.7 GB of the difference in these two types (2.9 GB of arrays for 1.2 GB of blocks).
What the blocks cost in time, for all of a pass's products of these types at once, the embedding left out (it
is a lookup): a window of 1, 2, 4 and 8 tokens through blocks.window_matvec, a chunk of a prompt through the
tiles and through weights unpacked for the chunk and multiplied by MLX.

    uv run python bench/experiments/e36_dense_as_blocks.py --model <shard 1> --json bench/results/e36_dense_as_blocks.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.blocks import TOKENS_TILE, dequantize, tile_matmul  # noqa: E402
from strata_mlx.gguf import GGUF  # noqa: E402
from strata_mlx.load import AS_BLOCKS, linear  # noqa: E402


def best(run, rounds: int) -> float:
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--chunk", type=int, nargs="+", default=[256, 1024])
    p.add_argument("--json")
    args = p.parse_args()

    g = GGUF(args.model)
    names = [n for n, t in g.tensors.items()
             if t.type in AS_BLOCKS and len(t.shape) == 2 and not n.endswith("_exps.weight") and n != "token_embd.weight"]
    layers = {}
    for n in names:
        t = g[n]
        q = linear(t)
        q.set_dtype(mx.float16)  # as the loader leaves it: float32 scales cost MLX's product half its speed
        b = linear(t, as_blocks=AS_BLOCKS)
        mx.eval(q.parameters(), b.blocks)
        layers[n] = (t.type, t.shape, q, b)
    print(f"{len(layers)} tensors, {sum(np.prod(v[1]) for v in layers.values()) / 1e9:.2f} G weights")
    out = {"tensors": len(layers), "windows": {}, "chunks": {}}

    def products(form: int, x_of, kind=None):
        return [v[form](x_of[v[1][1]]) for v in layers.values() if kind is None or v[0] == kind]

    widths = sorted({v[1][1] for v in layers.values()})
    print(f"\nall products of a pass, ms (best of {args.rounds}, the two in turn)")
    print(f"{'tokens':>7} {'MLX 8-bit':>10} {'blocks':>8} {'more':>7}   by type: " + ", ".join(AS_BLOCKS))
    for tokens in (1, 2, 4, 8):
        x_of = {w: mx.random.normal((1, tokens, w)).astype(mx.float16) for w in widths}
        mx.eval(x_of)
        for form in (2, 3):
            mx.eval(products(form, x_of))  # compile the kernels
        row, parts = {}, []
        for kind in (None, *AS_BLOCKS):
            a, b = [], []
            for _ in range(args.rounds):
                a.append(best(lambda: mx.eval(products(2, x_of, kind)), 1))
                b.append(best(lambda: mx.eval(products(3, x_of, kind)), 1))
            row[kind or "all"] = (1e3 * min(a), 1e3 * min(b))
        parts = "  ".join(f"{row[k][0]:.2f} / {row[k][1]:.2f}" for k in AS_BLOCKS)
        print(f"{tokens:7d} {row['all'][0]:10.2f} {row['all'][1]:8.2f} {row['all'][1] - row['all'][0]:+7.2f}   {parts}")
        out["windows"][tokens] = row

    print(f"\na chunk of a prompt, all products, ms")
    print(f"{'tokens':>7} {'MLX 8-bit':>10} {'tiles':>8} {'unpacked, MLX matmul':>22}")
    for tokens in args.chunk:
        x_of = {w: mx.random.normal((1, tokens, w)).astype(mx.float16) for w in widths}
        mx.eval(x_of)

        def unpacked():
            outs = []
            for kind, shape, _, b in layers.values():
                x = x_of[shape[1]]
                if b.to_file is not None:
                    x = mx.take(x, b.to_file, axis=-1)
                outs.append(x @ dequantize(b.blocks, kind, shape, mx.float16).T)
            return outs

        def tiles():
            """Each layer as a stack of one expert whose tiles are the chunk's tokens."""
            outs, n = [], -(-tokens // TOKENS_TILE)
            which = mx.zeros((n,), dtype=mx.uint32)
            for kind, shape, _, b in layers.values():
                x = x_of[shape[1]].reshape(tokens, -1)
                if b.to_file is not None:
                    x = mx.take(x, b.to_file, axis=-1)
                if tokens % TOKENS_TILE:
                    x = mx.concatenate([x, mx.zeros((n * TOKENS_TILE - tokens, shape[1]), dtype=x.dtype)])
                outs.append(tile_matmul(b.blocks, kind, (1, *shape), x, which)[:tokens].astype(mx.float16))
            return outs

        runs = {"mlx": lambda: mx.eval(products(2, x_of)), "tiles": lambda: mx.eval(tiles()),
                "unpacked": lambda: mx.eval(unpacked())}
        for run in runs.values():
            run()
        times = {k: [] for k in runs}
        for _ in range(args.rounds):
            for k, run in runs.items():
                times[k].append(best(run, 1))
        row = {k: 1e3 * min(v) for k, v in times.items()}
        print(f"{tokens:7d} {row['mlx']:10.1f} {row['tiles']:8.1f} {row['unpacked']:22.1f}")
        out["chunks"][tokens] = row

    sizes = {}
    for kind in AS_BLOCKS:
        mlx_b = sum(sum(a.nbytes for a in (v[2].weight, v[2].scales, v[2].biases)) for v in layers.values() if v[0] == kind)
        blk_b = sum(v[3].nbytes for v in layers.values() if v[0] == kind)
        sizes[kind] = (mlx_b, blk_b)
        print(f"{kind}: {mlx_b / 1e6:.0f} MB as MLX holds it, {blk_b / 1e6:.0f} MB of blocks")
    out["bytes"] = sizes
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
