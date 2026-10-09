"""Experiment 12: the table-lookup expert kernel inside real passes.

Experiments 2, 3, 10 and 11 together: the expert kernel is bound by arithmetic, about eight operations a weight
at all this GPU has (loading four values at once, or sharing the activations between rows, changed nothing;
leaving the arithmetic out did). A table of the 16 sums two weights can make of two inputs brings it to about
three operations a weight: -20% alone for four tokens, nothing for one. Here in a pass, from a number of
(token, expert) pairs on, against the engine as it is.

    uv run python bench/experiments/e12_lookup_in_pass.py --model <shard 1>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.activations import swiglu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx import blocks  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

# for every two inputs, the 16 sums two weights (each -1, 0, 1 or 2) can make of them
TABLE = """
    uint g = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    float a = float(x[n * COLS + 2 * g]), b = float(x[n * COLS + 2 * g + 1]);
    auto t = table + (n * (COLS / 2) + g) * 16;
    for (uint c = 0; c < 16; ++c) {
        t[c] = static_cast<OutT>((float(c & 3) - 1.0f) * a + (float(c >> 2) - 1.0f) * b);
    }
"""
LOOKUP = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    auto t_ = table + (pair / SHARE) * (COLS / 2) * 16;
    float sum = 0.0f;
    for (uint n = part; n < RUNS; n += P) {
        const device uchar* b = row + (n / 2) * 18;
        const device uchar* q = b + 2 + 8 * (n % 2);
        auto t = t_ + 256 * n;
        float s = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            uint c = q[i];
            s += float(t[32 * i + (c & 15)]) + float(t[32 * i + 16 + (c >> 4)]);
        }
        sum += half_at(b) * s;
    }
    float total = sum;
    for (ushort k = 1; k < P; ++k) {
        total += simd_shuffle_xor(sum, k);
    }
    if (part == 0) {
        out[pair * ROWS + r] = static_cast<OutT>(total);
    }
"""
_kernels = {}


def table(x: mx.array) -> mx.array:
    n, cols = x.shape
    if cols not in _kernels:
        _kernels[cols] = mx.fast.metal_kernel(name=f"lut4_table_{cols}", input_names=["x"], output_names=["table"],
                                              source=TABLE, header=f"constant uint COLS = {cols};\n")
    return _kernels[cols](inputs=[x], template=[("OutT", x.dtype)], grid=(cols // 2, n, 1), threadgroup=(32, 1, 1),
                          output_shapes=[(n, cols // 2, 16)], output_dtypes=[x.dtype])[0]


def lookup(lin, idx: mx.array, tab: mx.array, share: int, per_row: int) -> mx.array:
    key = (lin.output_dims, lin.input_dims, share, per_row)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"lut4_matvec_{share}_{per_row}", input_names=["blocks", "idx", "table"], output_names=["out"], source=LOOKUP,
            header=blocks._header("Q2_0", lin.output_dims, lin.input_dims, P=per_row, SHARE=share))
    pairs = idx.shape[0]
    return _kernels[key](inputs=[lin.blocks, idx, tab], template=[("OutT", tab.dtype)],
                         grid=(per_row * lin.output_dims, pairs, 1), threadgroup=(32, 1, 1),
                         output_shapes=[(pairs, lin.output_dims)], output_dtypes=[tab.dtype])[0]


def variant(from_pairs: int, wide: int, narrow: int):
    """switch_glu with lookups from `from_pairs` pairs on; `wide`, `narrow`: threads to a row of gate/up, of down."""
    plain = blocks.switch_glu

    def switch_glu(x, indices, up, gate, down):
        if indices.size < from_pairs or x.dtype != mx.float16 or any(l.kind != "Q2_0" for l in (up, gate, down)):
            return plain(x, indices, up, gate, down)
        k = indices.shape[-1]
        idx = indices.reshape(-1).astype(mx.uint32)
        tab = table(x.reshape(-1, x.shape[-1]))
        h = swiglu(lookup(gate, idx, tab, k, wide), lookup(up, idx, tab, k, wide))
        return lookup(down, idx, table(h), 1, narrow).reshape(*indices.shape, down.output_dims)
    return switch_glu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--passes", type=int, default=60)
    p.add_argument("--windows", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model)
    plain = blocks.switch_glu
    engines = {"as it is": Engine(model)}
    engines["as it is"].warm()
    for name, v in (("lookups 16/4", variant(1, 16, 4)), ("lookups 4/4", variant(1, 4, 4)), ("lookups 16/1", variant(1, 16, 1))):
        blocks.switch_glu = v
        engines[name] = Engine(model)
        engines[name].warm()  # traces every window size with this variant
        blocks.switch_glu = plain
    for e in engines.values():
        e.read(list(range(1000, 1200)))
    rows = []
    for T in args.windows:
        times, last = {name: [] for name in engines}, {}
        for i in range(args.passes):
            ids = mx.array([[1000 + (i * 7 + j) % 500 for j in range(T)]])
            for name, e in engines.items():
                t0 = time.perf_counter()
                logits, w = e._forward(ids)
                mx.eval(logits)
                e._commit(w, T)
                times[name].append((time.perf_counter() - t0) * 1e3)
                last[name] = logits
        q = {name: sorted(ts)[len(ts) // 4] for name, ts in times.items()}
        rows.append({"tokens": T, **q})
        print(f"{T} tokens: " + "   ".join(f"{name} {ms:6.2f} ms ({ms - q['as it is']:+.2f})" for name, ms in q.items()), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
