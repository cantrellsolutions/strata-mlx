"""Experiment 7: hyper-connection kernels that read their weights once for a whole window.

Experiment 6: the hyper-connections cost 5.1 ms of a one-token pass and 8.1 ms of a four-token one, for 1.27 GB
of weights that this GPU reads in about 3 ms. The fused kernels (fused.py) run once per token, reading the
weights again for each; from three tokens on the engine falls back to MLX's own operations. Here:

    norm    the residual and its normalized form, once per token and stream (it used to be recomputed per output)
    down    each weight read once and multiplied into every token's sum
    up      the same

built for a fixed number of tokens each. A first version with 32 threads to a row of `down` was slower than
MLX's own from one token to eight; experiment 8 showed why (a thread should read ten weights, not 320), so the
rows are now shared by up to 1,024 threads through threadgroup memory. Timed inside real passes, alternating
with the engine as it is; the columns are threads to a row of `down` / of `up`.

    uv run python bench/experiments/e07_hc_multi.py --model <shard 1>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

# 32 * NG threads to a token's stream, each with D / (32 * NG) of its values
_NORM = """
    uint lane = thread_position_in_threadgroup.x;
    uint g = thread_position_in_grid.x / (32 * NG);
    uint s = g % HC;
    uint t = g / HC;
    threadgroup float shared[NG];
    float back = float(inject_in[t * HC + s]);
    uint base = t * HD + s * D;
    auto x_ = out + t * D;
    uint per = D / (32 * NG);
    float sq = 0.0f;
    for (uint i = lane * per; i < (lane + 1) * per; ++i) {
        float h = float(before[base + i]) + float(x_[i]) * back;
        sq += h * h;
    }
    sq = simd_sum(sq);
    if (lane % 32 == 0) { shared[lane / 32] = sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint k = 0; k < NG; ++k) { total += shared[k]; }
    float scale = metal::rsqrt(total / float(D) + EPS);
    for (uint i = lane * per; i < (lane + 1) * per; ++i) {
        float h = float(before[base + i]) + float(x_[i]) * back;
        residual[base + i] = static_cast<OutT>(h);
        normed[base + i] = h * scale * float(norm[s * D + i]);
    }
"""

# 32 * G threads to a row; no arrays local to a thread (with one of eight sums the kernel took six times as long)
_DOWN = """
    uint lane = thread_position_in_threadgroup.x;
    uint j = thread_position_in_grid.x / (32 * G);
    threadgroup float shared[G * T];
    auto row = j < R ? down + j * HD : inject_w + (j - R) * HD;
    uint per = HD / (32 * G);
    for (uint t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (uint i = lane * per; i < (lane + 1) * per; ++i) {
            acc += float(row[i]) * normed[t * HD + i];
        }
        acc = simd_sum(acc);
        if (lane % 32 == 0) { shared[(lane / 32) * T + t] = acc; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < T) {
        uint t = lane;
        float total = 0.0f;
        for (uint g = 0; g < G; ++g) { total += shared[g * T + t]; }
        total /= float(HC);
        float gate = 1.0f / (1.0f + metal::exp(-total));
        if (j < R) {
            low[t * R + j] = total * gate;
        } else {
            inject[t * HC + (j - R)] = static_cast<OutT>(2.0f * gate);
        }
    }
"""

# a column's HC rows of `up` in one group of threads, L of them to a row
_UP = """
    uint lane = thread_position_in_threadgroup.x;
    uint c = thread_position_in_grid.x / (HC * L);
    uint s = lane / L;
    uint part = lane % L;
    uint at = s * D + c;
    threadgroup float shared[HC * T];
    auto row = up + at * R;
    for (uint t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (uint r = part * (R / L); r < (part + 1) * (R / L); ++r) {
            acc += float(row[r]) * low[t * R + r];
        }
        float gate = acc;
        for (ushort k = 1; k < L; ++k) {
            gate += simd_shuffle_xor(acc, k);  // the other threads of this row
        }
        if (part == 0) { shared[s * T + t] = normed[t * HD + at] / (1.0f + metal::exp(-gate)); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < T) {
        uint t = lane;
        float mix = 0.0f;
        for (uint k = 0; k < HC; ++k) { mix += shared[k * T + t]; }
        mixed[t * D + c] = static_cast<OutT>(mix / float(HC));
    }
"""

NG = 8  # groups of 32 threads to a token's stream in the norm kernel

_kernels = {}


def kernels(rank, wide, d, hc, eps, tokens, groups, lanes):
    key = (rank, wide, d, hc, eps, tokens, groups, lanes)
    if key not in _kernels:
        header = (f"constant float EPS = {eps:.9g}f;\nconstant uint R = {rank};\nconstant uint HD = {wide};\n"
                  f"constant uint D = {d};\nconstant uint HC = {hc};\nconstant uint T = {tokens};\n"
                  f"constant uint G = {groups};\nconstant uint L = {lanes};\nconstant uint NG = {NG};\n")
        tag = f"{tokens}_{groups}_{lanes}"
        _kernels[key] = (
            mx.fast.metal_kernel(name=f"hcm_norm_{tag}", input_names=["before", "out", "inject_in", "norm"],
                                 output_names=["residual", "normed"], source=_NORM, header=header),
            mx.fast.metal_kernel(name=f"hcm_down_{tag}", input_names=["normed", "down", "inject_w"],
                                 output_names=["low", "inject"], source=_DOWN, header=header),
            mx.fast.metal_kernel(name=f"hcm_up_{tag}", input_names=["normed", "up", "low"],
                                 output_names=["mixed"], source=_UP, header=header))
    return _kernels[key]


class MultiResidual:
    def __init__(self, gr, groups: int = 32, lanes: int = 32, eps: float = 1e-6):
        self.norm = gr.hc_norm.weight
        self.down, self.up = gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight
        self.inject_w = gr.block_inject_weight.weight if gr.block_inject_weight is not None else None
        self.rank, self.wide = self.down.shape
        self.hc, self.eps, self.groups, self.lanes = gr.hc, eps, groups, lanes

    def __call__(self, before, out, inject_in):
        T, d, dtype = before.shape[1], out.shape[2], before.dtype
        G, L = self.groups, self.lanes
        norm_k, down_k, up_k = kernels(self.rank, self.wide, d, self.hc, self.eps, T, G, L)
        n_inject = self.hc if self.inject_w is not None else 0
        residual, normed = norm_k(
            inputs=[before.reshape(T, -1), out.reshape(T, d), inject_in.reshape(T, self.hc), self.norm],
            template=[("OutT", dtype)], grid=(32 * NG * self.hc * T, 1, 1), threadgroup=(32 * NG, 1, 1),
            output_shapes=[(T, self.wide), (T, self.wide)], output_dtypes=[dtype, mx.float32])
        low, inject = down_k(
            inputs=[normed, self.down, self.inject_w if n_inject else self.down], template=[("OutT", dtype)],
            grid=(32 * G * (self.rank + n_inject), 1, 1), threadgroup=(32 * G, 1, 1),
            output_shapes=[(T, self.rank), (T, self.hc)], output_dtypes=[mx.float32, dtype])
        (mixed,) = up_k(inputs=[normed, self.up, low], template=[("OutT", dtype)], grid=(self.hc * L * d, 1, 1),
                        threadgroup=(self.hc * L, 1, 1), output_shapes=[(T, d)], output_dtypes=[dtype])
        return mixed[None], residual[None], (inject[None] if n_inject else inject_in)


def check():
    """Against the model's own operations, on a small connection."""
    import numpy as np
    from mlx_lm.models import qwen4_exp
    mx.random.seed(5)
    gr = qwen4_exp.GatedResidual(qwen4_exp.TextArgs(hidden_size=256, hc_count=4, hc_lowrank=32))
    gr.hc_norm.weight = 1 + 0.2 * mx.random.normal(gr.hc_norm.weight.shape)
    worst = 0.0
    for T in (1, 3, 8):
        before, out = mx.random.normal((1, T, 4 * 256)), mx.random.normal((1, T, 256))
        inject_in = 2 * mx.random.uniform(shape=(1, T, 4))
        residual = before + (out[..., None, :] * inject_in[..., None]).reshape(1, T, -1)
        want = gr(residual)
        for groups, lanes in ((1, 8), (4, 32), (2, 16)):
            got = MultiResidual(gr, groups, lanes)(before, out, inject_in)
            for g, w in zip(got, (want[0], residual, want[2])):
                worst = max(worst, float(np.abs(np.array(g) - np.array(w)).max() / np.abs(np.array(w)).max()))
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--passes", type=int, default=60)
    p.add_argument("--windows", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--json")
    args = p.parse_args()
    model, _ = load(args.model)
    now = Engine(model)
    print(f"against the model's own operations, float32: differs by at most {check():.1e}")
    engines = {"as it is": now}
    for groups, lanes in ((32, 32), (32, 8)):
        e = Engine(model)
        e.hc = {key: (MultiResidual(pair[1].gr, groups, lanes),) * 2 for key, pair in e.hc.items()}
        engines[f"{32 * groups}/{lanes}"] = e
    for e in engines.values():
        e.warm()
        e.read(list(range(1000, 1200)))
    rows = []
    for T in args.windows:
        times = {name: [] for name in engines}
        for i in range(args.passes):
            ids = mx.array([[1000 + (i * 7 + j) % 500 for j in range(T)]])
            tops = []
            for name, e in engines.items():
                t0 = time.perf_counter()
                logits, w = e._forward(ids)
                mx.eval(logits)
                e._commit(w, T)
                times[name].append((time.perf_counter() - t0) * 1e3)
                tops.append(mx.argmax(logits[0], axis=-1).tolist())
        q = {name: sorted(ts)[len(ts) // 4] for name, ts in times.items()}
        rows.append({"tokens": T, **q})
        print(f"{T} tokens: " + "   ".join(f"{name} {ms:6.2f} ms ({ms - q['as it is']:+.2f})" for name, ms in q.items()),
              flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
