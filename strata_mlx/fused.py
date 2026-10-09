"""The hyper-connection between two blocks as two Metal kernels.

Every block reads the four residual streams through a hyper-connection and writes back into them: normalize each
stream, a low-rank gate over all of them, mix them into one vector for the block, and work out how strongly the
block's output goes back into each stream. As MLX operations that is about nine small kernels, twice a layer,
and a decoding pass costs by the number of its kernels more than by their work.

Here the write of the block before is folded into the read: the residual is `before + out * inject`, computed
on the fly by both kernels.

    kernel A   normalize, project down (and the inject logits)   a group of 32 threads per output
    kernel B   project up, gate, mix; also stores the residual   a group of 32 threads per column

What makes such a kernel fast or slow, measured on an M4 Max (a chain of them, each waiting for the one before):
a kernel that does nothing costs 7 us; the sizes must be constants in the source (read from the shape buffers
at run time, the same kernels took 64 and 101 us instead of 27 and 32); and no thread should loop long, so
every sum is split over the 32 threads of a group, each taking a contiguous share.

Sums are taken in float32 whatever the type of the weights.
"""

import mlx.core as mx

_A = """
    uint lane = thread_position_in_grid.x % 32;
    uint j = thread_position_in_grid.x / 32;
    uint t = thread_position_in_grid.y;
    auto row = j < R ? down + j * HD : inject_w + (j - R) * HD;
    auto h_ = before + t * HD;
    auto x_ = out + t * D;
    float total = 0.0f;
    for (uint s = 0; s < HC; ++s) {
        float sq = 0.0f, dot = 0.0f;
        float back = float(inject_in[t * HC + s]);
        for (uint i = lane * (D / 32); i < (lane + 1) * (D / 32); ++i) {
            uint at = s * D + i;
            float h = float(h_[at]) + float(x_[i]) * back;
            sq += h * h;
            dot += h * float(norm[at]) * float(row[at]);
        }
        sq = simd_sum(sq);
        dot = simd_sum(dot);
        float scale = metal::rsqrt(sq / float(D) + EPS);
        total += dot * scale;
        if (lane == 0 && j == 0) {
            inv_rms[t * HC + s] = scale;
        }
    }
    total /= float(HC);
    if (lane == 0) {
        float gate = 1.0f / (1.0f + metal::exp(-total));
        if (j < R) {
            low[t * R + j] = total * gate;
        } else {
            inject[t * HC + (j - R)] = static_cast<OutT>(2.0f * gate);
        }
    }
"""

# the 32 threads of a column: PER of them to each stream, each summing a PER-th of that stream's row of `up`
_B = """
    uint lane = thread_position_in_grid.x % 32;
    uint c = thread_position_in_grid.x / 32;
    uint t = thread_position_in_grid.y;
    uint s = lane / PER;
    uint part = lane % PER;
    uint at = s * D + c;
    float h = float(before[t * HD + at]) + float(out[t * D + c]) * float(inject_in[t * HC + s]);
    if (part == 0) {
        residual[t * HD + at] = static_cast<OutT>(h);
    }
    auto row = up + at * R;
    auto low_ = low + t * R;
    float partial = 0.0f;
    for (uint r = part * (R / PER); r < (part + 1) * (R / PER); ++r) {
        partial += float(row[r]) * low_[r];
    }
    float gate = partial;
    for (ushort k = 1; k < PER; ++k) {
        gate += simd_shuffle_xor(partial, k);  // the other threads of this stream
    }
    float mix = h * inv_rms[t * HC + s] * float(norm[at]) / (1.0f + metal::exp(-gate));
    mix = simd_sum(mix);
    if (lane == 0) {
        mixed[t * D + c] = static_cast<OutT>(mix / float(PER * HC));
    }
"""

_kernels = {}


def _kernel_pair(rank: int, wide: int, d: int, hc: int, eps: float):
    key = (rank, wide, d, hc, eps)
    if key not in _kernels:
        per = 32 // hc
        if 32 % hc or d % 32 or rank % per:
            raise ValueError(f"hyper-connection sizes not supported: {hc} streams of {d}, rank {rank}")
        header = (f"constant float EPS = {eps:.9g}f;\nconstant uint R = {rank};\nconstant uint HD = {wide};\n"
                  f"constant uint D = {d};\nconstant uint HC = {hc};\nconstant uint PER = {per};\n")
        _kernels[key] = (
            mx.fast.metal_kernel(
                name="hc_down", input_names=["before", "out", "inject_in", "norm", "down", "inject_w"],
                output_names=["low", "inject", "inv_rms"], source=_A, header=header),
            mx.fast.metal_kernel(
                name="hc_up", input_names=["before", "out", "inject_in", "norm", "up", "low", "inv_rms"],
                output_names=["mixed", "residual"], source=_B, header=header))
    return _kernels[key]


class FusedResidual:
    """A `GatedResidual` of the model, with the write that precedes it."""

    def __init__(self, gr, eps: float = 1e-6):
        self.norm = gr.hc_norm.weight
        self.down, self.up = gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight
        self.inject_w = gr.block_inject_weight.weight if gr.block_inject_weight is not None else None
        self.rank, self.wide = self.down.shape
        self.hc = gr.hc
        self.a, self.b = _kernel_pair(self.rank, self.wide, self.wide // self.hc, self.hc, eps)

    def __call__(self, before: mx.array, out: mx.array, inject_in: mx.array):
        """before (1, T, hc * d), out (1, T, d), inject_in (1, T, hc): the residual is before + out * inject_in.

        Returns (the block's input (1, T, d), the residual (1, T, hc * d), inject for the block's output (1, T, hc)).
        """
        T, d, dtype = before.shape[1], out.shape[2], before.dtype
        before, out, inject_in = before.reshape(T, -1), out.reshape(T, d), inject_in.reshape(T, self.hc)
        n_inject = self.hc if self.inject_w is not None else 0
        low, inject, inv_rms = self.a(
            inputs=[before, out, inject_in, self.norm, self.down, self.inject_w if n_inject else self.down],
            template=[("OutT", dtype)], grid=(32 * (self.rank + n_inject), T, 1), threadgroup=(32, 1, 1),
            output_shapes=[(T, self.rank), (T, self.hc), (T, self.hc)],
            output_dtypes=[mx.float32, dtype, mx.float32])
        mixed, residual = self.b(
            inputs=[before, out, inject_in, self.norm, self.up, low, inv_rms],
            template=[("OutT", dtype)], grid=(32 * d, T, 1), threadgroup=(32, 1, 1),
            output_shapes=[(T, d), (T, self.wide)], output_dtypes=[dtype, dtype])
        return mixed[None], residual[None], inject[None]


class PlainResidual:
    """The same call in the model's own operations: what a window of several tokens uses.

    The two kernels above are quicker for one token (46 against 72 us) and slower from about three tokens on
    (95 against 81 us at four): MLX's matrix products spread over a batch far better than these loops do.
    """

    def __init__(self, gr):
        self.gr = gr

    def __call__(self, before: mx.array, out: mx.array, inject_in: mx.array):
        residual = before + (out[..., None, :] * inject_in[..., None]).reshape(*out.shape[:-1], -1)
        got = self.gr(residual)
        return (got, residual, inject_in) if self.gr.block_inject_weight is None else (got[0], residual, got[2])
