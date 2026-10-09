"""The hyper-connection between two blocks as three Metal kernels that read its weights once for a whole window.

Every block reads the four residual streams through a hyper-connection and writes back into them: normalize each
stream, a low-rank gate over all of them, mix them into one vector for the block, and work out how strongly the
block's output goes back into each stream. The write of the block before is folded into the read: the residual
is `before + out * inject`.

    norm   the residual and its normalized form             256 threads to a token's stream
    down   project down (and the inject logits)             1,024 threads to an output, every token in each
    up     project up, gate, mix                            32 threads to a row of `up`, every token in each

What makes such a kernel fast or slow, measured on an M4 Max (bench/experiments/e04, e08, e09):

  * one thread is slow, about 13 ns an instruction, and a kernel takes as long as its longest thread: a thread
    should do tens of instructions, not hundreds. The projection down with 32 threads to a row (320 weights
    each) took 38 us for four tokens, with 1,024 (10 weights each) 21; a norm kernel of 128 threads that each
    looped 160 times took 21 us for a few kilobytes;
  * threads that share a sum meet through threadgroup memory and a barrier, 32 at a time through `simd_sum`;
  * no arrays local to a thread: with an array of eight sums the same kernel took six times as long;
  * the sizes are constants in the source, the number of tokens among them (a kernel is built for each);
  * a weight is read once and used for every token of the window. Kernels that ran once per token read the
    weights again for each and lost to MLX's own operations from three tokens on.

A chain of 97 connections, microseconds each, for 1 / 2 / 4 / 8 tokens: these kernels 37 / 39 / 55 / 103,
MLX's own operations 62 / 66 / 72 / 77. In a pass that is 0.3, 1.6, 1.7 ms less for 1, 2, 4 tokens, 6 and 5 ms
for 3 and 5 (odd windows were slow in MLX's products), 2.4 ms more for 8: the engine uses these up to six.

Each token's sums are taken in the same order whatever the window, in float32 whatever the type of the weights.
"""

import mlx.core as mx

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
    if (lane == 0) {
        for (uint t = 0; t < T; ++t) {
            float mix = 0.0f;
            for (uint k = 0; k < HC; ++k) { mix += shared[k * T + t]; }
            mixed[t * D + c] = static_cast<OutT>(mix / float(HC));
        }
    }
"""

_kernels = {}


def _split(n: int, most: int) -> int:
    """The largest power of two, at most `most`, that divides n."""
    g = 1
    while 2 * g <= most and n % (2 * g) == 0:
        g *= 2
    return g


def _kernel_set(rank: int, wide: int, d: int, hc: int, eps: float, tokens: int):
    key = (rank, wide, d, hc, eps, tokens)
    if key not in _kernels:
        if 32 % hc or d % 32:
            raise ValueError(f"hyper-connection sizes not supported: {hc} streams of {d}, rank {rank}")
        ng, g, lanes = _split(d // 32, 8), _split(wide // 32, 32), _split(rank, 32)
        header = (f"constant float EPS = {eps:.9g}f;\nconstant uint R = {rank};\nconstant uint HD = {wide};\n"
                  f"constant uint D = {d};\nconstant uint HC = {hc};\nconstant uint T = {tokens};\n"
                  f"constant uint NG = {ng};\nconstant uint G = {g};\nconstant uint L = {lanes};\n")
        _kernels[key] = (
            mx.fast.metal_kernel(name=f"hc_norm_{tokens}", input_names=["before", "out", "inject_in", "norm"],
                                 output_names=["residual", "normed"], source=_NORM, header=header),
            mx.fast.metal_kernel(name=f"hc_down_{tokens}", input_names=["normed", "down", "inject_w"],
                                 output_names=["low", "inject"], source=_DOWN, header=header),
            mx.fast.metal_kernel(name=f"hc_up_{tokens}", input_names=["normed", "up", "low"],
                                 output_names=["mixed"], source=_UP, header=header),
            (ng, g, lanes))
    return _kernels[key]


class FusedResidual:
    """A `GatedResidual` of the model, with the write that precedes it."""

    def __init__(self, gr, eps: float = 1e-6):
        self.norm = gr.hc_norm.weight
        self.down, self.up = gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight
        self.inject_w = gr.block_inject_weight.weight if gr.block_inject_weight is not None else None
        self.rank, self.wide = self.down.shape
        self.hc, self.eps = gr.hc, eps

    def __call__(self, before: mx.array, out: mx.array, inject_in: mx.array):
        """before (1, T, hc * d), out (1, T, d), inject_in (1, T, hc): the residual is before + out * inject_in.

        Returns (the block's input (1, T, d), the residual (1, T, hc * d), inject for the block's output (1, T, hc)).
        """
        T, d, dtype = before.shape[1], out.shape[2], before.dtype
        norm, down, up, (ng, g, lanes) = _kernel_set(self.rank, self.wide, d, self.hc, self.eps, T)
        n_inject = self.hc if self.inject_w is not None else 0
        residual, normed = norm(
            inputs=[before.reshape(T, -1), out.reshape(T, d), inject_in.reshape(T, self.hc), self.norm],
            template=[("OutT", dtype)], grid=(32 * ng * self.hc * T, 1, 1), threadgroup=(32 * ng, 1, 1),
            output_shapes=[(T, self.wide), (T, self.wide)], output_dtypes=[dtype, mx.float32])
        low, inject = down(
            inputs=[normed, self.down, self.inject_w if n_inject else self.down], template=[("OutT", dtype)],
            grid=(32 * g * (self.rank + n_inject), 1, 1), threadgroup=(32 * g, 1, 1),
            output_shapes=[(T, self.rank), (T, self.hc)], output_dtypes=[mx.float32, dtype])
        (mixed,) = up(inputs=[normed, self.up, low], template=[("OutT", dtype)], grid=(self.hc * lanes * d, 1, 1),
                      threadgroup=(self.hc * lanes, 1, 1), output_shapes=[(T, d)], output_dtypes=[dtype])
        return mixed[None], residual[None], (inject[None] if n_inject else inject_in)


# ---- a float16 matrix times a window, summed in float32
#
# A layer's router is such a product (512 by 2,560), wanted in float32 because the ten largest of its results
# choose the experts. The port casts the input to float32 and lets MLX's matmul promote the weights: 2.6 MB
# converted and written again for every pass. Read as they are, with the sums in float32, a chain of the 48
# routers' products takes 16, 16, 18 and 25 us each for 1, 2, 4 and 8 tokens against 31 to 37. Each token's sum
# is taken in the same order whatever the window.

_MATVEC = """
    uint lane = thread_position_in_threadgroup.x;
    uint j = thread_position_in_grid.x / (32 * G);
    threadgroup float shared[G * T];
    auto row = w + j * COLS;
    uint per = COLS / (32 * G);
    for (uint t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (uint i = lane * per; i < (lane + 1) * per; ++i) {
            acc += float(row[i]) * float(x[t * COLS + i]);
        }
        acc = simd_sum(acc);
        if (lane % 32 == 0) { shared[(lane / 32) * T + t] = acc; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < T) {
        float total = 0.0f;
        for (uint g = 0; g < G; ++g) { total += shared[g * T + lane]; }
        out[lane * ROWS + j] = total;
    }
"""


class WindowLinear:
    """A float16 `nn.Linear` without bias whose products come back in float32; called with a window of a few
    tokens it reads its weights once, as they are. Longer inputs go through MLX's matmul as before."""

    upto, groups = 8, 16

    def __init__(self, linear):
        self.linear = linear  # its weight is read at every call: the model's type is set after it is built
        self.rows, self.cols = linear.weight.shape
        self.g = _split(self.cols // 32, self.groups)

    @property
    def weight(self) -> mx.array:
        return self.linear.weight

    def __call__(self, x: mx.array) -> mx.array:
        tokens = x.size // self.cols
        if tokens > self.upto or self.weight.dtype != mx.float16:
            return self.linear(x)
        key = ("matvec", self.rows, self.cols, tokens, self.g)
        if key not in _kernels:
            header = (f"constant uint ROWS = {self.rows};\nconstant uint COLS = {self.cols};\n"
                      f"constant uint T = {tokens};\nconstant uint G = {self.g};\n")
            _kernels[key] = mx.fast.metal_kernel(name=f"window_matvec_{tokens}", input_names=["w", "x"],
                                                 output_names=["out"], source=_MATVEC, header=header)
        (out,) = _kernels[key](inputs=[self.weight, x.reshape(tokens, self.cols)], grid=(32 * self.g * self.rows, 1, 1),
                               threadgroup=(32 * self.g, 1, 1), output_shapes=[(tokens, self.rows)],
                               output_dtypes=[mx.float32])
        return out.reshape(*x.shape[:-1], self.rows)


def install_routers(layers) -> None:
    """Make the routers of `layers` products of this kind."""
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and isinstance(getattr(mlp.gate, "weight", None), mx.array) and not isinstance(mlp.gate, WindowLinear):
            mlp.gate = WindowLinear(mlp.gate)


class PlainResidual:
    """The same call in the model's own operations: what a prompt uses, and a window of more than six tokens."""

    def __init__(self, gr):
        self.gr = gr

    def __call__(self, before: mx.array, out: mx.array, inject_in: mx.array):
        residual = before + (out[..., None, :] * inject_in[..., None]).reshape(*out.shape[:-1], -1)
        got = self.gr(residual)
        return (got, residual, inject_in) if self.gr.block_inject_weight is None else (got[0], residual, got[2])
