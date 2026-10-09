"""Experts computed on the GGUF file's own blocks, for the types MLX has no exact form for.

MLX holds a quantized weight as `code * scale + bias` with one scale to 32 weights or more. The codebook types
of the larger files do not fit that: IQ2_S, IQ2_XS and IQ1_M scale every 16 weights, and IQ3_XXS and the 4-bit
types fit only at 8 bits, twice their size. Unpacked for MLX the experts of the IQ3_S file would take 94 GB and
twelve minutes of conversion at every start; as they are in the file they take 50 GB and none.

So the blocks go to the GPU as bytes, and two Metal kernels read them, each built from one small function per
type, `unpack32`, that turns a run of 32 weights into floats:

    matvec     out[pair, row] = W[expert of pair, row, :] . x     a few threads per row, for decoding
               (also with every pair's expert as an array of its own, for experts paged in from the SSD)
    dequant    every weight of the stack as float16 or float32    one thread per run of 32, for reading prompts

Q2_0, whose weights are a 2-bit code times a scale per 64, has two ways of its own. For answers, a table: two
weights make one of 16 sums of two inputs, so those sums are worked out once for the input and a row is then a
lookup for every two weights instead of a multiplication for every one (see `lookup_matvec`). For prompts,
Apple's matrix product for shaders (Metal Performance Primitives) takes 2-bit packed weights as an operand, so
the stack is only split into its codes and its scales and never unpacked (see `packed_matmul`).

The tables are ggml's, taken from gguf-py. Sums are float32 whatever the type of the activations.
"""

import mlx.core as mx
import numpy as np
from gguf import quants
from mlx_lm.models.activations import swiglu
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from .gguf import KVALUES_IQ4NL

# type -> (weights per block, bytes per block)
BLOCK = {"IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "IQ3_S": (256, 110), "IQ3_XXS": (256, 98),
         "IQ2_S": (256, 82), "IQ2_XS": (256, 74), "IQ2_XXS": (256, 66), "IQ1_M": (256, 56), "Q2_0": (64, 18)}

_COMMON = """
inline float half_at(const device uchar* b) {
    return float(as_type<half>(ushort(uint(b[0]) | (uint(b[1]) << 8))));
}
inline uint u16_at(const device uchar* b) { return uint(b[0]) | (uint(b[1]) << 8); }
inline uint u32_at(const device uchar* b) {
    return uint(b[0]) | (uint(b[1]) << 8) | (uint(b[2]) << 16) | (uint(b[3]) << 24);
}
"""

# each: the tables it needs, and unpack32(row, n, w): the n-th run of 32 weights of a row, scaled
_UNPACK = {
    "Q2_0": ((), """
    const device uchar* b = row + (n / 2) * 18;
    const device uchar* q = b + 2 + 8 * (n % 2);
    float d = half_at(b);
    for (uint i = 0; i < 8; ++i) {
        uint c = q[i];
        for (uint j = 0; j < 4; ++j) {
            w[4 * i + j] = d * (float((c >> (2 * j)) & 3) - 1.0f);
        }
    }
"""),
    "IQ4_NL": (("kvalues",), """
    const device uchar* b = row + n * 18;
    float d = half_at(b);
    for (uint i = 0; i < 16; ++i) {
        uchar q = b[2 + i];
        w[i] = d * float(kvalues[q & 15]);
        w[16 + i] = d * float(kvalues[q >> 4]);
    }
"""),
    "IQ4_XS": (("kvalues",), """
    const device uchar* b = row + (n / 8) * 136;
    uint ib = n % 8;
    uint ls = ((uint(b[4 + ib / 2]) >> (4 * (ib % 2))) & 15) | (((u16_at(b + 2) >> (2 * ib)) & 3) << 4);
    float d = half_at(b) * (float(ls) - 32.0f);
    const device uchar* q = b + 8 + 16 * ib;
    for (uint i = 0; i < 16; ++i) {
        w[i] = d * float(kvalues[q[i] & 15]);
        w[16 + i] = d * float(kvalues[q[i] >> 4]);
    }
"""),
    "IQ3_S": (("iq3s_grid",), """
    const device uchar* b = row + (n / 8) * 110;
    uint ib = n % 8;
    float d = half_at(b) * float(1 + 2 * ((uint(b[106 + ib / 2]) >> (4 * (ib % 2))) & 15));
    uint high = b[66 + ib];
    for (uint l = 0; l < 8; ++l) {
        constant uchar* g = iq3s_grid + 4 * (uint(b[2 + 8 * ib + l]) | (((high >> l) & 1) << 8));
        uint sign = uint(b[74 + 4 * ib + l / 2]) >> (4 * (l % 2));
        for (uint j = 0; j < 4; ++j) {
            float v = d * float(g[j]);
            w[4 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    "IQ3_XXS": (("iq3xxs_grid", "ksigns"), """
    const device uchar* b = row + (n / 8) * 98;
    uint ib = n % 8;
    uint aux = u32_at(b + 66 + 4 * ib);
    float d = half_at(b) * (0.5f + float(aux >> 28)) * 0.5f;
    for (uint l = 0; l < 8; ++l) {
        constant uchar* g = iq3xxs_grid + 4 * uint(b[2 + 8 * ib + l]);
        uint sign = uint(ksigns[(aux >> (7 * (l / 2))) & 127]) >> (4 * (l % 2));
        for (uint j = 0; j < 4; ++j) {
            float v = d * float(g[j]);
            w[4 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    "IQ2_S": (("iq2s_grid",), """
    const device uchar* b = row + (n / 8) * 82;
    uint ib = n % 8;
    float d = half_at(b) * 0.25f;
    uint high = b[66 + ib], scales = b[74 + ib];
    for (uint l = 0; l < 4; ++l) {
        constant uchar* g = iq2s_grid + 8 * (uint(b[2 + 4 * ib + l]) | (((high >> (2 * l)) & 3) << 8));
        uint sign = b[34 + 4 * ib + l];
        float dl = d * (0.5f + float(l < 2 ? scales & 15 : scales >> 4));
        for (uint j = 0; j < 8; ++j) {
            float v = dl * float(g[j]);
            w[8 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    "IQ2_XS": (("iq2xs_grid", "ksigns"), """
    const device uchar* b = row + (n / 8) * 74;
    uint ib = n % 8;
    float d = half_at(b) * 0.25f;
    uint scales = b[66 + ib];
    for (uint l = 0; l < 4; ++l) {
        uint q = u16_at(b + 2 + 2 * (4 * ib + l));
        constant uchar* g = iq2xs_grid + 8 * (q & 511);
        uint sign = ksigns[q >> 9];
        float dl = d * (0.5f + float(l < 2 ? scales & 15 : scales >> 4));
        for (uint j = 0; j < 8; ++j) {
            float v = dl * float(g[j]);
            w[8 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    "IQ2_XXS": (("iq2xxs_grid", "ksigns"), """
    const device uchar* b = row + (n / 8) * 66;
    const device uchar* q = b + 2 + 8 * (n % 8);
    uint aux = u32_at(q + 4);
    float d = half_at(b) * (0.5f + float(aux >> 28)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
        constant uchar* g = iq2xxs_grid + 8 * uint(q[l]);
        uint sign = ksigns[(aux >> (7 * l)) & 127];
        for (uint j = 0; j < 8; ++j) {
            float v = d * float(g[j]);
            w[8 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    # the block's 16-bit scale is spread over the top four bits of its four scale words
    "IQ1_M": (("iq1_grid",), """
    const device uchar* b = row + (n / 8) * 56;
    uint ib = n % 8;
    uint packed = (u16_at(b + 48) >> 12) | ((u16_at(b + 50) >> 8) & 0x00f0) | ((u16_at(b + 52) >> 4) & 0x0f00)
        | (u16_at(b + 54) & 0xf000);
    float d = float(as_type<half>(ushort(packed)));
    uint scales = u16_at(b + 48 + 2 * (ib / 2)) >> (6 * (ib % 2));
    for (uint l = 0; l < 4; ++l) {
        uint high = (uint(b[32 + 2 * ib + l / 2]) >> (4 * (l % 2))) & 15;
        constant uchar* g = iq1_grid + 8 * (uint(b[4 * ib + l]) | ((high & 7) << 8));
        float dl = d * float(2 * ((scales >> (3 * (l / 2))) & 7) + 1);
        float delta = high & 8 ? -1.125f : -0.875f;  // the table holds the grid's -1, 0, 1 as 0, 1, 2
        for (uint j = 0; j < 8; ++j) {
            w[8 * l + j] = dl * (float(g[j]) + delta);
        }
    }
"""),
}

_MATVEC = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    auto x_ = x + (pair / SHARE) * COLS;
    float w[32];
    float sum = 0.0f;
    for (uint n = part; n < RUNS; n += P) {
        unpack32(row, n, w);
        auto xr = x_ + 32 * n;
        float s = 0.0f;
        for (uint j = 0; j < 32; ++j) {
            s += w[j] * float(xr[j]);
        }
        sum += s;
    }
    float total = sum;
    for (ushort k = 1; k < P; ++k) {
        total += simd_shuffle_xor(sum, k);  // the other threads of this row
    }
    if (part == 0) {
        out[pair * ROWS + r] = static_cast<OutT>(total);
    }
"""

_DEQUANT = """
    uint n = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y + ROWS * thread_position_in_grid.z;
    float w[32];
    unpack32(blocks + r * ROW_BYTES, n, w);
    auto o = out + r * COLS + 32 * n;
    for (uint j = 0; j < 32; ++j) {
        o[j] = static_cast<OutT>(w[j]);
    }
"""

_tables = {}


def _table(name: str) -> str:
    """A table of ggml's as Metal source."""
    if name not in _tables:
        if name == "kvalues":
            kind, values = "char", KVALUES_IQ4NL
        elif name == "ksigns":
            kind, values = "uchar", np.frombuffer(quants.IQ2_XXS.ksigns, dtype=np.uint8)
        else:
            cls = {"iq3s_grid": quants.IQ3_S, "iq3xxs_grid": quants.IQ3_XXS, "iq2s_grid": quants.IQ2_S,
                   "iq2xs_grid": quants.IQ2_XS, "iq2xxs_grid": quants.IQ2_XXS, "iq1_grid": quants.IQ1_M}[name]
            cls.init_grid()
            kind, values = "uchar", cls.grid.reshape(-1) + (1 if name == "iq1_grid" else 0)
        values = [int(v) for v in values]
        _tables[name] = f"constant {kind} {name}[{len(values)}] = {{{','.join(map(str, values))}}};\n"
    return _tables[name]


def _header(kind: str, rows: int, cols: int, **constants) -> str:
    per_block, block_bytes = BLOCK[kind]
    if cols % per_block:
        raise ValueError(f"{kind}: rows of {cols} weights do not end on a block of {per_block}")
    tables, body = _UNPACK[kind]
    constants |= {"ROWS": rows, "COLS": cols, "RUNS": cols // 32, "ROW_BYTES": cols // per_block * block_bytes}
    return ("".join(f"constant uint {k} = {v};\n" for k, v in constants.items()) + _COMMON
            + "".join(_table(t) for t in tables)
            + "inline void unpack32(const device uchar* row, uint n, thread float* w) {" + body + "}\n")


def _threads_per_row(runs: int) -> int:
    """As many as divide a row evenly and leave each at least four runs: a thread should not loop long."""
    p = 1
    while p < 32 and runs % (2 * p) == 0 and runs // (2 * p) >= 4:
        p *= 2
    return p


_kernels = {}


def _kernel(what: str, kind: str, rows: int, cols: int, share: int = 1, per_row: int = 1):
    key = (what, kind, rows, cols, share, per_row)
    if key not in _kernels:
        if what == "matvec":
            _kernels[key] = mx.fast.metal_kernel(
                name=f"matvec_{kind.lower()}", input_names=["blocks", "idx", "x"], output_names=["out"],
                source=_MATVEC, header=_header(kind, rows, cols, P=per_row, SHARE=share))
        else:
            _kernels[key] = mx.fast.metal_kernel(
                name=f"dequant_{kind.lower()}", input_names=["blocks"], output_names=["out"],
                source=_DEQUANT, header=_header(kind, rows, cols))
    return _kernels[key]


MAX_SEPARATE = 29  # Metal binds 31 buffers to a kernel: these, the activations and the output


def matvec_separate(experts: list, kind: str, rows: int, cols: int, x: mx.array, share: int) -> mx.array:
    """out[pair] = W[pair] . x, where every pair brings its expert's blocks as an array of its own, so that
    nothing has to be joined first. x (pairs / share, cols): one row to every `share` pairs. Returns (pairs, rows)."""
    n, per_row = len(experts), _threads_per_row(cols // 32)
    key = ("separate", kind, rows, cols, share, per_row, n)
    if key not in _kernels:
        pick = " ".join(f"pair == {k} ? b{k} :" for k in range(n - 1)) + f" b{n - 1}"
        source = _MATVEC.replace("blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES", f"({pick}) + r * ROW_BYTES")
        assert source != _MATVEC
        _kernels[key] = mx.fast.metal_kernel(
            name=f"matvec_{kind.lower()}_{n}", input_names=[f"b{k}" for k in range(n)] + ["x"], output_names=["out"],
            source=source, header=_header(kind, rows, cols, P=per_row, SHARE=share))
    (out,) = _kernels[key](
        inputs=[*experts, x], template=[("OutT", x.dtype)], grid=(per_row * rows, n, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n, rows)], output_dtypes=[x.dtype])
    return out


def dequantize(blocks: mx.array, kind: str, shape: tuple, dtype: mx.Dtype = mx.float32) -> mx.array:
    """The weights of `blocks` (uint8, a tensor of `shape` as the file holds it) as floats."""
    *lead, rows, cols = shape
    n = int(np.prod(lead)) if lead else 1
    (out,) = _kernel("dequant", kind, rows, cols)(
        inputs=[blocks], template=[("OutT", dtype)], grid=(cols // 32, rows, n), threadgroup=(32, 1, 1),
        output_shapes=[tuple(shape)], output_dtypes=[dtype])
    return out


class BlockSwitchLinear:
    """A stack of linear layers kept as the file's blocks; called like mlx-lm's `QuantizedSwitchLinear`."""

    # From this many (token, expert) pairs on, the stack is unpacked once and multiplied as floats. Measured on an
    # M4 Max, a gate tensor: 256 tokens 4.5 ms a pair at a time against 10 unpacked, 1,024 tokens 18 against 11.5.
    unpack_from = 5000

    def __init__(self, blocks: mx.array, kind: str, shape: tuple, per_row: int | None = None):
        self.blocks, self.kind = blocks, kind
        self.num_experts, self.output_dims, self.input_dims = shape
        self.per_row = per_row or _threads_per_row(self.input_dims // 32)

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """x (..., 1, in), one row to each index or to each run of the last axis of `indices`."""
        batch, k = x.shape[:-2], indices.shape[-1] if indices.ndim else 1
        if batch == indices.shape:
            share = 1
        elif batch == (*indices.shape[:-1], 1):
            share = k
        else:
            x, share = mx.broadcast_to(x, (*indices.shape, 1, self.input_dims)), 1
        pairs = indices.size
        if pairs >= self.unpack_from:
            return self._as_floats(x, indices, sorted_indices)
        (out,) = _kernel("matvec", self.kind, self.output_dims, self.input_dims, share, self.per_row)(
            inputs=[self.blocks, indices.astype(mx.uint32).reshape(-1), x.reshape(-1, self.input_dims)],
            template=[("OutT", x.dtype)], grid=(self.per_row * self.output_dims, pairs, 1),
            threadgroup=(32, 1, 1), output_shapes=[(pairs, self.output_dims)], output_dtypes=[x.dtype])
        return out.reshape(*indices.shape, 1, self.output_dims)

    def _as_floats(self, x: mx.array, indices: mx.array, sorted_indices: bool) -> mx.array:
        w = dequantize(self.blocks, self.kind, (self.num_experts, self.output_dims, self.input_dims), x.dtype)
        return mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices)


# ---- Q2_0 for prompts: Apple's matrix product over the 2-bit codes

# Measured on an M4 Max, a gate tensor (512 experts, 2560 -> 640), tokens spread evenly over the experts, against
# unpacking the stack to float16 (5.3 ms) and MLX's gather_mm over it: 1,600 tokens of prompt 7.4 ms against
# 11.8, 3,300 tokens 13.7 against 16.4, 6,600 tokens the same 25. The product itself runs at 10.5 T operations
# a second whatever the group sizes, in tiles of 16 tokens by 64 outputs on two SIMD groups; tiles of 64 tokens
# on four ran at 4.6, and the stack is 0.24 GB on the way instead of 1.7.
PACKED_FROM = 3000  # (token, expert) pairs from which a prompt goes this way
TOKENS_TILE, ROWS_TILE, _SIMD_GROUPS = 16, 64, 2

_SPLIT = """
    uint blk = thread_position_in_grid.x, row = thread_position_in_grid.y;
    const device uchar* src = blocks + (row * NBLK + blk) * 18;
    scales[row * NBLK + blk] = as_type<half>(ushort(uint(src[0]) | (uint(src[1]) << 8)));
    device uchar* dst = codes + (row * NBLK + blk) * 16;
    for (uint i = 0; i < 16; ++i) {
        dst[i] = src[2 + i];
    }
"""

# One tile: TM token rows of `a`, all of one expert, by TN of its output rows. The scale differs from one 64 of
# weights to the next, so the product is taken 64 at a time into registers and each part scaled as it is added:
# w = d * (code - 1), hence d * (codes . a - sum of a).
_PACKED = """
    uint tx = threadgroup_position_in_grid.x, ty = threadgroup_position_in_grid.y;
    uint row0 = uint(tile_expert[ty]) * ROWS + tx * TN;
    constexpr auto desc = matmul2d_descriptor(TM, TN, 64, false, true, false);
    matmul2d<desc, execution_simdgroups<SIMD_GROUPS>> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A((device half *)a, dextents<int32_t, 2>(COLS, int(a_shape[0])));
    tensor<device uint2b_format, dextents<int32_t, 2>, tensor_inline> B(
        (device uchar *)codes, dextents<int32_t, 2>(COLS, int(codes_shape[0])));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(ROWS, int(a_shape[0])));
    auto tA = A.slice<64, TM>(0, ty * TM);
    auto tB = B.slice<64, TN>(0, row0);
    auto acc = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0;
        }
    }
    for (uint blk = 0; blk < NBLK; ++blk) {
        tA = A.slice<64, TM>(blk * 64, ty * TM);
        tB = B.slice<64, TN>(blk * 64, row0);
        auto part = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
        for (uint16_t i = 0; i < part.get_capacity(); ++i) {
            if (part.is_valid_element(i)) {
                part[i] = 0;
            }
        }
        op.run(tA, tB, part);
        for (uint16_t i = 0; i < part.get_capacity(); ++i) {
            if (part.is_valid_element(i)) {
                auto at = part.get_multidimensional_index(i);
                acc[i] += float(scales[(row0 + at[0]) * NBLK + blk]) * (part[i] - sums[(ty * TM + at[1]) * NBLK + blk]);
            }
        }
    }
    auto tC = C.slice<TN, TM>(tx * TN, ty * TM);
    acc.store(tC);
"""


def packable(kind: str, rows: int, cols: int, dtype: mx.Dtype) -> bool:
    """Whether `packed_matmul` takes such a stack: the primitive multiplies half by 2-bit, not float by 2-bit."""
    return kind == "Q2_0" and dtype == mx.float16 and rows % ROWS_TILE == 0 and cols % 64 == 0


def packed_matmul(blocks: mx.array, shape: tuple, a: mx.array, tile_expert: mx.array) -> mx.array:
    """out[p] = W[expert of p's tile] . a[p] for a Q2_0 stack `blocks` of `shape` (experts, rows, cols).

    a (P, cols) float16, P a multiple of TOKENS_TILE, every TOKENS_TILE rows for one expert; tile_expert (P /
    TOKENS_TILE,) says which. Returns (P, rows) float32.
    """
    experts, rows, cols = shape
    nblk, p = cols // 64, a.shape[0]
    key = ("packed", rows, cols)
    if key not in _kernels:
        constants = {"ROWS": rows, "COLS": cols, "NBLK": nblk, "TM": TOKENS_TILE, "TN": ROWS_TILE, "SIMD_GROUPS": _SIMD_GROUPS}
        header = ("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"
                  + "".join(f"constant uint {k} = {v};\n" for k, v in constants.items()))
        _kernels[key] = (
            mx.fast.metal_kernel(name="q2_split", input_names=["blocks"], output_names=["codes", "scales"],
                                 source=_SPLIT, header=f"constant uint NBLK = {nblk};\n"),
            mx.fast.metal_kernel(name="q2_packed", input_names=["a", "sums", "codes", "scales", "tile_expert"],
                                 output_names=["c"], source=_PACKED, header=header))
    split, product = _kernels[key]
    codes, scales = split(inputs=[blocks], grid=(nblk, experts * rows, 1), threadgroup=(min(nblk, 32), 1, 1),
                          output_shapes=[(experts * rows, cols // 4), (experts * rows, nblk)],
                          output_dtypes=[mx.uint8, mx.float16])
    sums = a.reshape(p, nblk, 64).astype(mx.float32).sum(axis=-1)
    width = 32 * _SIMD_GROUPS
    (out,) = product(inputs=[a, sums, codes, scales, tile_expert], grid=(width * (rows // ROWS_TILE), p // TOKENS_TILE, 1),
                     threadgroup=(width, 1, 1), output_shapes=[(p, rows)], output_dtypes=[mx.float32])
    return out


def _packed_glu(x: mx.array, indices: mx.array, up, gate, down) -> mx.array:
    """The three products of a prompt's pairs through `packed_matmul`: pairs sorted by expert, each expert's run
    filled up to whole tiles (what fills it is computed and thrown away)."""
    idx = np.array(indices)  # the tiles depend on what the router chose: the pass waits for it here
    k, flat = idx.shape[-1], idx.reshape(-1)
    order = np.argsort(flat, kind="stable")
    used, counts = np.unique(flat, return_counts=True)
    padded = -(-counts // TOKENS_TILE) * TOKENS_TILE
    slot = np.repeat(np.cumsum(padded) - padded, counts) + np.arange(len(flat)) - np.repeat(np.cumsum(counts) - counts, counts)
    token = np.zeros(int(padded.sum()), dtype=np.uint32)
    token[slot] = order // k  # the token each slot stands for, in order of expert
    where = np.empty(len(flat), dtype=np.uint32)
    where[order] = slot  # and the slot of each pair
    tiles = mx.array(np.repeat(used, padded // TOKENS_TILE).astype(np.uint32))
    a = x.reshape(-1, x.shape[-1])[mx.array(token)]

    def product(lin, a):
        return packed_matmul(lin.blocks, (lin.num_experts, lin.output_dims, lin.input_dims), a, tiles)

    h = swiglu(product(gate, a), product(up, a)).astype(x.dtype)
    out = product(down, h)[mx.array(where)].astype(x.dtype)
    return out.reshape(*idx.shape, down.output_dims)


# ---- Q2_0 for answers: a table of sums instead of a multiplication for every weight
#
# The matvec kernel above is bound by arithmetic: about eight operations a weight, at all the operations this
# GPU has (bench/experiments/e02, e03, e10, e11: neither the inner loop's shape, nor the threads to a row, nor
# loading four values at once, nor sharing the activations between rows changed its time; leaving the
# arithmetic out did). Two Q2_0 weights are each -1, 0, 1 or 2 times their scale: of two inputs they make one
# of 16 sums. With those sums in a table, a row costs about three operations a weight. A table of the 256 sums
# of four weights would halve that again and was four times slower: 650 KB for a token do not stay in a cache,
# 41 KB do, and only as float16. Hence float16 activations only; a sum then carries about one more rounding of
# a float16 than before.
#
# In a pass, against the matvec kernel (e12): 0.4 ms less for one token, 1.2 for two, 3.4 for four, 9.4 for
# eight. Few pairs want many threads to a row, many pairs few.

LOOKUP_FROM = 1  # (token, expert) pairs from which Q2_0 experts over float16 go this way

_TABLE = """
    uint g = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    float a = float(x[n * COLS + 2 * g]), b = float(x[n * COLS + 2 * g + 1]);
    auto t = table + (n * (COLS / 2) + g) * 16;
    for (uint c = 0; c < 16; ++c) {
        t[c] = static_cast<OutT>((float(c & 3) - 1.0f) * a + (float(c >> 2) - 1.0f) * b);
    }
"""

_LOOKUP = """
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
        total += simd_shuffle_xor(sum, k);  // the other threads of this row
    }
    if (part == 0) {
        out[pair * ROWS + r] = static_cast<OutT>(total);
    }
"""


def lookable(lin, dtype: mx.Dtype) -> bool:
    return lin.kind == "Q2_0" and dtype == mx.float16


def lookup_table(x: mx.array) -> mx.array:
    """x (n, cols) float16 -> (n, cols / 2, 16): for every two inputs, the sums two weights can make of them."""
    n, cols = x.shape
    key = ("table", cols)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(name="q2_table", input_names=["x"], output_names=["table"],
                                             source=_TABLE, header=f"constant uint COLS = {cols};\n")
    (table,) = _kernels[key](inputs=[x], template=[("OutT", x.dtype)], grid=(cols // 2, n, 1), threadgroup=(32, 1, 1),
                             output_shapes=[(n, cols // 2, 16)], output_dtypes=[x.dtype])
    return table


def lookup_matvec(lin, idx: mx.array, table: mx.array, share: int) -> mx.array:
    """out[pair] = W[idx[pair]] . x[pair / share], x given as its `lookup_table`. Returns (pairs, rows)."""
    pairs = idx.shape[0]
    per_row = _threads_per_row(lin.input_dims // 32)
    if pairs > 20:
        per_row = min(per_row, 4)
    key = ("lookup", lin.output_dims, lin.input_dims, share, per_row)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name="q2_lookup", input_names=["blocks", "idx", "table"], output_names=["out"], source=_LOOKUP,
            header=_header("Q2_0", lin.output_dims, lin.input_dims, P=per_row, SHARE=share))
    (out,) = _kernels[key](inputs=[lin.blocks, idx, table], template=[("OutT", table.dtype)],
                           grid=(per_row * lin.output_dims, pairs, 1), threadgroup=(32, 1, 1),
                           output_shapes=[(pairs, lin.output_dims)], output_dtypes=[table.dtype])
    return out


def _lookup_glu(x: mx.array, indices: mx.array, up, gate, down) -> mx.array:
    k = indices.shape[-1]
    idx = indices.reshape(-1).astype(mx.uint32)
    table = lookup_table(x.reshape(-1, x.shape[-1]))  # one for gate and up: they read the same input
    h = swiglu(lookup_matvec(gate, idx, table, k), lookup_matvec(up, idx, table, k))
    return lookup_matvec(down, idx, lookup_table(h), 1).reshape(*indices.shape, down.output_dims)


def switch_glu(x: mx.array, indices: mx.array, up, gate, down) -> mx.array:
    """down(silu(gate(x)) * up(x)) through each token's experts: x (..., in), indices (..., experts per token)."""
    if indices.size >= PACKED_FROM and all(
            packable(lin.kind, lin.output_dims, lin.input_dims, x.dtype) for lin in (up, gate, down)):
        return _packed_glu(x, indices, up, gate, down)
    if indices.size >= LOOKUP_FROM and all(lookable(lin, x.dtype) for lin in (up, gate, down)):
        return _lookup_glu(x, indices, up, gate, down)
    # as mlx-lm's SwitchGLU
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    shape, inv_order = indices.shape, None
    if do_sort:
        x, indices, inv_order = _gather_sort(x, indices)
    x = down(swiglu(gate(x, indices, sorted_indices=do_sort), up(x, indices, sorted_indices=do_sort)), indices,
             sorted_indices=do_sort)
    if do_sort:
        x = _scatter_unsort(x, inv_order, shape)
    return x.squeeze(-2)


class BlockSwitchGLU:
    """A layer's routed experts, called like mlx-lm's `SwitchGLU`."""

    def __init__(self, gate: BlockSwitchLinear, up: BlockSwitchLinear, down: BlockSwitchLinear):
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return switch_glu(x, indices, self.up_proj, self.gate_proj, self.down_proj)
