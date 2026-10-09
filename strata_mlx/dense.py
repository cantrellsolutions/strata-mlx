"""Dense layers of a type MLX has no exact form for: Q6_K, as a byte a weight under the file's own scales.

MLX holds a quantized weight as `code * scale + bias` with one scale to 32 weights or more; Q6_K scales every
16. Unpacked to floats, as this loader did at first, the Q6_K layers of the IQ3_S file (its delta-net and
attention projections and its output layer) are 4.6 GB of float16 for 1.9 GB of file, read by every token, the
output layer again by every guess of the draft head. A weight is `code * scale * d`: a code of six bits, a scale
of eight for 16 weights, a float16 `d` for 256. Kept as those three, a signed byte to a code, it is the same
number in 8.56 bits.

A kernel on the file's blocks themselves reads 6.56 bits and was tried first (bench/experiments/e20, e21): a
block interleaves the codes' bits, and undoing that for every weight costs more than the bytes it spares once a
window has three tokens or more. In a pass of the IQ3_S model (e23), against float16, a byte a weight: 4.0 ms
less of 26.7 for one token, 3.2 of 42.9 for four, 1.4 of 54.5 for six, 1.7 more for eight.

The kernel is laid out as MLX's own quantized product: 32 threads share four rows, each takes eight weights of
each row at a time and goes along the rows; the activations come with a position's tokens side by side, so one
load brings a weight's activations for the whole window, once for the four rows. Sums are float32.
"""

import mlx.core as mx
import numpy as np


MAX_TOKENS = 8  # more than a window: a chunk of a prompt, multiplied as floats
_ROWS, _WEIGHTS = 4, 8  # to 32 threads; to a thread at a time
_VEC = {1: "float", 2: "float2", 3: "float3", 4: "float4"}
_PACKED = {1: "float", 2: "packed_float2", 3: "packed_float3", 4: "packed_float4"}


def _source(tokens: int) -> str:
    """x: (cols, tokens) float32, or for one token the activations as they are. For one token the source does
    not name the number of rows: a list of the output layer's rows (rows.py) changes it at every step."""
    chunks = [(o, min(4, tokens - o)) for o in range(0, tokens, 4)]  # (first token, how many) of each vector
    sums = "".join(f"{_VEC[n]} a{r}_{o} = 0.0f; " for r in range(_ROWS) for o, n in chunks)
    if tokens == 1:
        loads = "".join(f"float x{v}_0 = float(x[col + {v}]); " for v in range(_WEIGHTS))
    else:
        loads = "".join(f"{_VEC[n]} x{v}_{o} = {_VEC[n]}(*(const device {_PACKED[n]}*)(x + (col + {v}) * T + {o})); "
                        for v in range(_WEIGHTS) for o, n in chunks)
    rows = ""
    for r in range(_ROWS):
        rows += (f"        {{ uint row = {_ROWS} * group + {r}; const device int8_t* c = codes + row * COLS + col;\n"
                 f"          float s = float(d[row * (COLS / D_STEP) + col / D_STEP]) * float(scales[row * (COLS / 16) + col / 16]);\n")
        for o, n in chunks:
            rows += f"          a{r}_{o} += s * (" + " + ".join(f"float(c[{v}]) * x{v}_{o}" for v in range(_WEIGHTS)) + ");\n"
        rows += "        }\n"
    outs = ""
    for r in range(_ROWS):
        for o, n in chunks:
            at = f"{_ROWS} * group + {r}"
            writes = " ".join(f"out[{at if tokens == 1 else f'{o + j} * ROWS + {at}'}] = static_cast<OutT>({'s' if n == 1 else f's[{j}]'});"
                              for j in range(n))
            outs += f"    {{ {_VEC[n]} s = simd_sum(a{r}_{o}); if (lane == 0) {{ {writes} }} }}\n"
    return f"""
    uint lane = thread_position_in_grid.x % 32;
    uint group = thread_position_in_grid.x / 32;
    {sums}
    for (uint col = lane * {_WEIGHTS}; col < COLS; col += {32 * _WEIGHTS}) {{
        {loads}
{rows}    }}
{outs}"""


_kernels = {}


def fits(rows: int, cols: int) -> bool:
    """Whether the kernel takes a layer of this shape."""
    return rows % _ROWS == 0 and cols % (32 * _WEIGHTS) == 0


class ScaledLinear:
    """A linear layer held as codes, scales and d; called like `nn.Linear`."""

    def __init__(self, codes: mx.array, scales: mx.array, d: mx.array):
        self.codes, self.scales, self.d = codes, scales, d
        self.output_dims, self.input_dims = codes.shape
        self.d_step = self.input_dims // d.shape[1]  # weights to a d: 256, or fewer where the inputs were reordered
        if not fits(*codes.shape):
            raise ValueError(f"a layer of {codes.shape} does not fit the kernel")

    @property
    def nbytes(self) -> int:
        return self.codes.nbytes + self.scales.nbytes + self.d.nbytes

    def floats(self, dtype: mx.Dtype) -> mx.array:
        """The weights (out, in), each rounded once to `dtype`."""
        scale = mx.repeat(self.d.astype(mx.float32), self.d_step // 16, axis=1) * self.scales.astype(mx.float32)
        return (self.codes.astype(mx.float32) * mx.repeat(scale, 16, axis=1)).astype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        rows, cols = self.output_dims, self.input_dims
        lead, dtype = x.shape[:-1], x.dtype
        tokens = int(np.prod(lead))
        if tokens > MAX_TOKENS:
            return x @ self.floats(dtype).T
        x = x.reshape(tokens, cols)
        if tokens > 1:
            x = mx.contiguous(x.astype(mx.float32).T)
        key = (rows if tokens > 1 else 0, cols, tokens, self.d_step)
        if key not in _kernels:
            constants = {"COLS": cols, "T": tokens, "D_STEP": self.d_step} | ({"ROWS": rows} if tokens > 1 else {})
            header = "".join(f"constant uint {k} = {v};\n" for k, v in constants.items())
            _kernels[key] = mx.fast.metal_kernel(name=f"scaled_linear_{tokens}", input_names=["codes", "scales", "d", "x"],
                                                 output_names=["out"], source=_source(tokens), header=header)
        (out,) = _kernels[key](inputs=[self.codes, self.scales, self.d, x], template=[("OutT", dtype)],
                               grid=(32 * rows // _ROWS, 1, 1), threadgroup=(32, 1, 1),
                               output_shapes=[(tokens, rows)], output_dtypes=[dtype])
        return out.reshape(*lead, rows)
