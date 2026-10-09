"""A Q6_K layer held as codes and scales must multiply as the weights gguf-py's dequantizer gives."""

import gguf
import mlx.core as mx
import numpy as np
import pytest

from strata_mlx.dense import MAX_TOKENS, ScaledLinear
from strata_mlx.gguf import Tensor, q6k_parts, to_float
from strata_mlx.load import linear

ROWS, COLS = 12, 1024


def q6k(seed: int = 0) -> Tensor:
    rng = np.random.default_rng(seed)
    n = ROWS * COLS // 256
    blocks = rng.integers(0, 256, (n, 210), dtype=np.uint8)
    blocks[:, 208:210] = rng.uniform(-0.01, 0.01, (n, 1)).astype("<f2").view(np.uint8)
    return Tensor("q6k", (ROWS, COLS), "Q6_K", blocks.reshape(-1))


def reference(t: Tensor) -> np.ndarray:
    return gguf.quants.dequantize(t.data.reshape(ROWS, -1), gguf.GGMLQuantizationType.Q6_K).reshape(t.shape)


def test_the_parts_are_the_weights():
    t = q6k()
    codes, scales, d = q6k_parts(t)
    assert codes.dtype == np.int8 and scales.dtype == np.int8 and d.dtype == np.float16
    assert codes.min() >= -32 and codes.max() <= 31
    w = codes.astype(np.float64) * np.repeat(scales, 16, 1) * np.repeat(d.astype(np.float64), 256, 1)
    want = reference(t)
    assert np.abs(w - want).max() <= 1e-6 * np.abs(want).max()
    assert np.abs(to_float(t) - want).max() <= 1e-6 * np.abs(want).max()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 7, 8, MAX_TOKENS + 4])
def test_a_window_multiplies_as_the_weights_do(tokens, dtype):
    t = q6k(tokens)
    lin = linear(t)
    assert isinstance(lin, ScaledLinear)
    x = mx.random.normal((1, tokens, COLS)).astype(dtype)
    got = np.array(lin(x).astype(mx.float32))
    want = np.array(x.astype(mx.float32)) @ reference(t).T
    assert got.shape == (1, tokens, ROWS)
    # float16: the output is rounded to it, and a chunk of a prompt multiplies weights rounded to it
    assert np.abs(got - want).max() <= (2e-5 if dtype == mx.float32 else 4e-3) * np.abs(want).max()


@pytest.mark.parametrize("tokens", [1, 4, MAX_TOKENS + 4])
def test_reordered_outputs_and_inputs(tokens):
    t = q6k(7)
    rng = np.random.default_rng(1)
    rows = rng.permutation(ROWS)
    cols = (rng.permutation(COLS // 128)[:, None] * 128 + np.arange(128)).reshape(-1)  # heads of 128 inputs change places
    lin = linear(t, rows=rows, cols=cols)
    assert lin.d_step == 128
    x = mx.random.normal((tokens, COLS))
    want = np.array(x) @ reference(t)[rows][:, cols].T
    got = np.array(lin(x))
    assert np.abs(got - want).max() <= 2e-5 * np.abs(want).max()
    with pytest.raises(ValueError, match="splits a run"):
        linear(t, cols=rng.permutation(COLS))


def test_a_shape_the_kernel_does_not_take_stays_floats():
    blocks = np.zeros((6 * 2, 210), dtype=np.uint8)
    lin = linear(Tensor("odd", (6, 512), "Q6_K", blocks.reshape(-1)))  # six rows: not groups of four
    assert not isinstance(lin, ScaledLinear) and lin.weight.shape == (6, 512)
