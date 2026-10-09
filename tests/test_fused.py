"""The fused hyper-connection kernels against the model's own operations."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import qwen4_exp

from strata_mlx.fused import FusedResidual, WindowLinear

ARGS = qwen4_exp.TextArgs(hidden_size=96, hc_count=4, hc_lowrank=24)


def reference(gr, before, out, inject_in):
    residual = before + (out[..., None, :] * inject_in[..., None]).reshape(*out.shape[:-1], -1)
    got = gr(residual)
    return (got, residual, None) if gr.block_inject_weight is None else (got[0], residual, got[2])


@pytest.mark.parametrize("dtype, tol", [(mx.float32, 2e-5), (mx.float16, 4e-3)])
@pytest.mark.parametrize("tokens", [1, 2, 5, 8])
@pytest.mark.parametrize("combine", [True, False])
def test_the_kernels_compute_the_read_and_the_write_before_it(dtype, tol, tokens, combine):
    mx.random.seed(5)
    gr = qwen4_exp.GatedResidual(ARGS, use_combine=combine)
    gr.hc_norm.weight = 1 + 0.2 * mx.random.normal(gr.hc_norm.weight.shape)
    gr.set_dtype(dtype)
    before = mx.random.normal((1, tokens, 4 * 96)).astype(dtype)
    out = mx.random.normal((1, tokens, 96)).astype(dtype)
    inject_in = (2 * mx.random.uniform(shape=(1, tokens, 4))).astype(dtype)
    want = reference(gr, before, out, inject_in)
    got = FusedResidual(gr)(before, out, inject_in)
    for g, w in zip(got, want):
        if w is None:
            continue
        assert g.shape == w.shape and g.dtype == dtype
        g, w = np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
        assert np.abs(g - w).max() <= tol * np.abs(w).max()


def test_nothing_before_is_a_plain_read():
    mx.random.seed(6)
    gr = qwen4_exp.GatedResidual(ARGS)
    h = mx.random.normal((1, 2, 4 * 96))
    mixed, residual, inject = FusedResidual(gr)(h, mx.zeros((1, 2, 96)), mx.zeros((1, 2, 4)))
    want_mixed, _, want_inject = gr(h)
    assert mx.array_equal(residual, h)
    assert mx.allclose(mixed, want_mixed, rtol=1e-4, atol=1e-5) and mx.allclose(inject, want_inject, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_a_token_gets_the_same_bits_alone_and_in_a_window(dtype):
    """Each token's sums are taken in the same order whatever the window: nothing here can make an answer
    written with drafts part from one written without."""
    mx.random.seed(7)
    gr = qwen4_exp.GatedResidual(ARGS)
    gr.hc_norm.weight = 1 + 0.2 * mx.random.normal(gr.hc_norm.weight.shape)
    gr.set_dtype(dtype)
    fused = FusedResidual(gr)
    before = mx.random.normal((1, 6, 4 * 96)).astype(dtype)
    out = mx.random.normal((1, 6, 96)).astype(dtype)
    inject_in = (2 * mx.random.uniform(shape=(1, 6, 4))).astype(dtype)
    window = fused(before, out, inject_in)
    for t in range(6):
        alone = fused(before[:, t : t + 1], out[:, t : t + 1], inject_in[:, t : t + 1])
        for a, w in zip(alone, window):
            assert mx.array_equal(a, w[:, t : t + 1]).item()
    three = fused(before[:, 2:5], out[:, 2:5], inject_in[:, 2:5])
    assert all(mx.array_equal(a, w[:, 2:5]).item() for a, w in zip(three, window))



@pytest.mark.parametrize("tokens", [1, 3, 8, 20])
def test_a_router_product_in_float32_from_float16_weights(tokens):
    import mlx.nn as nn

    mx.random.seed(tokens)
    linear = nn.Linear(256, 48, bias=False)
    linear.set_dtype(mx.float16)
    router = WindowLinear(linear)
    x = mx.random.normal((1, tokens, 256)).astype(mx.float16).astype(mx.float32)
    got, want = router(x), linear(x)
    assert got.shape == want.shape == (1, tokens, 48) and got.dtype == mx.float32
    assert mx.abs(got - want).max().item() <= 1e-5 * mx.abs(want).max().item()
    if tokens <= 8:  # a token's sum does not depend on the window it is in
        alone = mx.concatenate([router(x[:, t : t + 1]) for t in range(tokens)], axis=1)
        assert mx.array_equal(alone, got).item()
    linear.set_dtype(mx.float32)  # the exact way: MLX's own product
    assert mx.array_equal(router(x), linear(x)).item()
