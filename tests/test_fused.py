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



@pytest.mark.parametrize("tokens", [6, 8, 11, 16])
@pytest.mark.parametrize("combine", [True, False])
def test_the_wide_kernels_give_the_bits_of_the_narrow_ones(monkeypatch, tokens, combine):
    from strata_mlx import fused

    mx.random.seed(tokens)
    gr = qwen4_exp.GatedResidual(ARGS, use_combine=combine)
    gr.hc_norm.weight = 1 + 0.2 * mx.random.normal(gr.hc_norm.weight.shape)
    gr.set_dtype(mx.float16)
    before = mx.random.normal((1, tokens, 4 * 96)).astype(mx.float16)
    out = mx.random.normal((1, tokens, 96)).astype(mx.float16)
    inject_in = (2 * mx.random.uniform(shape=(1, tokens, 4))).astype(mx.float16)
    assert fused._wide_kernel_set(24, 4 * 96, 96, 4, tokens, 4 if combine else 0) is not None
    wide = FusedResidual(gr)(before, out, inject_in)
    monkeypatch.setattr(fused, "WIDE_FROM", 99)
    narrow = FusedResidual(gr)(before, out, inject_in)
    assert all(mx.array_equal(a, b).item() for a, b in zip(wide, narrow))

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


@pytest.mark.parametrize("tokens", [1, 3, 8, 40])
@pytest.mark.parametrize("ties", [False, True])
def test_the_routers_choice_is_argpartitions(tokens, ties):
    from strata_mlx.fused import top_k

    mx.random.seed(tokens + 10 * ties)
    for _ in range(20):
        logits = mx.random.normal((1, tokens, 512))
        if ties:
            logits = mx.round(logits * 4) / 4
        want = mx.argpartition(-logits, 9, axis=-1)[..., :10]
        assert mx.array_equal(top_k(logits, 10), want).item()



@pytest.mark.parametrize("tokens", [1, 3, 8, 16])
def test_the_end_of_the_moe_block_is_mlxs_bits(tokens):
    from strata_mlx.fused import moe_combine

    def plain(y, w, g, s):
        return (y * w[..., None]).sum(axis=-2).astype(y.dtype) + mx.sigmoid(g) * s

    mx.random.seed(tokens)
    for _ in range(20):
        y = (mx.random.normal((1, tokens, 10, 2560)) * 0.5).astype(mx.float16)
        w = mx.softmax(mx.random.normal((1, tokens, 10)), axis=-1)
        g = (2 * mx.random.normal((1, tokens, 1))).astype(mx.float16)
        s = (0.5 * mx.random.normal((1, tokens, 2560))).astype(mx.float16)
        got = moe_combine(y, w, g, s)
        assert mx.array_equal(got, plain(y, w, g, s)).item() and mx.array_equal(got, mx.compile(plain)(y, w, g, s)).item()
