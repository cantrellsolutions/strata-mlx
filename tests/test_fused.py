"""The fused hyper-connection kernels against the model's own operations."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import qwen4_exp

from strata_mlx.fused import FusedResidual

ARGS = qwen4_exp.TextArgs(hidden_size=96, hc_count=4, hc_lowrank=24)


def reference(gr, before, out, inject_in):
    residual = before + (out[..., None, :] * inject_in[..., None]).reshape(*out.shape[:-1], -1)
    got = gr(residual)
    return (got, residual, None) if gr.block_inject_weight is None else (got[0], residual, got[2])


@pytest.mark.parametrize("dtype, tol", [(mx.float32, 2e-5), (mx.float16, 4e-3)])
@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("combine", [True, False])
def test_two_kernels_compute_the_read_and_the_write_before_it(dtype, tol, tokens, combine):
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
