"""Experts kept as the file's blocks must multiply with the weights gguf-py's own dequantizer gives."""

import mlx.core as mx
import numpy as np
import pytest

from strata_mlx.blocks import BLOCK, BlockSwitchLinear, dequantize
from test_gguf import random_tensor, reference

EXPERTS, OUT = 3, 2  # test_gguf's tensors are 6 rows of 2560: three experts of two rows


@pytest.mark.parametrize("kind", sorted(BLOCK))
def test_unpacked_blocks_are_the_reference_weights(kind):
    t = random_tensor(kind)
    want = reference(t)
    got = np.array(dequantize(mx.array(t.data), kind, t.shape))
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-6 * np.abs(want).max()


@pytest.mark.parametrize("kind", sorted(BLOCK))
@pytest.mark.parametrize("per_row", [1, 4, 16])
def test_products_on_blocks_are_products_on_the_weights(kind, per_row):
    t = random_tensor(kind, seed=1)
    w = reference(t).reshape(EXPERTS, OUT, -1)
    layer = BlockSwitchLinear(mx.array(t.data), kind, w.shape, per_row=per_row)
    rng = np.random.default_rng(2)
    idx = np.array([[2, 0], [1, 1], [0, 2], [2, 2], [0, 1]])  # five tokens, two experts each
    scale = np.abs(w).max() * np.sqrt(w.shape[-1])

    x = rng.standard_normal((5, 1, 1, w.shape[-1])).astype(np.float32)  # a row to each token, as gate and up get
    got = np.array(layer(mx.array(x), mx.array(idx)))
    assert got.shape == (5, 2, 1, OUT)
    assert np.abs(got[:, :, 0] - np.einsum("tkoi,ti->tko", w[idx], x[:, 0, 0])).max() <= 1e-5 * scale

    x = rng.standard_normal((5, 2, 1, w.shape[-1])).astype(np.float32)  # a row to each pair, as down gets
    got = np.array(layer(mx.array(x), mx.array(idx)))
    assert np.abs(got[:, :, 0] - np.einsum("tkoi,tki->tko", w[idx], x[:, :, 0])).max() <= 1e-5 * scale


@pytest.mark.parametrize("kind", ["IQ4_NL", "IQ2_S"])
@pytest.mark.parametrize("unpacked", [False, True])
def test_a_prompt_gets_the_same_products_a_pair_at_a_time_and_from_unpacked_weights(kind, unpacked):
    t = random_tensor(kind, seed=3)
    w = reference(t).reshape(EXPERTS, OUT, -1)
    layer = BlockSwitchLinear(mx.array(t.data), kind, w.shape)
    layer.unpack_from = 100 if unpacked else 1000
    rng = np.random.default_rng(4)
    idx = np.sort(rng.integers(0, EXPERTS, 200))  # sorted, one row each: how a long prompt arrives
    x = rng.standard_normal((200, 1, w.shape[-1])).astype(np.float32)
    got = np.array(layer(mx.array(x), mx.array(idx), sorted_indices=True))
    assert got.shape == (200, 1, OUT)
    scale = np.abs(w).max() * np.sqrt(w.shape[-1])
    assert np.abs(got[:, 0] - np.einsum("noi,ni->no", w[idx], x[:, 0])).max() <= 1e-5 * scale


def test_half_precision_activations_come_back_as_half():
    t = random_tensor("IQ3_XXS", seed=5)
    layer = BlockSwitchLinear(mx.array(t.data), "IQ3_XXS", (EXPERTS, OUT, 2560))
    x = mx.random.normal((2, 1, 1, 2560)).astype(mx.float16)
    out = layer(x, mx.array([[0, 1], [2, 0]]))
    assert out.dtype == mx.float16 and out.shape == (2, 2, 1, OUT)
    want = BlockSwitchLinear(mx.array(t.data), "IQ3_XXS", (EXPERTS, OUT, 2560))(x.astype(mx.float32), mx.array([[0, 1], [2, 0]]))
    assert mx.abs(out.astype(mx.float32) - want).max().item() <= 2e-3 * mx.abs(want).max().item()


# ---- Q2_0 for prompts: Apple's matrix product over the 2-bit codes

from strata_mlx import blocks as blocks_module  # noqa: E402
from strata_mlx.blocks import BlockSwitchGLU, packable, packed_matmul  # noqa: E402


def q2_stack(rng, experts, rows, cols):
    b = rng.integers(0, 256, (experts * rows * cols // 64, 18), dtype=np.uint8)
    b[:, :2] = rng.uniform(0.005, 0.03, (len(b), 1)).astype("<f2").view(np.uint8)
    d = b[:, :2].copy().view("<f2").astype(np.float64)
    code = np.stack([(b[:, 2:] >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, 64)
    return mx.array(b.reshape(-1)), ((code.astype(np.float64) - 1) * d).reshape(experts, rows, cols)


def test_the_packed_product_is_the_product_with_the_weights():
    rng = np.random.default_rng(0)
    blocks, w = q2_stack(rng, 4, 128, 192)
    tiles = np.array([3, 0, 0, 2, 3])  # tiles of 16 rows, each for one expert, in no particular order
    a = rng.standard_normal((16 * len(tiles), 192)).astype(np.float16)
    got = np.array(packed_matmul(blocks, (4, 128, 192), mx.array(a), mx.array(tiles.astype(np.uint32))))
    want = np.einsum("poi,pi->po", w[np.repeat(tiles, 16)], a.astype(np.float64))
    assert got.shape == want.shape and got.dtype == np.float32
    assert np.abs(got - want).max() <= 2e-3 * np.abs(want).max()  # the activations are float16


def test_a_prompt_through_packed_products_is_the_prompt_a_pair_at_a_time(monkeypatch):
    rng = np.random.default_rng(1)
    experts, hidden, inner, k = 7, 128, 64, 3
    stacks = {name: q2_stack(rng, experts, rows, cols)
              for name, (rows, cols) in (("gate", (inner, hidden)), ("up", (inner, hidden)), ("down", (hidden, inner)))}
    glu = BlockSwitchGLU(*(BlockSwitchLinear(stacks[n][0], "Q2_0", stacks[n][1].shape) for n in ("gate", "up", "down")))
    x = mx.array(rng.standard_normal((1, 211, hidden)).astype(np.float16))
    idx = rng.integers(0, experts - 1, (1, 211, k))  # runs of every length; the last expert is never chosen
    idx[0, :40] = 2  # and one expert far more than the others
    assert all(packable("Q2_0", *stacks[n][1].shape[1:], mx.float16) for n in stacks)

    a_pair = np.array(glu(x, mx.array(idx))).astype(np.float64)
    monkeypatch.setattr(blocks_module, "PACKED_FROM", 100)
    packed = np.array(glu(x, mx.array(idx))).astype(np.float64)
    assert packed.shape == a_pair.shape == (1, 211, k, hidden)

    xs = np.array(x).astype(np.float64)[0]
    gate, up = (np.einsum("tkoi,ti->tko", stacks[n][1][idx[0]], xs) for n in ("gate", "up"))
    want = np.einsum("tkoi,tki->tko", stacks["down"][1][idx[0]], gate / (1 + np.exp(-gate)) * up)
    scale = np.abs(want).max()
    assert np.abs(a_pair[0] - want).max() <= 5e-3 * scale
    assert np.abs(packed[0] - want).max() <= 5e-3 * scale


def test_only_half_floats_over_q2_0_go_the_packed_way():
    assert packable("Q2_0", 640, 2560, mx.float16)
    assert not packable("Q2_0", 640, 2560, mx.float32)  # the primitive has no float32 by 2-bit product
    assert not packable("IQ4_NL", 640, 2560, mx.float16)
    assert not packable("Q2_0", 32, 64, mx.float16)  # fewer output rows than a tile
