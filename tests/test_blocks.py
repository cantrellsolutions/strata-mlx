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


# ---- prompts: an expert's blocks a tile at a time, under Apple's matrix product

from strata_mlx import blocks as blocks_module  # noqa: E402
from strata_mlx.blocks import BlockSwitchGLU, tile_matmul, tileable  # noqa: E402
from test_gguf import SCALE_BYTES  # noqa: E402


def stack(kind, rng, experts, rows, cols):
    """Random blocks of a stack of `kind`, and its weights as the unpacking kernel gives them (tested above)."""
    per_block, block_bytes = BLOCK[kind]
    n = experts * rows * cols // per_block
    b = rng.integers(0, 256, (n, block_bytes), dtype=np.uint8)
    for at in SCALE_BYTES[kind]:
        b[:, at : at + 2] = rng.uniform(0.001, 0.01, (n, 1)).astype("<f2").view(np.uint8)
    if kind == "IQ1_M":  # its scale is spread over the top four bits of the four words at byte 48
        d = rng.uniform(0.001, 0.01, n).astype("<f2").view("<u2")
        words = b[:, 48:56].copy().view("<u2")
        for i in range(4):
            words[:, i] = (words[:, i] & 0x0FFF) | (((d >> (4 * i)) & 15) << 12)
        b[:, 48:56] = words.view(np.uint8)
    blocks = mx.array(b.reshape(-1))
    return blocks, np.array(dequantize(blocks, kind, (experts, rows, cols), mx.float32))


def q2_stack(rng, experts, rows, cols):
    blocks, w = stack("Q2_0", rng, experts, rows, cols)
    return blocks, w.astype(np.float64)


@pytest.mark.parametrize("kind", sorted(BLOCK))
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_the_tile_product_is_the_product_with_the_weights(kind, dtype):
    rng = np.random.default_rng(0)
    shape = (4, 128, 512)
    blocks, w = stack(kind, rng, *shape)
    w = w.astype(dtype).astype(np.float64)  # a tile's weights are rounded to the activations' type
    tiles = np.array([3, 0, 0, 2, 3])  # tiles of 16 rows, each for one expert, in no particular order
    a = rng.standard_normal((16 * len(tiles), shape[2])).astype(dtype)
    got = np.array(tile_matmul(blocks, kind, shape, mx.array(a), mx.array(tiles.astype(np.uint32))))
    want = np.einsum("poi,pi->po", w[np.repeat(tiles, 16)], a.astype(np.float64))
    assert got.shape == want.shape and got.dtype == np.float32
    assert np.abs(got - want).max() <= 2e-6 * np.abs(want).max()  # sums are float32


@pytest.mark.parametrize("kinds", [("Q2_0", "Q2_0"), ("IQ3_S", "IQ4_NL"), ("IQ2_S", "Q2_0")])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_a_prompt_through_tiles_is_the_prompt_a_pair_at_a_time(monkeypatch, kinds, dtype):
    rng = np.random.default_rng(1)
    experts, hidden, inner, k = 7, 256, 64, 3
    stacks = {name: (kind, *stack(kind, rng, experts, rows, cols))
              for name, kind, (rows, cols) in (("gate", kinds[0], (inner, hidden)), ("up", kinds[0], (inner, hidden)),
                                               ("down", kinds[1], (hidden, inner * 4)))}
    # down reads four times what gate and up give: the test's own widening, so that its rows end on a block
    glu = BlockSwitchGLU(*(BlockSwitchLinear(stacks[n][1], stacks[n][0], stacks[n][2].shape) for n in ("gate", "up")),
                         BlockSwitchLinear(stacks["down"][1], stacks["down"][0], stacks["down"][2].shape))
    assert all(tileable(stacks[n][0], *stacks[n][2].shape[1:]) for n in stacks)
    rows = 211
    x = rng.standard_normal((1, rows, hidden)).astype(dtype)
    idx = rng.integers(0, experts - 1, (1, rows, k))  # runs of every length; the last expert is never chosen
    idx[0, :40] = 2  # and one expert far more than the others
    w = {n: stacks[n][2].astype(np.float64) for n in stacks}

    def through(glu_fn):
        gate_, up_, down_ = glu.gate_proj, glu.up_proj, glu.down_proj
        xs, ix = mx.array(x), mx.array(idx)
        t, token, where = blocks_module.tiles_of(ix, experts)
        a = xs.reshape(-1, hidden)[token]
        g = glu_fn(gate_, a, t)[where].reshape(1, rows, k, inner)
        u = glu_fn(up_, a, t)[where].reshape(1, rows, k, inner)
        return np.array(g).astype(np.float64), np.array(u).astype(np.float64)

    g, u = through(lambda lin, a, t: tile_matmul(lin.blocks, lin.kind, (experts, lin.output_dims, lin.input_dims), a, t))
    xs = x.astype(np.float64)[0]
    tol = 2e-3 if dtype == np.float16 else 2e-5
    for got, name in ((g, "gate"), (u, "up")):
        want = np.einsum("tkoi,ti->tko", w[name][idx[0]], xs)
        assert np.abs(got[0] - want).max() <= tol * np.abs(want).max()


def test_a_prompts_three_products_go_through_tiles(monkeypatch):
    """switch_glu sends a chunk's pairs through the tiles, and what comes out is what a pair at a time gives."""
    rng = np.random.default_rng(1)
    experts, hidden, inner, k = 7, 128, 64, 3
    stacks = {name: q2_stack(rng, experts, rows, cols)
              for name, (rows, cols) in (("gate", (inner, hidden)), ("up", (inner, hidden)), ("down", (hidden, inner)))}
    glu = BlockSwitchGLU(*(BlockSwitchLinear(stacks[n][0], "Q2_0", stacks[n][1].shape) for n in ("gate", "up", "down")))
    x = mx.array(rng.standard_normal((1, 211, hidden)).astype(np.float16))
    idx = rng.integers(0, experts - 1, (1, 211, k))  # runs of every length; the last expert is never chosen
    idx[0, :40] = 2  # and one expert far more than the others

    called = []
    tiled = blocks_module._tiled_glu
    monkeypatch.setattr(blocks_module, "_tiled_glu", lambda *a: called.append(1) or tiled(*a))
    monkeypatch.setattr(blocks_module, "LOOKUP_FROM", 1 << 62)
    monkeypatch.setattr(blocks_module, "TILES_FROM", 1 << 62)
    a_pair = np.array(glu(x, mx.array(idx))).astype(np.float64)
    assert not called
    monkeypatch.setattr(blocks_module, "TILES_FROM", 100)
    in_tiles = np.array(glu(x, mx.array(idx))).astype(np.float64)
    assert called and in_tiles.shape == a_pair.shape == (1, 211, k, hidden)

    xs = np.array(x).astype(np.float64)[0]
    gate, up = (np.einsum("tkoi,ti->tko", stacks[n][1][idx[0]], xs) for n in ("gate", "up"))
    want = np.einsum("tkoi,tki->tko", stacks["down"][1][idx[0]], gate / (1 + np.exp(-gate)) * up)
    scale = np.abs(want).max()
    assert np.abs(a_pair[0] - want).max() <= 5e-3 * scale
    assert np.abs(in_tiles[0] - want).max() <= 5e-3 * scale


@pytest.mark.parametrize("tokens, k, experts", [(211, 3, 7), (5, 2, 40), (64, 10, 512), (1, 1, 3)])
def test_tiles_hold_every_pair_with_its_expert(tokens, k, experts):
    from strata_mlx.blocks import TOKENS_TILE, tiles_of

    rng = np.random.default_rng(tokens)
    idx = rng.integers(0, experts, (1, tokens, k))
    idx[0, : tokens // 3] = 1  # one expert with many pairs; and where it can be, runs of exactly a tile and one more
    if tokens >= 33 and experts > 4:
        idx[0, -33:, 0] = np.repeat([3, 4], [16, 17])
    tile_expert, token, where = (np.array(a) for a in tiles_of(mx.array(idx), experts))
    flat = idx.reshape(-1)
    assert len(tile_expert) * TOKENS_TILE == len(token) and len(where) == len(flat)
    assert len(set(where.tolist())) == len(flat)  # a slot to each pair
    np.testing.assert_array_equal(tile_expert[where // TOKENS_TILE], flat)  # in a tile of its expert
    np.testing.assert_array_equal(token[where], np.arange(len(flat)) // k)  # and the slot knows its token
    used = np.zeros(len(tile_expert), dtype=bool)
    used[where // TOKENS_TILE] = True
    assert (tile_expert[~used] == experts).all() and (tile_expert[used] < experts).all()
    assert used.sum() == sum(-(-c // TOKENS_TILE) for c in np.bincount(flat))  # no expert has a tile to spare


def test_a_tile_of_no_expert_is_left_out():
    rng = np.random.default_rng(3)
    blocks, w = q2_stack(rng, 4, 128, 192)
    tiles = np.array([1, 4, 2, 9])  # the second and the last name no expert
    a = rng.standard_normal((16 * len(tiles), 192)).astype(np.float16)
    got = np.array(tile_matmul(blocks, "Q2_0", (4, 128, 192), mx.array(a), mx.array(tiles.astype(np.uint32))))
    for t in (0, 2):
        want = np.einsum("oi,pi->po", w[tiles[t]], a[16 * t : 16 * t + 16].astype(np.float64))
        assert np.abs(got[16 * t : 16 * t + 16] - want).max() <= 2e-6 * np.abs(want).max()


def test_what_fits_in_tiles():
    assert tileable("Q2_0", 640, 2560) and tileable("IQ3_S", 640, 2560) and tileable("IQ4_NL", 2560, 640)
    assert not tileable("IQ3_S", 2560, 640)  # its blocks are 256 weights: a row of 640 does not end on one
    assert not tileable("Q2_0", 2, 2560)  # rows by the 64
