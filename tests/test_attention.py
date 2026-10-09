"""Attention to the chosen cells must be the port's attention under a mask, at every length and window size.

A small attention layer with a budget of 32 cells, in float32: the port's own call on its own cache beside this
engine's on its cache, fed the same tokens, cut back the same way.
"""

import mlx.core as mx
import pytest
from mlx_lm.models import qwen4_exp
from mlx_lm.models.base import create_attention_mask

from strata_mlx import attention
from strata_mlx.attention import AttnCache, Attention
from strata_mlx.engine import Engine

ARGS = qwen4_exp.TextArgs(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                          indexer_n_heads=2, indexer_head_dim=16, indexer_budget=32, indexer_compress_ratio=4)


@pytest.fixture(scope="module")
def layer():
    mx.random.seed(5)
    layer = qwen4_exp.Attention(ARGS)
    mx.eval(layer.parameters())
    return layer


@pytest.fixture(scope="module")
def rope():
    return qwen4_exp.RotaryEmbedding(int(ARGS.head_dim * ARGS.partial_rotary_factor), ARGS.rope_theta)


class Pair:
    """The port's attention and this engine's, side by side."""

    def __init__(self, layer, rope):
        self.layer, self.rope = layer, rope
        self.theirs, self.ours = qwen4_exp._AttnCache(), AttnCache()

    def feed(self, S: int):
        x = mx.random.normal((1, S, ARGS.hidden_size))
        want = qwen4_exp.Attention.__call__(self.layer, x, self.rope, create_attention_mask(x, self.theirs),
                                            self.theirs, self.theirs.indexer)
        got = Attention.__call__(self.layer, x, self.rope, create_attention_mask(x, self.ours), self.ours,
                                 self.ours.indexer)
        assert mx.allclose(got, want, rtol=1e-4, atol=1e-5).item(), f"{S} queries at {self.theirs.offset - S} cells"

    def trim(self, n: int):
        self.theirs.trim(n)
        self.ours.trim(n)


@pytest.mark.parametrize("before", [0, 25, 29, 33, 38, 200, 2433])
@pytest.mark.parametrize("S", [1, 2, 3, 4, 8])
def test_a_window_attends_as_the_port_does(layer, rope, before, S):
    """Lengths around the budget: a window that crosses it has queries with fewer whole blocks than the budget."""
    mx.random.seed(before + S)
    pair = Pair(layer, rope)
    if before:
        pair.feed(before)
    for _ in range(3):
        pair.feed(S)


def test_cut_back_and_gone_on(layer, rope):
    """A window kept in part, again and again: the pooled keys of blocks that were cut must not stay."""
    mx.random.seed(9)
    pair = Pair(layer, rope)
    pair.feed(41)
    for S, dropped in [(8, 5), (4, 4), (7, 2), (1, 1), (8, 7), (5, 0), (3, 2), (6, 6), (8, 3)] * 3:
        pair.feed(S)
        pair.trim(dropped)
    pair.trim(30)  # back to a checkpoint, into blocks pooled long ago
    pair.feed(6)
    pair.feed(1)
    ours = pair.ours.indexer
    whole = ours.offset // 4
    again = AttnCache().indexer
    again.state = ours.state
    assert mx.array_equal(ours.blocks(layer.indexer, rope), again.blocks(layer.indexer, rope)).item()
    assert ours.blocks(layer.indexer, rope).shape[1] == whole


@pytest.mark.parametrize("S", [9, 40, 130])
def test_a_chunk_of_prompt_gathers_in_parts(layer, rope, S, monkeypatch):
    monkeypatch.setattr(attention, "GATHER_FROM", 100)
    monkeypatch.setattr(attention, "GATHER_BYTES", 2 * 2 * 36 * 32 * 4 * 16)  # sixteen queries at a time
    mx.random.seed(S)
    pair = Pair(layer, rope)
    pair.feed(90)  # under GATHER_FROM once fed: through the mask
    pair.feed(S)
    pair.feed(S)


def test_a_saved_cache_is_the_cache(layer, rope):
    mx.random.seed(2)
    pair = Pair(layer, rope)
    pair.feed(77)
    restored = AttnCache()
    restored.state = pair.ours.state
    pair.ours = restored
    assert restored.offset == 77 and restored.indexer.offset == 77
    pair.feed(4)
    pair.feed(1)


# ---- the whole engine past the budget

TEXT = dict(
    hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=128,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
    hc_count=4, hc_lowrank=8, indexer_n_heads=2, indexer_head_dim=16, indexer_budget=16, indexer_compress_ratio=4,
    ngram_vocab_size_base=1000, split_ngram_parts=4, ple_embed_dim=64, ple_layer_ids=[2], eos_token_id=5,
)


def test_windows_past_the_budget_write_what_single_tokens_write():
    mx.random.seed(3)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config=TEXT))
    mx.eval(model.parameters())
    model.eval()
    prompt = [(7 * i + 3) % 120 + 6 for i in range(45)]
    answer = Engine(model).generate(prompt, 40)
    engine = Engine(model, chunk=13)
    out = [engine.prefill(prompt)]
    while len(out) < 40:
        engine.guesses = [t if i != 2 else (t + 1) % 128 for i, t in enumerate(answer[len(out) : len(out) + 5])]
        out += engine.step()
    assert out[:40] == answer
    assert 0 < engine.stats.accepted < engine.stats.drafted
