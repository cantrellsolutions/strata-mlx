"""Guesses worked out over a list of the output layer's rows: the listed rows' logits are the layer's own, and
whatever the list holds, the answer is the trunk's."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from strata_mlx.dense import ScaledLinear
from strata_mlx.engine import Engine
from strata_mlx.mtp import Mtp
from strata_mlx.rows import PIECES, GuessRows
from strata_mlx.sampler import Sampling
from tests.test_engine import N, PROMPT, TEXT, answer, model  # noqa: F401

VOCAB, DIM = 400, 256


def float_head():
    lin = nn.Linear(DIM, VOCAB, bias=False)
    lin.weight = mx.random.normal((VOCAB, DIM))
    return lin


def quantized_head():
    return nn.QuantizedLinear.from_linear(float_head(), group_size=64, bits=5)


def scaled_head():
    rng = np.random.default_rng(0)
    return ScaledLinear(mx.array(rng.integers(-32, 32, (VOCAB, DIM), dtype=np.int8)),
                        mx.array(rng.integers(-128, 128, (VOCAB, DIM // 16), dtype=np.int8)),
                        mx.array(rng.uniform(0.001, 0.01, (VOCAB, DIM // 256)).astype(np.float16)))


@pytest.mark.parametrize("make", [float_head, quantized_head, scaled_head])
def test_listed_rows_have_the_layers_own_logits(make):
    head = make()
    rows = GuessRows(head, VOCAB, top=8, start=10)
    x = mx.random.normal((1, DIM))
    full = np.array(head(x).astype(mx.float32))[0]
    rng = np.random.default_rng(1)
    for step in range(2 * PIECES + 3):  # past the point where the pieces are joined
        rows.add(rng.integers(0, VOCAB, 7))  # some of them listed already, some twice
        logits, tokens = rows.logits(x)
        tokens = np.array(tokens)
        assert len(tokens) == rows.count == len(set(tokens.tolist())) == rows.listed.sum()
        assert len(rows.pieces) <= PIECES
        got = np.array(logits.astype(mx.float32))[0]
        assert np.abs(got - full[tokens]).max() <= 1e-4 * np.abs(full).max()
    assert rows.ready


def test_the_largest_of_noted_rows_are_listed_by_the_next_call():
    head = float_head()
    rows = GuessRows(head, VOCAB, top=5, start=1)
    logits = mx.random.normal((3, VOCAB))
    rows.note(logits)
    assert rows.count == 0  # not yet: the GPU picks them out while the draft is made
    rows.note(mx.zeros((0, VOCAB)))
    want = set(np.argsort(-np.array(logits), axis=1)[:, :5].reshape(-1).tolist())
    assert set(np.flatnonzero(rows.listed).tolist()) == want


def test_a_list_that_has_grown_too_long_is_started_again():
    rows = GuessRows(float_head(), VOCAB, top=4, start=1, most=50)
    rows.add(range(40))
    rows.add(range(100, 130))
    assert rows.count == 30 and not rows.listed[:40].any() and rows.listed[100:130].all()


def head_for(model):  # noqa: F811
    mx.random.seed(4)
    mtp = Mtp(model.args.text)
    mx.eval(mtp.parameters())
    return mtp


def test_with_every_row_listed_the_guesses_are_the_same(model, answer):  # noqa: F811
    plain = Engine(model, head_for(model), drafts=3)
    assert plain.rows is None  # a vocabulary this small is not listed unless asked
    assert plain.generate(PROMPT, N) == answer
    listed = Engine(model, head_for(model), drafts=3)
    listed.rows = GuessRows(model.lm_head, TEXT["vocab_size"], top=8, start=1)
    listed.rows.add(range(TEXT["vocab_size"]))
    assert listed.generate(PROMPT, N) == answer
    assert (listed.stats.steps, listed.stats.accepted) == (plain.stats.steps, plain.stats.accepted)


def test_whatever_is_listed_the_answer_is_the_trunks(model, answer):  # noqa: F811
    engine = Engine(model, head_for(model), drafts=3)
    engine.rows = GuessRows(model.lm_head, TEXT["vocab_size"], top=2, start=1)
    engine.rows.add([t for t in range(TEXT["vocab_size"]) if t not in answer][:70])  # none of the tokens it will need
    assert engine.generate(PROMPT, N) == answer
    assert engine.rows.count > 70  # in use from the first guess, and grown by what the trunk's rows held


def test_a_seed_writes_the_same_text_with_the_list(model):  # noqa: F811
    sampled = dict(temperature=1.0, top_k=20, top_p=0.95, seed=17)
    plain = Engine(model).generate(PROMPT, N, sampling=Sampling(**sampled))
    engine = Engine(model, head_for(model), drafts=3)
    engine.rows = GuessRows(model.lm_head, TEXT["vocab_size"], top=8, start=1)
    engine.rows.add(range(0, TEXT["vocab_size"], 2))
    assert engine.rows.count == 64
    engine.rows.add([1])
    assert engine.generate(PROMPT, N, sampling=Sampling(**sampled)) == plain
