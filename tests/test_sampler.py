"""The sampling chain against a plain reference: penalties, top_k, top_p, min_p, temperature, then the draw."""

import math
import random

import mlx.core as mx
import numpy as np
import pytest

from strata_mlx.sampler import LIST, Sampling, history_counts, penalize, pick, uniform

VOCAB = 200


def reference(logits, u, s: Sampling):
    """One row, in float64. Returns (token, distance from u to the nearest edge of its interval)."""
    if s.greedy:
        return int(np.argmax(logits)), 1.0
    k = s.top_k if 1 <= s.top_k < LIST else LIST
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:k]
    vals = np.array([logits[i] for i in order], dtype=np.float64)
    p = np.exp(vals - vals.max())
    p /= p.sum()
    keep, total = [], 0.0
    for i in range(k):
        if i and total >= s.top_p:
            break
        if i and s.min_p > 0 and vals[i] < vals[0] + math.log(s.min_p):
            break
        keep.append(i)
        total += p[i]
    t = np.exp((vals[keep] - vals[keep].max()) / s.temperature)
    cum = np.cumsum(t / t.sum())
    i = min(int(np.searchsorted(cum, u, side="right")), len(keep) - 1)
    return order[keep[i]], float(np.min(np.abs(cum - u)))


@pytest.mark.parametrize("params", [
    dict(temperature=1.0, top_k=20, top_p=0.95),
    dict(temperature=0.7, top_k=0, top_p=1.0),
    dict(temperature=1.3, top_k=5, top_p=0.5, min_p=0.05),
    dict(temperature=0.3, top_k=64, top_p=0.9, min_p=0.2),
    dict(temperature=1.0, top_k=1),
    dict(temperature=0.0),
])
def test_pick_is_the_reference_chain(params):
    s = Sampling(seed=7, **params)
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, (400, VOCAB)).astype(np.float32)
    us = [uniform(s.seed, i) for i in range(400)]
    got = pick(mx.array(logits), mx.array(us), s).tolist()
    checked = 0
    for row, u, token in zip(logits, us, got):
        want, margin = reference(row, u, s)
        if margin > 1e-4:  # away from an interval's edge, where float32 and float64 may land on either side
            assert token == want
            checked += 1
    assert checked > 350


def test_the_draw_depends_on_seed_and_index_only():
    draws = [uniform(11, i) for i in range(2000)]
    assert all(0.0 <= d < 1.0 for d in draws)
    assert draws == [uniform(11, i) for i in range(2000)]
    assert abs(sum(draws) / len(draws) - 0.5) < 0.03
    assert uniform(12, 5) != uniform(11, 5)


def test_a_new_seed_for_each_request_unless_one_is_given():
    assert Sampling().seed != Sampling().seed
    assert Sampling(seed=5).seed == 5


def test_penalties_look_at_the_recent_history_of_each_row():
    s = Sampling(penalty_last_n=3, penalty_repeat=2.0, penalty_freq=0.5, penalty_present=0.25)
    history, window = [1, 2, 3, 4], [5, 5]
    counts = history_counts(history, window, VOCAB, s.penalty_last_n)
    assert counts[0].tolist()[:7] == [0, 0, 0, 1, 1, 1, 0]  # the row sees 3, 4, 5
    assert counts[1].tolist()[:7] == [0, 0, 0, 0, 1, 2, 0]  # the row sees 4, 5, 5
    logits = mx.array([[2.0] * VOCAB, [-2.0] * VOCAB])
    out = penalize(logits, counts, s)
    assert out[0, 5].item() == pytest.approx(2.0 / 2 - 1 * 0.5 - 0.25)
    assert out[1, 5].item() == pytest.approx(-2.0 * 2 - 2 * 0.5 - 0.25)
    assert out[0, 0].item() == 2.0 and out[1, 3].item() == -2.0
    assert not Sampling(penalty_repeat=2.0).penalties  # a penalty without a window does nothing
