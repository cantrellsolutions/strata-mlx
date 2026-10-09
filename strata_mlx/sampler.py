"""From logits to a token, the way Strata's sampler does it.

    penalties -> top_k -> top_p -> min_p -> temperature -> draw

top_p and min_p cut the list at temperature 1; the temperature only reshapes what survives. The list is never
longer than 64 tokens. The random number for the token at index i of the conversation is a function of the
request's seed and of i alone, so it is the same number whether that token is decided alone or as one row of a
window: a drafted token is kept exactly when it is the token this draw picks, and drafting cannot change what a
seed writes.

Everything runs as MLX operations, so a draft step can sample without stopping to hand numbers to the host.
"""

import os
from dataclasses import dataclass

import mlx.core as mx

LIST = 64
_MASK = (1 << 64) - 1


@dataclass
class Sampling:
    temperature: float = 0.0  # 0: always the most likely token
    top_k: int = 20
    top_p: float = 1.0
    min_p: float = 0.0
    penalty_last_n: int = 0  # how many of the latest tokens the penalties look at; 0 turns them off
    penalty_repeat: float = 1.0
    penalty_freq: float = 0.0
    penalty_present: float = 0.0
    seed: int = 0  # 0: a new one for each request

    def __post_init__(self):
        if not self.seed:
            self.seed = int.from_bytes(os.urandom(8), "little") | 1

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0

    @property
    def penalties(self) -> bool:
        return self.penalty_last_n > 0 and (
            self.penalty_repeat != 1.0 or self.penalty_freq != 0.0 or self.penalty_present != 0.0)


def uniform(seed: int, index: int) -> float:
    """The draw for the token at `index`: splitmix64 of (seed, index), its top 24 bits as a number in [0, 1)."""
    z = (seed * 0x9E3779B97F4A7C15 + index * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return ((z ^ (z >> 31)) >> 40) * 2.0**-24


def penalize(logits: mx.array, counts: mx.array, s: Sampling) -> mx.array:
    """logits, counts (rows, vocab): how often each token occurs in the row's recent history."""
    seen = counts > 0
    scaled = mx.where(logits <= 0, logits * s.penalty_repeat, logits / s.penalty_repeat)
    return mx.where(seen, scaled - counts * s.penalty_freq - s.penalty_present, logits)


def history_counts(history: list, window: list, vocab: int, last_n: int) -> mx.array:
    """Row t of a window sees the tokens before the window and the window's first t + 1 tokens."""
    rows, cols = [], []
    for t in range(len(window)):
        recent = (history + window[: t + 1])[-last_n:]
        rows += [t] * len(recent)
        cols += recent
    counts = mx.zeros((len(window), vocab), dtype=mx.float32)
    return counts.at[mx.array(rows), mx.array(cols)].add(1.0)


def pick(logits: mx.array, u: mx.array, s: Sampling) -> mx.array:
    """logits (rows, vocab), u (rows,) -> the token of each row."""
    if s.greedy:
        return mx.argmax(logits, axis=-1)
    logits = logits.astype(mx.float32)
    k = s.top_k if 1 <= s.top_k < LIST else LIST
    ids = mx.argpartition(-logits, LIST - 1, axis=-1)[..., :LIST]
    vals = mx.take_along_axis(logits, ids, axis=-1)
    order = mx.argsort(-vals, axis=-1)[..., :k]
    ids, vals = mx.take_along_axis(ids, order, axis=-1), mx.take_along_axis(vals, order, axis=-1)

    p = mx.softmax(vals, axis=-1)
    before = mx.cumsum(p, axis=-1) - p
    keep = before < s.top_p  # the token that carries the sum past top_p is the last one kept
    if s.min_p > 0:
        keep = keep & (vals >= vals[..., :1] + mx.log(mx.array(s.min_p)))
    keep = keep | (mx.arange(k) == 0)

    p = mx.softmax(mx.where(keep, vals / s.temperature, -mx.inf), axis=-1)
    cum = mx.cumsum(p, axis=-1)
    last = mx.sum(keep, axis=-1) - 1
    chosen = mx.minimum(mx.sum(cum <= u[..., None], axis=-1), last)
    return mx.take_along_axis(ids, chosen[..., None], axis=-1)[..., 0]
