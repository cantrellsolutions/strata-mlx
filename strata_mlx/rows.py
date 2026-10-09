"""The rows of the output layer that a guess of the draft head is worked out over.

A guess is the largest of 248,320 logits, a product with the whole output layer for one token: 1.0 ms with the
Q2_0 file and 1.4 with IQ3_S, three to seven times a step. But the token that wins is nearly always one that
has been in sight: recorded over four answers (bench/experiments/e25, e26), 72% of the guesses the trunk kept
were tokens of the conversation itself, and 98% were either that or among the 256 largest of some row of
logits the trunk had worked out by then. Those rows cost nothing to know, the trunk works out every one of
them anyway. So they are listed, and a guess multiplies only the listed rows, 5% of the layer after a few
hundred tokens. A guess that would have been a row not listed comes out as another token and is refused: that
costs speed, never the answer, which is the trunk's.

The list is kept from one conversation to the next (what one left behind found another 1% of the guesses of
the next) and started again when it passes `most` rows. Until it has `start` rows a guess goes over all rows
as before. The largest of a pass's rows are picked out by the GPU while the CPU is busy with the draft, and are
on the list from the next step on.

The head's probability of a guess, which decides how many guesses it makes (drafting.py), is taken among the
listed rows only. That makes it too sure by less than 0.1% for nine guesses in ten and moves one in a hundred
into another tenth.
"""

import mlx.core as mx
import numpy as np

from .dense import ScaledLinear

TOP, START, MOST, PIECES = 256, 1000, 48_000, 4


def _rows_of(head, ids: np.ndarray):
    """The rows `ids` of a layer as a layer of their own: x (1, in) -> (1, len(ids))."""
    n = len(ids)
    if isinstance(head, ScaledLinear):
        padded = mx.array(np.concatenate([ids, np.repeat(ids[-1:], -n % 4)]))  # its kernel takes rows by fours
        part = ScaledLinear(*(mx.take(a, padded, axis=0) for a in (head.codes, head.scales, head.d)))
        return lambda x: part(x)[:, :n]
    at = mx.array(ids)
    if hasattr(head, "scales"):
        w, s, b = (mx.take(a, at, axis=0) for a in (head.weight, head.scales, head.biases))
        return lambda x: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=head.group_size, bits=head.bits)
    w = mx.take(head.weight, at, axis=0)
    return lambda x: x @ w.T


class GuessRows:
    def __init__(self, head, vocab: int, top: int = TOP, start: int = START, most: int = MOST):
        self.head, self.vocab = head, vocab
        self.top, self.start, self.most = min(top, vocab - 1), start, most
        self.clear()

    def clear(self):
        self.listed = np.zeros(self.vocab, dtype=bool)
        self.pieces = []  # (ids, their rows as a layer): what each call of `add` took out, joined now and then
        self.count = 0
        self.pending = None
        self._ids = None

    @property
    def ready(self) -> bool:
        return self.count >= self.start

    def add(self, ids):
        """List these rows."""
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        new = ids[~self.listed[ids]]
        if not len(new):
            return
        if self.count + len(new) > self.most:
            pending = self.pending
            self.clear()
            self.pending = pending
        self.listed[new] = True
        self.count += len(new)
        self.pieces.append((new, _rows_of(self.head, new)))
        if len(self.pieces) > PIECES:
            everything = np.concatenate([i for i, _ in self.pieces])
            self.pieces = [(everything, _rows_of(self.head, everything))]
        self._ids = None

    def note(self, logits: mx.array):
        """logits (rows, vocab) the trunk has worked out: their largest are listed by the next call of this.
        Nothing here waits for the GPU unless the last call's are not picked out yet."""
        self.collect()
        self.pending = mx.argpartition(-logits, self.top, axis=-1)[:, : self.top]
        mx.async_eval(self.pending)

    def collect(self):
        if self.pending is not None:
            pending, self.pending = self.pending, None
            self.add(np.array(pending).reshape(-1))

    def logits(self, x: mx.array):
        """x (1, in) -> (the listed rows' logits (1, count), the token of each)."""
        if self._ids is None:
            self._ids = mx.array(np.concatenate([i for i, _ in self.pieces]))
        parts = [layer(x) for _, layer in self.pieces]
        return (parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)), self._ids
