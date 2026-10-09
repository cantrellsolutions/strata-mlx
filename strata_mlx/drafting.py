"""Where guesses come from besides the draft head, and which guesses to try.

Lookup: when the text being written repeats something earlier in the conversation (an edited file, a quoted
passage), the earlier copy says what probably comes next. The last three tokens are looked up in an index of every
position seen; of the places they occurred, the one whose preceding text agrees longest wins, and the tokens that
followed it are the guesses.

Policy: a window of lookup guesses can be longer than the draft head's, but a longer window costs more and a wrong
guess wastes it. Both kinds are scored by the tokens a pass is expected to settle per millisecond, from what this
run has measured; lookup is used only where it wins.
"""

from collections import defaultdict

MAX_WINDOW = 8
# time of a pass by window size, relative to one token, as measured on an M4 Max (bench/results/mlx_overhead.json);
# used only until a size has been timed in this run
_SHAPE = (0.0, 1.0, 1.0, 1.22, 1.41, 1.6, 1.79, 1.93, 2.08)
_FIRST_GUESS_MS = 30.0
# chance that a lookup guess is right, by how long the matched text was, before this run has any counts
_MATCH_BUCKETS = ((6, 0.75), (12, 0.88), (24, 0.93), (1 << 30, 0.96))
_MARGIN = 1.03


class Lookup:
    def __init__(self, min_match: int = 3, max_match: int = 64, ways: int = 4):
        self.min_match, self.max_match, self.ways = max(3, min_match), max_match, ways
        self.reset()

    def reset(self):
        self.seen = []
        self.index = defaultdict(list)  # the three tokens ending at a position -> the latest such positions

    def extend(self, tokens: list):
        for t in tokens:
            self.seen.append(t)
            if len(self.seen) >= 3:
                at = self.index[tuple(self.seen[-3:])]
                at.insert(0, len(self.seen) - 1)
                del at[self.ways :]

    def truncate(self, n: int):
        """Forget everything after the first n tokens."""
        if n < len(self.seen):
            kept = self.seen[:n]
            self.reset()
            self.extend(kept)

    def propose(self, pending: list, max_k: int):
        """Guesses for what follows `seen + pending` -> (tokens, length of the matched text)."""
        text = self.seen + pending
        cur = len(text) - 1
        if cur < 3 or max_k <= 0:
            return [], 0
        best_len, best_end = 0, -1
        for end in self.index.get(tuple(text[-3:]), ()):
            if end >= cur:
                continue
            n = 0
            while n < self.max_match and n <= end and text[end - n] == text[cur - n]:
                n += 1
            if n > best_len:
                best_len, best_end = n, end
        if best_len < self.min_match:
            return [], 0
        return text[best_end + 1 : best_end + 1 + max_k], best_len


class Policy:
    def __init__(self):
        self.cost = {}  # window size -> (milliseconds, smoothed; how many times measured)
        self.head_tokens = {}  # window size -> tokens a draft-head pass settles, smoothed
        self.lookup = [[0.0, 0.0] for _ in _MATCH_BUCKETS]  # per bucket: guesses right, guesses checked

    def _bucket(self, match: int) -> int:
        return next(i for i, (below, _) in enumerate(_MATCH_BUCKETS) if match < below)

    def cost_ms(self, size: int) -> float:
        if size in self.cost:
            return self.cost[size][0]
        if not self.cost:
            return _SHAPE[size] * _FIRST_GUESS_MS
        weight = sum(min(n, 20) for _, n in self.cost.values())
        return sum(ms / _SHAPE[s] * min(n, 20) for s, (ms, n) in self.cost.items()) / weight * _SHAPE[size]

    def lookup_rate(self, match: int) -> float:
        right, checked = self.lookup[self._bucket(match)]
        return (right + 4 * _MATCH_BUCKETS[self._bucket(match)][1]) / (checked + 4)

    def choose(self, head_window: int, lookup_len: int, match: int) -> int:
        """How many lookup guesses to try instead of the draft head's window; 0 keeps the head's."""
        if lookup_len <= 0:
            return 0
        head = self.head_tokens.get(head_window, 1 + 0.7 * (head_window - 1)) / self.cost_ms(head_window)
        q = self.lookup_rate(match)
        best_k, best = 0, head * _MARGIN
        expect = 1.0
        for k in range(1, min(lookup_len, MAX_WINDOW - 1) + 1):
            expect += q**k
            rate = expect / self.cost_ms(k + 1)
            if rate > best:
                best_k, best = k, rate
        return best_k

    def observe(self, size: int, ms: float, kept: int, from_lookup: bool, match: int):
        """A pass over `size` tokens took `ms` and kept `kept` of its size - 1 guesses."""
        old, n = self.cost.get(size, (ms, 0))
        self.cost[size] = (old + 0.1 * (ms - old), n + 1)
        if from_lookup:
            b = self.lookup[self._bucket(match)]
            checked = kept + (1 if kept < size - 1 else 0)  # the guesses up to and including the first wrong one
            b[0], b[1] = 0.97 * b[0] + kept, 0.97 * b[1] + checked
        elif size > 1:
            old = self.head_tokens.get(size, kept + 1)
            self.head_tokens[size] = old + 0.05 * (kept + 1 - old)
