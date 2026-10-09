"""Where guesses come from besides the draft head, and which guesses to try.

Lookup: when the text being written repeats something earlier in the conversation (an edited file, a quoted
passage), the earlier copy says what probably comes next. The last three tokens are looked up in an index of every
position seen; of the places they occurred, the one whose preceding text agrees longest wins, and the tokens that
followed it are the guesses.

Policy: a window of lookup guesses can be longer than the draft head's, but a longer window costs more and a wrong
guess wastes it. Both kinds are scored by the tokens a pass is expected to settle per millisecond, from what this
run has measured; lookup is used only where it wins.

How many guesses the head makes is decided the same way. It always makes three, in one go. The head says how
probable it finds each guess; how sure it is of a run of guesses is the product. The policy counts how often
guesses of each degree of sureness were kept, and further guesses are made, two at a time, and put into the
window, only while what they are expected to add, for the time they add, beats what the run averages
(bench/experiments/e13). Recorded passes of four prompts, replayed: 6% more tokens a second than three guesses
every pass, 15% where the answer repeats its prompt, 10% on reasoning, 2% less on prose, where the head's
sureness says little; knowing beforehand which guesses will be kept would give 32%. Deciding every guess that
way, the first three too, replayed as well and ran worse: more waits for the GPU, and on prose and code fewer
tokens a second than three guesses every pass.
"""

from collections import defaultdict

MAX_WINDOW = 8
# time of a pass by window size, relative to one token, as measured on an M4 Max (bench/results/mlx_overhead.json);
# used only until a size has been timed in this run
_SHAPE = (0.0, 1.0, 1.13, 1.28, 1.43, 1.58, 1.8, 2.0, 2.06)
_FIRST_GUESS_MS = 22.0
# chance that a lookup guess is right, by how long the matched text was, before this run has any counts
_MATCH_BUCKETS = ((6, 0.75), (12, 0.88), (24, 0.93), (1 << 30, 0.96))
_MARGIN = 1.03
# the share of the head's guesses kept, by how sure it was of the run up to them (tenths), before this run has
# any counts; measured over four prompts (e13). It is too sure of itself in the middle.
_SURE_KEPT = (0.05, 0.15, 0.28, 0.34, 0.40, 0.45, 0.52, 0.62, 0.80, 0.95)
_MAKE_MARGIN, _KEEP_MARGIN = 1.6, 1.0  # what a guess must promise, in the run's average, to be made; to be kept
FIRST_GUESSES = 3  # made in any case
_GUESS_MS, _RATE = 2.0, 0.075  # a guess of the head; tokens a millisecond: until measured


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
        self.sure = [[4 * p, 4.0] for p in _SURE_KEPT]  # per tenth of sureness: guesses kept, guesses checked
        self.guess_ms, self.rate = _GUESS_MS, _RATE

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

    # ---- how many guesses the head makes

    def _kept_rate(self, sure: float) -> float:
        kept, checked = self.sure[min(int(sure * 10), 9)]
        return kept / checked

    def _longer_ms(self, guesses: int) -> float:
        """What one more token in the window adds to a pass that has `guesses` guesses already."""
        return max(0.5, self.cost_ms(min(guesses + 2, MAX_WINDOW)) - self.cost_ms(guesses + 1))

    def worth_guessing(self, sure: float, guesses: int) -> bool:
        """Whether to make another guess, the head being `sure` of the `guesses` it has made."""
        return self._kept_rate(sure) / (self._longer_ms(guesses) + self.guess_ms) >= _MAKE_MARGIN * self.rate

    def worth_keeping(self, sure: float, guesses: int) -> bool:
        """Whether a guess just made, the head `sure` of the run up to it, goes into the window."""
        return self._kept_rate(sure) / self._longer_ms(guesses) >= _KEEP_MARGIN * self.rate

    def observe_step(self, tokens: int, ms: float):
        """A whole step, drafting included, settled `tokens` in `ms`."""
        self.rate += 0.05 * (tokens / ms - self.rate)

    def observe_guesses(self, made: int, ms: float):
        if made:
            self.guess_ms += 0.1 * (ms / made - self.guess_ms)

    def observe(self, size: int, ms: float, kept: int, from_lookup: bool, match: int, sure: tuple = ()):
        """A pass over `size` tokens took `ms` and kept `kept` of its size - 1 guesses; `sure`: how sure the head
        was of each run of its guesses, if they were its own."""
        old, n = self.cost.get(size, (ms, 0))
        self.cost[size] = (old + 0.1 * (ms - old), n + 1)
        for j, c in enumerate(sure[: kept + 1]):  # up to and including the first one refused
            b = self.sure[min(int(c * 10), 9)]
            b[0], b[1] = 0.98 * b[0] + (j < kept), 0.98 * b[1] + 1
        if from_lookup:
            b = self.lookup[self._bucket(match)]
            checked = kept + (1 if kept < size - 1 else 0)  # the guesses up to and including the first wrong one
            b[0], b[1] = 0.97 * b[0] + kept, 0.97 * b[1] + checked
        elif size > 1:
            old = self.head_tokens.get(size, kept + 1)
            self.head_tokens[size] = old + 0.05 * (kept + 1 - old)
