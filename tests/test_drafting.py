"""Guesses copied from earlier text, and the choice between them and the draft head's."""

from strata_mlx.drafting import Lookup, Policy


def test_lookup_continues_the_longest_earlier_copy():
    look = Lookup()
    #            0  1  2  3  4  5  6  7  8  9 10 11 12 13
    look.extend([9, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 8, 0])
    # "0 1 2 3" occurred at 8..11 (followed by 8) and "1 2 3" at 1..3 (followed by 4): the longer match wins
    assert look.propose([0, 1, 2, 3], 2) == ([8, 0], 4)
    assert look.propose([7, 1, 2, 3], 3) == ([8, 0, 7], 3)  # equal length: the latest copy
    assert look.propose([1, 2, 9], 3) == ([], 0)


def test_lookup_needs_three_tokens_and_indexes_what_it_is_given():
    look = Lookup()
    look.extend([1, 2])
    assert look.propose([3], 4) == ([], 0)
    look.extend([3, 4, 5, 1, 2])
    assert look.propose([3], 2) == ([4, 5], 3)
    look.truncate(4)  # 1 2 3 4: nothing follows "1 2 3" but 4
    assert look.propose([1, 2, 3], 5)[0][:1] == [4]


def test_policy_takes_lookup_only_when_it_settles_more_per_millisecond():
    p = Policy()
    for _ in range(30):
        p.observe(4, 52.0, 2, False, 0)  # the draft head: 3 tokens per 52 ms pass
    assert p.choose(4, 1, 30) == 0  # one guess, however sure, cannot beat that
    assert p.choose(4, 7, 30) >= 6  # a long, well matched copy can
    for _ in range(40):
        p.observe(8, 75.0, 0, True, 30)  # ... until it keeps being wrong
    assert p.choose(4, 7, 30) == 0
    assert p.choose(4, 0, 0) == 0
