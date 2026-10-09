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



def test_the_head_guesses_on_while_guesses_that_sure_are_kept():
    p = Policy()
    assert p.worth_keeping(0.97, 0) and p.worth_guessing(0.97, 3)
    assert not p.worth_keeping(0.03, 0)
    # a run in which the head, very sure, is refused every time: it stops trusting that degree of sureness
    for _ in range(200):
        p.observe(4, 30.0, 0, False, 0, sure=(0.95, 0.93, 0.91))
    assert not p.worth_keeping(0.95, 0)
    assert p.sure[9][0] / p.sure[9][1] < 0.1 and p.sure[5][1] == 4.0  # only what was checked is counted
    # and one in which unsure guesses are kept: it guesses on
    q = Policy()
    for _ in range(200):
        q.observe(4, 30.0, 3, False, 0, sure=(0.35, 0.33, 0.31))
    assert q.worth_keeping(0.33, 2)


def test_a_slow_run_wants_fewer_guesses_than_a_fast_one():
    slow, fast = Policy(), Policy()
    for _ in range(100):
        slow.observe_step(1, 40.0)  # a token a pass
        fast.observe_step(6, 40.0)
    sure = 0.55
    assert slow.worth_keeping(sure, 2) and not fast.worth_keeping(sure, 2)
