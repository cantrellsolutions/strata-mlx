"""The expert budget following the pressure: the rule from given readings, and answers that do not change."""

from strata_mlx.paged import LEAST, Follow

GB = 1 << 30


def test_it_sheds_only_after_two_high_readings_and_no_more_than_a_step():
    f = Follow(ceiling=20 * GB)
    assert f.decide(0.0, 5 * GB, 1, held=18 * GB, budget=20 * GB) is None  # one high reading is not enough
    new = f.decide(1.0, 5 * GB, 1, held=18 * GB, budget=20 * GB)
    assert new == 18 * GB  # at most 2 GiB at a time, though 5 are compressed
    assert f.decide(1.5, 5 * GB, 1, held=16 * GB, budget=new) is None  # not before a second has passed
    assert f.decide(2.0, 5 * GB, 1, held=16 * GB, budget=new) == 16 * GB


def test_it_sheds_to_what_is_not_compressed_and_never_below_the_least():
    f = Follow(ceiling=20 * GB, most=100 * GB)
    f.decide(0.0, 3 * GB, 1, held=10 * GB, budget=10 * GB)
    assert f.decide(1.0, 3 * GB, 1, held=10 * GB, budget=10 * GB) == 10 * GB - 3 * GB - (512 << 20)
    g = Follow(ceiling=20 * GB, most=100 * GB)
    assert g.decide(0.0, 50 * GB, 2, held=10 * GB, budget=10 * GB) == LEAST  # warning pressure: at once


def test_it_grows_back_after_a_quiet_spell_up_to_the_ceiling_only():
    f = Follow(ceiling=12 * GB, quiet_for=30.0)
    assert f.decide(0.0, 0, 1, held=10 * GB, budget=10 * GB) is None
    assert f.decide(20.0, 0, 1, held=10 * GB, budget=10 * GB) is None
    assert f.decide(30.0, 0, 1, held=10 * GB, budget=10 * GB) == 11 * GB
    assert f.decide(61.0, 0, 1, held=11 * GB, budget=11.5 * GB) == 12 * GB  # not past the ceiling
    assert f.decide(100.0, 0, 1, held=12 * GB, budget=12 * GB) is None


def test_a_reading_of_some_compression_stops_the_quiet_clock():
    f = Follow(ceiling=12 * GB, quiet_for=30.0)
    f.decide(0.0, 0, 1, held=10 * GB, budget=10 * GB)
    f.decide(20.0, 512 << 20, 1, held=10 * GB, budget=10 * GB)  # some compressed: not quiet
    assert f.decide(31.0, 0, 1, held=10 * GB, budget=10 * GB) is None  # the clock starts again
