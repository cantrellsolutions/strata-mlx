"""bench/memory.py: the footprint as the kernel counts it, and a hold that says when it did not stay whole."""

import importlib.util
import time
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("memory", Path(__file__).resolve().parent.parent / "bench" / "memory.py")
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


def test_the_footprint_sees_memory_that_is_not_an_mlx_array():
    before, _ = memory.footprint()
    kept = np.ones(256 << 20, dtype=np.uint8)  # 256 MiB, written
    now, peak = memory.footprint()
    assert now - before >= 0.9 * kept.nbytes and peak >= now


def test_a_hold_holds_what_it_says_and_says_so():
    with memory.Hold(0.25) as h:
        time.sleep(2.5)
    assert h.whole and h.least >= 0.25 * memory.GIB and len(h.samples) >= 2


def test_a_hold_that_died_is_not_whole():
    with memory.Hold(0.25) as h:
        time.sleep(1.5)
        h.proc.kill()
        h.proc.wait()
    assert not h.whole
