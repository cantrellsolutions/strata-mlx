"""Is the GPU at its full speed right now? Timings here move by up to a half for seconds or for minutes, and
nothing else shows it: a 4096-cube float16 product runs at 14.5 T operations a second on an M4 Max at rest,
at 12 after twenty seconds of full load (a 14-inch MacBook Pro; the system then calls its thermal state "fair"),
and at 9 or less in the middle of a long benchmark. Whether that is all the machine's own doing or sometimes
another program's was not settled.

`Quiet` times that product before and after a measurement: it waits for the GPU to be at speed before one and
has it repeated if the GPU was not at speed after, at speed being 93% of the best seen in this process. It is
only used where asked for (`spec_decode.py --quiet`).
"""

import time

import mlx.core as mx

_a = None


def gpu_speed(rounds: int = 3) -> float:
    """T operations a second of a float16 matrix product, the best of a few."""
    global _a
    if _a is None:
        _a = mx.random.normal((4096, 4096)).astype(mx.float16)
        mx.eval(_a)
        mx.eval(_a @ _a)
    best = 1e9
    for _ in range(rounds):
        t0 = time.perf_counter()
        mx.eval(_a @ _a)
        best = min(best, time.perf_counter() - t0)
    return 2 * 4096**3 / best / 1e12


class Quiet:
    def __init__(self, share: float = 0.93, tries: int = 4, patience: float = 600.0):
        self.share, self.tries, self.patience, self.best = share, tries, patience, gpu_speed()

    def wait(self) -> float:
        """Wait, `patience` seconds at most, for the GPU to be quiet; returns its speed then."""
        t0 = time.perf_counter()
        while True:
            now = gpu_speed()
            self.best = max(self.best, now)
            if now >= self.share * self.best or time.perf_counter() - t0 > self.patience:
                return now
            time.sleep(3)

    def measure(self, fn):
        """fn() -> its result, from a run with the GPU quiet before and after; and how quiet the run kept was
        (1.0: as fast as the best seen in this process). After `tries` runs the last is kept whatever it was."""
        for _ in range(self.tries):
            before = self.wait()
            out = fn()
            after = gpu_speed()
            self.best = max(self.best, after)
            if min(before, after) >= self.share * self.best:
                break
        return out, min(before, after) / self.best
