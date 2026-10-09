"""How many expert-sized reads a second the disk gives, past the page cache.

A Q2_0 expert is three tensors of 460,800 bytes; the larger files' are up to 0.9 MB. Reads are at random offsets
of the files given on the command line, but only where mincore() says no page is in memory, and with F_NOCACHE
set so that reading does not put them there. (Without the first, a machine with a lot of memory reports 20-48
GB/s: its page cache.)
"""

import argparse
import ctypes
import fcntl
import mmap
import os
import random
import threading
import time

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--sizes", nargs="+", type=int, default=[460_800, 921_600])
p.add_argument("--threads", nargs="+", type=int, default=[1, 4, 8, 16])
p.add_argument("--seconds", type=float, default=4.0)
args = p.parse_args()
PAGE, WINDOW = mmap.PAGESIZE, 1 << 20
libc = ctypes.CDLL(None, use_errno=True)


def cold_windows(path: str) -> list:
    """Offsets of the megabyte-aligned 1 MiB windows of the file that have no page in memory."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
    base = ctypes.c_void_p()
    length = ctypes.c_ssize_t()
    ctypes.pythonapi.PyObject_AsReadBuffer(ctypes.py_object(m), ctypes.byref(base), ctypes.byref(length))
    pages = (size + PAGE - 1) // PAGE
    vec = (ctypes.c_ubyte * pages)()
    if libc.mincore(base, ctypes.c_size_t(size), vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    per = WINDOW // PAGE
    resident = bytes(v & 1 for v in vec)
    return [w * WINDOW for w in range(size // WINDOW - 1) if not any(resident[w * per : (w + 2) * per])]


spots = [(path, at) for path in args.files for at in cold_windows(path)]
whole = sum(os.path.getsize(f) for f in args.files)
print(f"{len(spots) * WINDOW / 1e9:.1f} of {whole / 1e9:.1f} GB is not in memory: reads go there", flush=True)
if len(spots) < 20000:
    raise SystemExit("too little of the files is out of memory to measure the disk")


def run(size: int, threads: int) -> float:
    counts, stop = [0] * threads, time.perf_counter() + args.seconds

    def work(k):
        fds = {}
        for path in args.files:
            fds[path] = os.open(path, os.O_RDONLY)
            fcntl.fcntl(fds[path], fcntl.F_NOCACHE, 1)
        rng = random.Random(k * 7919 + size)
        while time.perf_counter() < stop:
            path, at = rng.choice(spots)
            if len(os.pread(fds[path], size, at)) == size:
                counts[k] += 1
        for fd in fds.values():
            os.close(fd)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return sum(counts) / (time.perf_counter() - t0)


for size in args.sizes:
    for threads in args.threads:
        rate = run(size, threads)
        print(f"{size / 1e6:.2f} MB reads, {threads:2} threads: {rate:7.0f} reads/s  {rate * size / 1e9:5.2f} GB/s", flush=True)
