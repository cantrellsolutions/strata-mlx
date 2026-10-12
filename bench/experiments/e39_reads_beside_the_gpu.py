"""Experiment 39: a layer's experts read while the GPU works: what does each lose to the other?

The estimate for reading the next layer's experts during this layer's computation takes the two not to disturb
each other. e38 read a layer alone at 5.5 GB/s; in the engine (e37) the same read came at 4.3 to 4.8. With no
model loaded and the engine unchanged, this puts the two side by side:

    1  the read as `paged.ExpertStore.whole` does it, in a thread of its own, while the main thread keeps the
       GPU at work; each alone before and after, since this machine slows under a long load. Two kinds of work:
       products of matrices (what a layer's experts mostly are), and gathers over a large array (memory moved);
    2  whether the number of threads and the size of piece still make no difference beside the GPU;
    3  arrays made in the reading thread, by copy (`mx.array`) and of the buffer as it is (`mx.asarray`), and
       used by the main thread while its work goes on: does the GPU see the bytes that were read?

The files' pages are dropped before every row.

    uv run python bench/experiments/e39_reads_beside_the_gpu.py --model <shard 1> [--layers 8]
"""

import argparse
import fcntl
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))
from drop_cache import in_memory  # noqa: E402

from strata_mlx.check import shards  # noqa: E402

PAGE, MIB = 16384, 1 << 20


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--layers", type=int, default=8, help="layers read for each row")
    p.add_argument("--json")
    args = p.parse_args()

    by_layer = {}
    for f in shards(args.model):
        for name, t in f.tensors.items():
            if name.endswith("_exps.weight"):
                by_layer.setdefault(int(name.split(".")[1]), []).append(t)
    layers = [by_layer[i] for i in sorted(by_layer)]
    files = sorted({t.path for ts in layers for t in ts})
    fds = {}
    for path in files:
        fds[path] = os.open(path, os.O_RDONLY)
        fcntl.fcntl(fds[path], fcntl.F_NOCACHE, 1)
    one = sum(t.data.nbytes for t in layers[0])
    print(f"{len(layers)} layers, {one / 1e9:.2f} GB of experts each; {args.layers} layers a row", flush=True)

    def drop():
        for path in files:
            in_memory(path, drop=True)

    def read_layer(pool, tensors, piece):
        """As `whole` reads. Returns [(view, skip, bytes)] and the seconds it took."""
        t0, bufs = time.perf_counter(), []
        for t in tensors:
            start = t.offset // PAGE * PAGE
            buf = bytearray(t.offset + t.data.nbytes - start)
            view, fd = memoryview(buf), fds[t.path]
            got = pool.map(lambda at: os.preadv(fd, [view[at : at + piece]], start + at), range(0, len(buf), piece))
            if sum(got) != len(buf):
                raise OSError("short read")
            bufs.append((view, t.offset - start, t.data.nbytes))
        return bufs, time.perf_counter() - t0

    # -- the GPU's work: a step is one evaluation, sized to tens of milliseconds
    a = mx.random.normal((20480, 2560)).astype(mx.float16)  # a chunk's rows for the experts, as many as 2,048 tokens have
    up, down = mx.random.normal((2560, 640)).astype(mx.float16), mx.random.normal((640, 2560)).astype(mx.float16)
    big = mx.random.normal((1 << 28,)).astype(mx.float16)  # 0.5 GB
    where = mx.random.randint(0, 1 << 28, (1 << 26,))
    mx.eval(a, up, down, big, where)

    def products():
        h = a
        for _ in range(4):
            h = (h @ up) @ down * 1e-3
        mx.eval(mx.sum(h))

    def gathers():
        mx.eval(mx.sum(big[where]) + mx.sum(big[where + 1 & (1 << 28) - 1]))

    work = {"products": products, "gathers": gathers}

    def steps(fn, until) -> list:
        """Steps of the GPU's work until `until()` is true; the seconds each took."""
        took = []
        while not until():
            t0 = time.perf_counter()
            fn()
            took.append(time.perf_counter() - t0)
        return took

    def for_seconds(s):
        end = time.perf_counter() + s
        return lambda: time.perf_counter() > end

    turn = [0]

    def some_layers():
        mine = [layers[(turn[0] + j) % len(layers)] for j in range(args.layers)]
        turn[0] += args.layers
        return mine, sum(t.data.nbytes for ts in mine for t in ts)

    def read_alone(threads=8, piece=MIB) -> float:
        drop()
        mine, size = some_layers()
        with ThreadPoolExecutor(threads) as pool:
            s = sum(read_layer(pool, ts, piece)[1] for ts in mine)
        return size / 1e9 / s

    def together(fn, threads=8, piece=MIB, make=None):
        """The read in a thread, the GPU's work in this one. Returns (GB/s, the steps' seconds, checks)."""
        drop()
        mine, size = some_layers()
        out = {"s": 0.0, "arrays": [], "error": None}

        def reader():
            try:
                with ThreadPoolExecutor(threads) as pool:
                    for ts in mine:
                        bufs, dt = read_layer(pool, ts, piece)
                        out["s"] += dt
                        if make:
                            view, skip, n = bufs[0]
                            host = np.frombuffer(view[skip : skip + n], dtype=np.uint8)
                            out["arrays"].append((make(host), int(host.sum(dtype=np.uint64))))
            except Exception as e:  # noqa: BLE001
                out["error"] = repr(e)

        th = threading.Thread(target=reader)
        th.start()
        took = steps(fn, lambda: not th.is_alive())
        th.join()
        if out["error"]:
            raise RuntimeError(out["error"])
        right = [int(mx.sum(arr.astype(mx.uint64)).item()) == want for arr, want in out["arrays"]]
        return size / 1e9 / out["s"], took, right

    def ms(took):
        return float(np.median(took) * 1e3)

    rows = []
    for fn in work.values():  # warm
        steps(fn, for_seconds(1.0))

    print("-- 1  each alone, then together, then alone again", flush=True)
    for name, fn in work.items():
        for round_ in (1, 2):
            gpu0, read0 = ms(steps(fn, for_seconds(3.0))), read_alone()
            both_read, took, _ = together(fn)
            gpu1, read1 = ms(steps(fn, for_seconds(3.0))), read_alone()
            gpu_alone, read_alone_ = (gpu0 + gpu1) / 2, (read0 + read1) / 2
            r = {"row": f"{name}, round {round_}", "read_alone_gb_s": [round(read0, 2), round(read1, 2)],
                 "read_beside_gb_s": round(both_read, 2), "read_kept": round(both_read / read_alone_, 2),
                 "step_alone_ms": [round(gpu0, 1), round(gpu1, 1)], "step_beside_ms": round(ms(took), 1),
                 "step_longest_ms": round(max(took) * 1e3, 1), "gpu_kept": round(gpu_alone / ms(took), 2)}
            rows.append(r)
            print(f"{name:<9} read {read0:.2f}, {read1:.2f} GB/s alone, {both_read:.2f} beside ({r['read_kept']:.2f});  "
                  f"a step {gpu0:.1f}, {gpu1:.1f} ms alone, {ms(took):.1f} beside ({r['gpu_kept']:.2f}), longest {max(took) * 1e3:.0f}", flush=True)

    print("-- 2  beside the products: threads and size of piece", flush=True)
    fn = work["products"]
    for threads, piece in ((2, MIB), (4, MIB), (8, MIB), (16, MIB), (8, 4 * MIB), (8, 16 * MIB), (4, 4 * MIB)):
        gpu0 = ms(steps(fn, for_seconds(2.0)))
        both_read, took, _ = together(fn, threads, piece)
        r = {"row": f"{threads} threads, {piece // MIB} MiB", "read_beside_gb_s": round(both_read, 2),
             "step_alone_ms": round(gpu0, 1), "step_beside_ms": round(ms(took), 1), "gpu_kept": round(gpu0 / ms(took), 2)}
        rows.append(r)
        print(f"{threads:>2} threads, {piece // MIB:>2} MiB   read {both_read:.2f} GB/s;  a step {gpu0:.1f} ms alone, "
              f"{ms(took):.1f} beside ({r['gpu_kept']:.2f})", flush=True)

    print("-- 3  arrays made in the reading thread, used in this one", flush=True)
    for label, make in (("mx.array (a copy)", lambda h: mx.array(h)), ("mx.asarray (the buffer)", lambda h: mx.asarray(h))):
        both_read, took, right = together(fn, make=make)
        r = {"row": label, "arrays": len(right), "gpu_saw_what_was_read": sum(right), "read_beside_gb_s": round(both_read, 2),
             "step_beside_ms": round(ms(took), 1)}
        rows.append(r)
        print(f"{label:<24} {sum(right)} of {len(right)} arrays right;  read {both_read:.2f} GB/s, a step {ms(took):.1f} ms", flush=True)

    drop()
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
