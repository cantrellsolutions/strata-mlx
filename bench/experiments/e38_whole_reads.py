"""Experiment 38: why does a layer's experts read whole come at 4.5 GB/s from a disk that gives 6.5?

`paged.ExpertStore.whole` reads a layer's three tensors for a chunk of a prompt: a bytearray for each, 1 MiB
pieces read into it by eight threads past the page cache, then an MLX array made of it. e37 timed it at 4.3 to
4.8 GB/s; bench/ssd_reads.py gave 6.5 GB/s for the same disk with expert-sized reads at random places. Before
reads are put beside the GPU's work, this asks what the difference is made of, with no model loaded:

    1  the read itself: by size of piece, by number of threads, into a bytearray or into pages of their own,
       with `preadv` into a buffer or `pread` returning bytes, a tensor at a time or a layer's three together;
       and for each, how much of what was read the page cache holds afterwards (the note in ROADMAP.md says
       reads of 2 MB or more go through it whatever F_NOCACHE says);
    2  what `whole` does besides reading: the bytearray, which is zeroed, and the copy into an MLX array;
    3  whether MLX can take the buffer as it is (`mx.asarray`, no copy) at a layer's size: that the GPU sees the
       bytes read, and sees new ones when the buffer is read into again, which an array made once did not
       (ROADMAP.md, "Tried and closed").

The file's pages are dropped before every row, so a row is the disk's and not the page cache's.

    uv run python bench/experiments/e38_whole_reads.py --model <shard 1> [--layers 8]
"""

import argparse
import fcntl
import json
import mmap
import os
import sys
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
    print(f"{len(layers)} layers, {one / 1e9:.2f} GB of experts each in {len(layers[0])} tensors; {args.layers} layers a row", flush=True)

    def drop() -> float:
        return sum(in_memory(path, drop=True) for path in files)

    def cached() -> float:
        return sum(in_memory(path) for path in files)

    def room(size: int, kind: str):
        """A buffer of `size` bytes and what to read into: a zeroed bytearray, or pages of the process's own."""
        buf = bytearray(size) if kind == "bytearray" else mmap.mmap(-1, size)
        return buf, memoryview(buf)

    def read_layer(pool, tensors, piece: int, kind: str = "bytearray", how: str = "preadv", together: bool = False):
        """As `whole` reads: each tensor from the page before its start. Returns (buffers, seconds reading)."""
        bufs, jobs, made = [], [], 0.0
        for t in tensors:
            start = t.offset // PAGE * PAGE
            size = t.offset + t.data.nbytes - start
            t0 = time.perf_counter()
            buf, view = room(size, kind)
            made += time.perf_counter() - t0
            bufs.append((buf, view, t.offset - start, t.data.nbytes))
            jobs.append([(fds[t.path], view, at, min(piece, size - at), start + at) for at in range(0, size, piece)])

        def one_piece(job):
            fd, view, at, n, where = job
            if how == "preadv":
                return os.preadv(fd, [view[at : at + n]], where)
            data = os.pread(fd, n, where)  # bytes of its own, copied into place
            view[at : at + len(data)] = data
            return len(data)

        t0 = time.perf_counter()
        if together:
            got = sum(pool.map(one_piece, [j for js in jobs for j in js]))
        else:
            got = sum(sum(pool.map(one_piece, js)) for js in jobs)
        dt = time.perf_counter() - t0
        if got != sum(b[3] + b[2] for b in bufs):
            raise OSError("short read")
        return bufs, dt, made

    rows, turn = [], [0]

    def row(label, threads=8, **kw):
        drop()
        mine = [layers[(turn[0] + j) % len(layers)] for j in range(args.layers)]
        turn[0] += args.layers
        size = sum(t.data.nbytes for ts in mine for t in ts)
        with ThreadPoolExecutor(threads) as pool:
            t0 = time.perf_counter()
            read_s = made_s = 0.0
            for ts in mine:
                bufs, dt, made = read_layer(pool, ts, **kw)
                read_s, made_s = read_s + dt, made_s + made
                del bufs
            total = time.perf_counter() - t0
        left = cached()
        r = {"row": label, "threads": threads, **{k: v for k, v in kw.items()}, "gb": round(size / 1e9, 2),
             "read_gb_s": round(size / 1e9 / read_s, 2), "buffers_s": round(made_s, 3), "read_s": round(read_s, 3),
             "all_gb_s": round(size / 1e9 / total, 2), "in_page_cache_gb": round(left, 2)}
        rows.append(r)
        print(f"{label:<44} {r['read_gb_s']:5.2f} GB/s reading, {r['all_gb_s']:5.2f} with the buffers made; "
              f"{left:5.2f} of {r['gb']:.2f} GB in the page cache after", flush=True)

    print("-- 1  the read", flush=True)
    row("as `whole` does: 1 MiB, bytearray, 8 threads", piece=MIB)
    for piece in (256 << 10, 2 * MIB, 4 * MIB, 16 * MIB):
        row(f"pieces of {piece / MIB:g} MiB", piece=piece)
    for piece in (MIB, 4 * MIB, 16 * MIB):
        row(f"pieces of {piece / MIB:g} MiB into pages of their own", piece=piece, kind="mmap")
    for threads in (4, 16, 32):
        row(f"1 MiB, {threads} threads", threads=threads, piece=MIB)
    row("1 MiB, the three tensors together", piece=MIB, together=True)
    row("1 MiB, three together, 16 threads", threads=16, piece=MIB, together=True)
    row("1 MiB with pread (bytes, then copied)", piece=MIB, how="pread")
    row("4 MiB with pread (bytes, then copied)", piece=4 * MIB, how="pread")

    print("-- 2  what `whole` does besides reading, a layer at a time", flush=True)
    drop()
    parts = {"bytearray": 0.0, "read": 0.0, "mx.array (a copy)": 0.0, "mx.asarray (none)": 0.0}
    mine = layers[: args.layers]
    with ThreadPoolExecutor(8) as pool:
        for ts in mine:
            bufs, dt, made = read_layer(pool, ts, piece=MIB)
            parts["bytearray"] += made
            parts["read"] += dt
            t0 = time.perf_counter()
            arrays = [mx.array(np.frombuffer(view[skip:], dtype=np.uint8)) for _, view, skip, _ in bufs]
            mx.eval(arrays)
            parts["mx.array (a copy)"] += time.perf_counter() - t0
            del arrays
            t0 = time.perf_counter()
            arrays = [mx.asarray(np.frombuffer(view[skip:], dtype=np.uint8)) for _, view, skip, _ in bufs]
            mx.eval(arrays)
            parts["mx.asarray (none)"] += time.perf_counter() - t0
            del arrays, bufs
    size = sum(t.data.nbytes for ts in mine for t in ts)
    for name, s in parts.items():
        print(f"{name:<22} {s / len(mine) * 1e3:7.1f} ms a layer", flush=True)
    as_now = parts["bytearray"] + parts["read"] + parts["mx.array (a copy)"]
    print(f"a layer as now: {as_now / len(mine) * 1e3:.0f} ms, {size / 1e9 / as_now:.2f} GB/s; with pages of its own and no copy "
          f"it is the read: {size / 1e9 / parts['read']:.2f} GB/s", flush=True)
    rows.append({"row": "parts of a layer, ms", **{k: round(v / len(mine) * 1e3, 1) for k, v in parts.items()}})

    print("-- 3  a buffer taken as it is", flush=True)
    checks = {}
    with ThreadPoolExecutor(8) as pool:
        for kind in ("bytearray", "mmap"):
            a_t = layers[0][0]  # two tensors of one size, different bytes
            b_t = next(t for t in layers[1] if t.data.nbytes == a_t.data.nbytes)
            bufs, _, _ = read_layer(pool, [a_t], piece=MIB, kind=kind)
            _, view, skip, n = bufs[0]
            host = np.frombuffer(view[skip : skip + n], dtype=np.uint8)
            before = mx.get_active_memory()
            arr = mx.asarray(host)
            first = int(mx.sum(arr.astype(mx.uint64)).item())
            ok_first = first == int(host.sum(dtype=np.uint64))
            grew = mx.get_active_memory() - before
            # the same buffer read into again: does the array made before see it, and does one made now?
            start, size = b_t.offset - skip, skip + n  # so that its bytes lie where the first tensor's did
            sum(pool.map(lambda at: os.preadv(fds[b_t.path], [view[at : min(at + MIB, size)]], start + at), range(0, size, MIB)))
            want = int(np.frombuffer(view[skip : skip + n], dtype=np.uint8).sum(dtype=np.uint64))
            old_sees = int(mx.sum(arr.astype(mx.uint64)).item())
            new = mx.asarray(np.frombuffer(view[skip : skip + n], dtype=np.uint8))
            new_sees = int(mx.sum(new.astype(mx.uint64)).item())
            checks[kind] = {"mb": round(n / 1e6), "gpu_sees_what_was_read": ok_first, "active_memory_grew_mb": round(grew / 1e6),
                            "array_made_before_sees_the_new_bytes": old_sees == want and want != first,
                            "array_made_after_sees_the_new_bytes": new_sees == want}
            print(f"{kind:<10} {checks[kind]}", flush=True)
            del arr, new
    rows.append({"row": "a buffer taken as it is", **checks})
    print(f"{drop():.2f} GB of the files was in the page cache at the end, now dropped", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
