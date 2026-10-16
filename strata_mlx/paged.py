"""Experts read from the SSD as tokens ask for them, for a machine whose memory does not hold them all.

A token uses ten of a layer's 512 experts, and over a conversation they are not the same few (bench/expert_use.py),
so a small memory means reading from the disk all the time, and what matters is how. Here:

  * every expert in memory is three arrays of its own (its gate, up and down tensors, as the file's blocks), in
    one store shared by all layers, the least recently used dropped first when the budget is full;
  * a layer waits for its router's choice to reach the CPU (0.2 ms), reads what is missing with a few parallel
    reads past the page cache, and hands each (token, expert) pair's arrays to the kernels as they are;
  * a prompt, whose tokens use nearly every expert of a layer, takes the layer's three tensors whole instead: what
    memory does not hold of them read in file order, joined on the GPU with what it does, and none of it kept;
  * at the start the budget is filled with the experts used most: by this machine's own answers so far, and
    before there are any, by a profile of seven conversations that comes with the engine (expert_use.npy).

Measured on the way, a token at a time on an M4 Max with nothing to read: 23 ms a pass with every expert in one
array; 34 with the wait at every layer; 42 if the ten experts are first joined into a stack, which is why the
kernels take them separately. Replacing one expert inside a 1.4 GB array copies the array (2 ms), which is why
each expert is its own array.
"""

import ctypes
import fcntl
import os
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, wait

import mlx.core as mx
import numpy as np
from mlx_lm.models.activations import swiglu

from .blocks import BLOCK, MAX_SEPARATE, BlockSwitchLinear, matvec_separate, switch_glu
from .gguf import Tensor

WHOLE_FROM = 384  # experts of a layer in one call from which its tensors are read whole
KEEP_UPTO = 640  # (token, expert) pairs in one call up to which what was read stays in memory: answers, not prompts
PAGE, PIECE = 16384, 1 << 20
SEPARATE_UPTO = 80  # pairs up to which each brings its own arrays to the kernels: a window of an answer


# What a machine's memory is set aside for before experts get the rest, in bytes: macOS and whatever else runs;
# and what this process takes beyond its weights. Measured with Q2_0: 1.3 GB that is not arrays, 2.1 GB at the peak
# of reading a 1,500-token prompt (a layer's tensors read whole, the activations), MLX's pool of 0.5.
# #6: macOS had 4.5 GB until a 32 GB MacBook Air was heard from: with nothing else open it holds 6.1 GB itself, and
# the machine swapped until the experts had 16 GB and not 20.5, which is what 8.5 GB gives them. One machine.
# A prompt of more than a chunk is read a layer at a time (bylayer.py) and holds more while it is: its chunks'
# residuals, and two layers' experts where one was, the one computed and the one read ahead. Measured (e40) at
# the peak above rest, against chunk by chunk: 3.90 GB and 2.82-3.03 for 8,192 tokens, 5.00 and 3.72 for 32,768.
# So the working memory is 5 GB where it was 4.
# A 48 GB M4 Pro was heard from next (Q2_0): with the 30.1 GB of experts 8.5 GB left it, it swapped 5 GB out as it
# started; capped at 24 it did not, and Strata's own Metal engine took 25.6 there. What macOS and the rest hold
# grows with the machine, so they are left a quarter of its memory, and 8.5 GB at least: the 32 GB Air's 8.5 is
# its quarter, and the 48 GB machine's 12.9 leaves the experts 26.1 GiB where 30.1. Two machines.
SYSTEM, SHARE, WORKING = int(8.5e9), 0.25, int(5.0e9)
DRAFT_HEAD = int(1.6e9)
# The dense tensors as MLX's arrays take this many times their size in the file: several are held at 8 bits. So a
# machine that holds the model whole has them. One that reads experts from the SSD keeps two of those types as
# the file's blocks (dense.BlockLinear), 1.7 GB less with the Q2_0 file; its store finds that out when it fits
# itself to what is loaded (`fit`), and `check.py` counts it.
DENSE_AS_ARRAYS = 1.6
# Of what `--expert-memory available` finds at start, this much is left to other programs: taking all of it, the
# process grew to 29.9 of the 31.3 GB there were and macOS compressed up to 23.3 GB of it at once (e49). Margins of
# 8, 0 and 4 GB did not settle it: what was available moved by 10.7 GB between runs 30 s apart (e51). So none yet.
AVAILABLE_MARGIN = 0
LEAST = int(0.5e9)  # of experts in memory, however little is left for them


def left_to_system(machine: int) -> int:
    """What a machine of that much memory leaves macOS and other programs, in bytes."""
    return max(SYSTEM, int(SHARE * machine))


def machine_for(rest: int) -> int:
    """The memory of the smallest machine that has `rest` bytes besides what it leaves macOS."""
    return max(rest + SYSTEM, int(rest / (1 - SHARE)) + 1)


def machine_memory() -> int:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def available_memory() -> tuple | None:
    """(free, file-backed) bytes of this machine's memory now, as `vm_stat` counts "Pages free" and
    "File-backed pages": what nothing holds, and what holds a file's pages, which macOS lets go of before it
    swaps. None where the counts cannot be read."""
    try:
        counts = [int(subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, check=True).stdout)
                  for name in ("hw.pagesize", "vm.page_free_count", "vm.page_pageable_external_count")]
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    return counts[1] * counts[0], counts[2] * counts[0]


def sizes(files: list) -> tuple:
    """(experts, dense tensors) of a model's shards, in bytes as the files have them. The dense tensors are
    those of the first shard: the n-gram table, in the second, stays on the disk."""
    experts = sum(t.data.nbytes for f in files for name, t in f.tensors.items() if name.endswith("_exps.weight"))
    dense = sum(t.data.nbytes for name, t in files[0].tensors.items() if not name.endswith("_exps.weight"))
    return experts, dense


def all_fit(machine: int, experts: int, dense: int, floats: int = 0) -> bool:
    """Whether a machine of that much memory holds the whole model, sizes as in the file; with `floats`, and the
    weights of its Q6_K layers as floats beside it, kept for prompts (dense.ScaledLinear, #3)."""
    return machine - left_to_system(machine) - WORKING >= experts + DENSE_AS_ARRAYS * dense + DRAFT_HEAD + floats


# ---- the budget following the pressure while the engine runs (`ExpertStore.follow_pressure`, opt-in)
#
# A budget set at start cannot see what other programs ask for after it, and macOS then compresses the engine:
# as a 48 GB machine with a full desktop, 30 GB of the process at once and answers at a third of their speed
# (e48); what was available at start moved by 10.7 GB between runs a minute apart (e51). Letting go of arrays
# takes their compression off nearly one for one, and macOS compresses the experts in use as much as idle ones,
# so which experts go does not matter, how many do (e52). So the store watches its own process: while more than
# a little of it is compressed, or the system says its memory is under pressure, the budget comes down by what is
# compressed, a step at a time; once nothing has been compressed for a while it grows back to what the rule gave.


class _TaskVmInfo(ctypes.Structure):
    # struct task_vm_info (mach/task_info.h): 64-bit fields from resident_size on; room for later revisions
    _fields_ = [("virtual_size", ctypes.c_uint64), ("region_count", ctypes.c_int32), ("page_size", ctypes.c_int32),
                ("fields", ctypes.c_uint64 * 60)]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
_COMPRESSED = 13  # of task_vm_info's 64-bit fields from resident_size on


def compressed_bytes() -> int | None:
    """Of this process, the bytes in the compressor or swapped out; None where they cannot be read."""
    try:
        info, count = _TaskVmInfo(), ctypes.c_uint(ctypes.sizeof(_TaskVmInfo) // 4)
        task = ctypes.c_uint.in_dll(_libc, "mach_task_self_")
        if _libc.task_info(task, 22, ctypes.byref(info), ctypes.byref(count)) != 0:  # TASK_VM_INFO
            return None
        return info.fields[_COMPRESSED]
    except (OSError, ValueError, AttributeError):
        return None


def pressure_level() -> int:
    """The system's memory pressure: 1 normal, 2 warning, 4 critical; 1 where it cannot be read."""
    value, size = ctypes.c_uint32(0), ctypes.c_size_t(4)
    if _libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(value), ctypes.byref(size), None, 0):
        return 1
    return value.value or 1


@dataclass
class Follow:
    """When and how far the budget moves, from what is read; `decide` is the whole rule, so it can be tested."""

    ceiling: int  # what the rule gave at start: never more
    every: float = 1.0  # seconds between readings
    high: int = 1 << 30  # compressed bytes that count as being under pressure, read twice in a row
    slack: int = 512 << 20  # shed this much beyond what is compressed
    most: int = 2 << 30  # the most shed at one reading
    quiet_for: float = 30.0  # seconds with nothing compressed and no pressure before growing
    grow: int = 1 << 30  # grown at a time
    last: float = float("-inf")
    highs: int = 0
    quiet_since: float | None = None
    log: list = field(default_factory=list)

    def decide(self, now: float, compressed: int, level: int, held: int, budget: int) -> int | None:
        """The new budget, or None to leave it."""
        if now - self.last < self.every:
            return None
        self.last = now
        self.highs = self.highs + 1 if compressed > self.high else 0
        if level >= 2 or self.highs >= 2:
            self.quiet_since = None
            new = max(LEAST, held - compressed - self.slack, budget - self.most)
            return new if new < budget else None
        if compressed < self.high // 4 and level <= 1:
            if self.quiet_since is None:
                self.quiet_since = now
            elif now - self.quiet_since >= self.quiet_for and budget < self.ceiling:
                self.quiet_since = now
                return min(self.ceiling, budget + self.grow)
        else:
            self.quiet_since = None
        return None


class ExpertStore:
    """The experts in memory, within a budget of bytes."""

    def __init__(self, budget: int, threads: int = 8):
        self.budget, self.held = budget, 0
        # MLX keeps the buffers of freed arrays for the next ones, without limit unless told: 4 GB after a hundred
        # tokens. Enough of them to take the experts that come and go, no more.
        mx.set_cache_limit(512 << 20)
        self.lru = OrderedDict()  # (layer, expert) -> (arrays, bytes)
        self.layers = {}  # layer -> its three tensors
        self.use = {}  # (layer, expert) -> times an answer asked for it since the start
        self.fds = {}
        self.pool = ThreadPoolExecutor(threads)
        # while a prompt goes through a layer chunk after chunk (bylayer.py): what was read of that layer's
        # experts, kept from the first chunk that reads it to the last; and the next layer's, being read
        self.kept = None  # [layer, its tensors whole or None, {expert: its arrays}]
        # (layer, what `_join` takes, the read under way); the read has threads of its own, the store's being for
        # the reads of the thread that runs the model meanwhile
        self.reader, self.ahead_pool, self.coming = ThreadPoolExecutor(1), ThreadPoolExecutor(threads), None
        self.counting = threading.Lock()  # bytes_read and read_s: a read ahead counts on its own thread
        self.follow = None  # a Follow while the budget follows the pressure
        self.reset_stats()

    def fit(self, machine: int, available: int | None = None, margin: int | None = None) -> int:
        """Set the budget to what a machine of that much memory has left, everything else being loaded now.

        With `available`, the bytes of memory that were free or held a file's pages before anything was
        loaded, the budget is no more than that less what is loaded and the working memory: the rule leaves
        macOS and other programs a constant, and a machine on which they hold more than that swaps (#6: 11.5 GB
        free at the start where another run had 24.5, and 16.8 GB swapped out with the same 16.5 GB of experts).
        Counted before loading, not now: loading reads the model's files, and their pages, 9 GB of them on
        that machine, are file cache for the next twenty seconds. `margin` bytes of what was available are left to
        what other programs will ask for after the start (AVAILABLE_MARGIN)."""
        loaded = mx.get_active_memory() - self.held
        self.budget = machine - left_to_system(machine) - WORKING - loaded
        if available is not None:
            self.budget = min(self.budget, available - loaded - WORKING - (AVAILABLE_MARGIN if margin is None else margin))
        self.budget = max(LEAST, self.budget)
        return self.budget

    def follow_pressure(self, on: bool = True, **rule) -> None:
        """From now on the budget follows the pressure, up to what it is now (`Follow`)."""
        self.follow = Follow(ceiling=self.budget, **rule) if on else None

    def shrink_to(self, budget: int) -> None:
        """The budget set to `budget` and the least recently used experts let go until it holds. (The rule that
        asks for it keeps to the least; this does as it is told.)"""
        self.budget = max(0, budget)
        while self.held > self.budget and self.lru:
            _, (_, size) = self.lru.popitem(last=False)
            self.held -= size
        mx.clear_cache()  # the pool keeps freed buffers otherwise, and they are what is compressed

    def _follow(self):
        f = self.follow
        now = time.monotonic()
        if now - f.last < f.every:
            return
        compressed = compressed_bytes()
        if compressed is None:
            return
        level = pressure_level()
        new = f.decide(now, compressed, level, self.held, self.budget)
        if new is not None:
            f.log.append({"t": round(now, 1), "compressed_gb": round(compressed / 1e9, 2), "level": level,
                          "from_gb": round(self.budget / 1e9, 2), "to_gb": round(new / 1e9, 2)})
            if new < self.budget:
                self.shrink_to(new)
            else:
                self.budget = new

    def reset_stats(self):
        self.hits = self.misses = self.bytes_read = 0
        self.read_s = 0.0
        self.waited_s = 0.0  # of the reads ahead, what the prompt still waited for

    def _fd(self, path: str) -> int:
        if path not in self.fds:
            self.fds[path] = os.open(path, os.O_RDONLY)
            fcntl.fcntl(self.fds[path], fcntl.F_NOCACHE, 1)  # the page cache is memory this machine does not have
        return self.fds[path]

    def _read(self, t: Tensor, first: int, count: int) -> mx.array:
        one = t.data.nbytes // t.shape[0]
        fd = self._fd(t.path)
        data = os.pread(fd, count * one, t.offset + first * one)
        if len(data) != count * one:
            raise OSError(f"{t.name}: short read")
        return mx.array(np.frombuffer(data, dtype=np.uint8))

    def hold(self, layer: int):
        """From here to `release` or the next `hold`, what a prompt's chunks read of this layer's experts is
        read once: the layer whole if it was asked for ahead or a chunk uses that much of it, else each expert
        the first time a chunk uses it."""
        stacks = None
        if self.coming is not None:
            ahead, made, read = self.coming
            self.coming = None
            t0 = time.perf_counter()
            if ahead == layer:
                read.result()  # a read that failed fails here
                stacks = self._join(*made)
            else:
                read.exception()  # waited for: nothing of it is used
            self.waited_s += time.perf_counter() - t0
        self.kept = [layer, stacks, {}]

    def ahead(self, layer: int):
        """Begin reading a layer's experts that are not in memory in a thread of its own, for the `hold` of that
        layer to take with those that are. The arrays they are read into are made here; the thread calls
        nothing of MLX."""
        if self.coming is not None and self.coming[0] == layer:
            return
        self.release_ahead()
        tensors = self.layers[layer]
        runs, held, dst, into = self._start(layer, tensors)
        self.coming = (layer, (runs, held, dst), self.reader.submit(self._fill, tensors, runs, into, self.ahead_pool))

    def release_ahead(self):
        if self.coming is not None:
            self.coming[2].exception()  # waited for: nothing of it is used
            self.coming = None

    def release(self):
        self.kept = None
        self.release_ahead()

    def whole(self, tensors: tuple, layer: int | None = None) -> list:
        """A layer's tensors, all experts: those not in memory read, and kept only while the layer is held."""
        if self.kept is None or self.kept[0] != layer:
            return self._layer(layer, tensors)
        if self.kept[1] is None:
            self.kept[1] = self._layer(layer, tensors)
            self.kept[2] = {}  # what was read one by one is all in there
        return self.kept[1]

    def _held(self, layer: int, tensors: tuple, experts: list) -> list:
        _, stacks, each = self.kept
        if stacks is not None:
            ones = [t.data.nbytes // t.shape[0] for t in tensors]
            return [tuple(s[e * one : (e + 1) * one] for s, one in zip(stacks, ones)) for e in experts]
        missing = [e for e in experts if e not in each]
        each.update(zip(missing, self._get(layer, tensors, missing, False)))
        return [each[e] for e in experts]

    def _layer(self, layer: int | None, tensors: tuple) -> list:
        """A layer's tensors, all experts, not kept: those in memory taken from there, the others read."""
        runs, held, dst, into = self._start(layer, tensors)
        self._fill(tensors, runs, into, self.pool)
        return self._join(runs, held, dst)

    def _start(self, layer: int | None, tensors: tuple) -> tuple:
        """What reading a layer's experts that are not in memory needs, made here, on the thread that runs MLX:
        the layer's experts in file order as runs (first, count, in memory), the arrays of those in memory as
        they are now (a token's chunk may let them go before the read is taken), and for each tensor an array
        for each run of the others, with its memory as numpy sees it."""
        runs = []
        for e in range(tensors[0].shape[0]):
            held = (layer, e) in self.lru
            if runs and runs[-1][2] == held:
                runs[-1][1] += 1
            else:
                runs.append([e, 1, held])
        runs = [tuple(r) for r in runs]
        held = {e: self.lru[(layer, e)][0] for first, count, h in runs if h for e in range(first, first + count)}
        dst = [[mx.zeros((count * (t.data.nbytes // t.shape[0]),), dtype=mx.uint8) for first, count, h in runs if not h]
               for t in tensors]
        mx.eval(dst)
        return runs, held, dst, [[np.array(a, copy=False) for a in arrays] for arrays in dst]

    def _fill(self, tensors: tuple, runs: list, into: list, pool: ThreadPoolExecutor):
        """Read the bytes of each run of experts not in memory into the arrays whose memory is `into`, where they
        lie. On the thread of a read ahead it calls nothing of MLX, and nothing reads the arrays until it is done.

        In pieces of 1 MiB from a page of the file on: read so, nothing is left in the page cache (e38: nor at 16
        MiB a piece; it is `os.pread`'s bytes of its own that go through the cache). A run that begins inside a
        page has its first piece read beside, from the page's start, and its part copied.
        """
        t0 = time.perf_counter()
        jobs, heads, size = [], [], 0
        missing = [(first, count) for first, count, held in runs if not held]
        for t, mems in zip(tensors, into):
            fd, one = self._fd(t.path), t.data.nbytes // t.shape[0]
            for (first, count), mem in zip(missing, mems):
                view = memoryview(mem)
                begin, end = t.offset + first * one, t.offset + (first + count) * one
                start = begin // PAGE * PAGE
                head = bytearray(min(start + PIECE, end) - start)
                heads.append((head, begin - start, view[: len(head) - (begin - start)]))
                jobs.append((fd, memoryview(head), start))
                jobs += [(fd, view[at - begin : min(at + PIECE, end) - begin], at) for at in range(start + PIECE, end, PIECE)]
                size += count * one
        got = [pool.submit(os.preadv, fd, [view], at) for fd, view, at in jobs]
        wait(got)  # every read ended, a failed one too, before this does: nothing writes into the arrays after it
        if any(f.result() != len(job[1]) for f, job in zip(got, jobs)):
            raise OSError(f"{tensors[0].name}: short read")
        for head, skip, part in heads:
            part[:] = memoryview(head)[skip:]
        with self.counting:
            self.bytes_read += size
            self.read_s += time.perf_counter() - t0

    @staticmethod
    def _join(runs: list, held: dict, dst: list) -> list:
        """Each tensor whole: the experts in memory and the arrays read, in file order, joined on the GPU."""
        out = []
        for j, arrays in enumerate(dst):
            arrays, pieces = iter(arrays), []
            for first, count, h in runs:
                pieces += [held[e][j] for e in range(first, first + count)] if h else [next(arrays)]
            out.append(pieces[0] if len(pieces) == 1 else mx.concatenate(pieces))
        return out

    def clear(self):
        """Let go of every expert and of the counts of their use."""
        self.lru.clear()
        self.use.clear()
        self.held = 0
        self.release()

    def preload(self, counts: np.ndarray, group: int = 512) -> int:
        """Fill what is free of the budget with the experts `counts` (layers, experts) says are used most.
        Returns how many were read. The most used are the last the store will drop."""
        if not self.layers or counts.ndim != 2 or counts.shape[0] <= max(self.layers) \
                or any(t[0].shape[0] != counts.shape[1] for t in self.layers.values()):
            return 0
        size = {layer: sum(t.data.nbytes // t.shape[0] for t in tensors) for layer, tensors in self.layers.items()}
        order = sorted(((int(counts[layer, e]), layer, e) for layer in self.layers for e in range(counts.shape[1])
                        if counts[layer, e] > 0 and (layer, e) not in self.lru), reverse=True)
        chosen, room = [], self.budget - self.held
        for _, layer, e in order:
            if size[layer] > room:
                break
            chosen.append((layer, e))
            room -= size[layer]
        chosen.reverse()  # the least used first: the first to go
        for a in range(0, len(chosen), group):
            part = chosen[a : a + group]
            jobs = [(e, t) for layer, e in part for t in self.layers[layer]]
            arrays = list(self.pool.map(lambda job: self._read(job[1], job[0], 1), jobs))
            for i, (layer, e) in enumerate(part):
                self.lru[(layer, e)] = (tuple(arrays[i * 3 : i * 3 + 3]), size[layer])
                self.held += size[layer]
        return len(chosen)

    def profile(self, before: np.ndarray | None = None) -> np.ndarray | None:
        """How often each expert was used, (layers, experts): this run's count, and half of what was counted
        before it, so that what the machine is used for now weighs more than what it was used for once."""
        if not self.layers:
            return None
        shape = (max(self.layers) + 1, next(iter(self.layers.values()))[0].shape[0])
        out = np.zeros(shape, dtype=np.uint32)
        if before is not None and before.shape == shape:
            out += before.astype(np.uint32) // 2
        for (layer, e), n in self.use.items():
            out[layer, e] += n
        return out

    def get(self, layer: int, tensors: tuple, experts: list, keep: bool = True) -> list:
        """The arrays of each of `experts`, read if they are not in memory."""
        if not keep and self.kept is not None and self.kept[0] == layer:
            return self._held(layer, tensors, experts)
        return self._get(layer, tensors, experts, keep)

    def _get(self, layer: int, tensors: tuple, experts: list, keep: bool) -> list:
        if self.follow is not None and self.kept is None:  # not while a prompt holds a layer whole
            self._follow()
        if keep:  # an answer's tokens, not a prompt's
            for e in experts:
                self.use[(layer, e)] = self.use.get((layer, e), 0) + 1
        missing = [e for e in experts if (layer, e) not in self.lru]
        self.hits += len(experts) - len(missing)
        self.misses += len(missing)
        got = {}
        if missing:
            t0 = time.perf_counter()
            jobs = [(e, t) for e in missing for t in tensors]
            arrays = list(self.pool.map(lambda job: self._read(job[1], job[0], 1), jobs))
            with self.counting:  # a read ahead may be counting on its thread meanwhile
                self.read_s += time.perf_counter() - t0
            n = len(tensors)
            for i, e in enumerate(missing):
                got[e] = tuple(arrays[i * n : (i + 1) * n])
                size = sum(a.nbytes for a in got[e])
                with self.counting:
                    self.bytes_read += size
                if keep:
                    self.lru[(layer, e)] = (got[e], size)
                    self.held += size
        out = []
        for e in experts:
            if e in got:
                out.append(got[e])
            else:
                self.lru.move_to_end((layer, e))
                out.append(self.lru[(layer, e)][0])
        # what this call uses is at the recent end; drop from the other one
        while self.held > self.budget and len(self.lru) > len(experts):
            _, (_, size) = self.lru.popitem(last=False)
            self.held -= size
        return out


class PagedSwitchGLU:
    """A layer's routed experts, called like mlx-lm's `SwitchGLU`, their weights in an `ExpertStore`."""

    def __init__(self, layer: int, store: ExpertStore, gate: Tensor, up: Tensor, down: Tensor):
        for t in (gate, up, down):
            if t.type not in BLOCK:
                raise ValueError(f"{t.name}: {t.type} experts are not supported")
        self.layer, self.store, self.tensors = layer, store, (up, gate, down)
        store.layers[layer] = self.tensors

    def _linear(self, blocks: mx.array, t: Tensor, experts: int) -> BlockSwitchLinear:
        lin = BlockSwitchLinear(blocks, t.type, (experts, *t.shape[1:]))
        lin.unpack_from = 1 << 62  # never unpacked to floats: 1.7 GB for a moment is more than there may be
        return lin

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        idx = np.array(indices)  # the layer waits here for its router
        used, local = np.unique(idx, return_inverse=True)
        if idx.size <= SEPARATE_UPTO:
            return self._separately(x, idx, used)
        if len(used) >= WHOLE_FROM:
            stacks, n = self.store.whole(self.tensors, self.layer), self.tensors[0].shape[0]
        else:
            each = self.store.get(self.layer, self.tensors, used.tolist(), keep=idx.size <= KEEP_UPTO)
            stacks, n = [mx.concatenate(col) if len(col) > 1 else col[0] for col in zip(*each)], len(used)
            indices = mx.array(local.reshape(idx.shape).astype(np.uint32))
        up, gate, down = (self._linear(b, t, n) for b, t in zip(stacks, self.tensors))

        return switch_glu(x, indices, up, gate, down)

    def _separately(self, x: mx.array, idx: np.ndarray, used: np.ndarray) -> mx.array:
        """A few tokens: no stack is built, the kernels read each pair's expert where it lies."""
        k = idx.shape[-1]
        by = dict(zip(used.tolist(), self.store.get(self.layer, self.tensors, used.tolist())))
        pairs = [by[e] for e in idx.reshape(-1).tolist()]
        (_, hidden, wide), kinds = self.tensors[0].shape, [t.type for t in self.tensors]
        x = x.reshape(-1, wide)
        tokens = max(1, MAX_SEPARATE // k)  # as many as one kernel can bind
        out = []
        for t in range(0, len(x), tokens):
            ups, gates, downs = zip(*pairs[t * k : (t + tokens) * k])
            xs = x[t : t + tokens]
            h = swiglu(matvec_separate(gates, kinds[1], hidden, wide, xs, k),
                       matvec_separate(ups, kinds[0], hidden, wide, xs, k))
            out.append(matvec_separate(downs, kinds[2], wide, hidden, h, 1))
        out = mx.concatenate(out) if len(out) > 1 else out[0]
        return out.reshape(*idx.shape, wide)


SEED = os.path.join(os.path.dirname(__file__), "expert_use.npy")  # bench/expert_use.py --counts, with Q2_0


def load_profile(path: str | None) -> np.ndarray | None:
    """The counts in `path`, or the ones that come with the engine if there is no such file or it cannot be read."""
    for p in (path, SEED):
        try:
            if p and os.path.exists(p):
                return np.load(p)
        except (OSError, ValueError):
            pass
    return None


def save_profile(path: str, counts: np.ndarray) -> None:
    """Beside its place, then renamed: a start never finds half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, counts)
    os.replace(tmp, path)
