"""What a process holds as the kernel counts it, and memory held by another process to make a smaller machine.

A machine of N GiB is simulated here by giving the experts what such a machine leaves them (`--machine-memory`).
That measures the disk and the kernels and never the pressure: on a 128 GB machine nothing else is short of
memory, so it never swaps, and a real 32 GB Mac did (issue 6). Two things close the gap:

- `footprint(pid)`: the kernel's physical footprint of a process now and at its peak (`proc_pid_rusage`), which
  counts what MLX's own counters do not: buffers that are not arrays, Python, Metal's own.
- `Hold(gib)`: another process that holds that many GiB of random bytes, which macOS can neither compress nor
  share, for as long as a measurement runs; so the machine has what is left. It reports its own footprint every
  second, and a measurement is valid only if the hold stayed whole through it (`Hold.whole`): macOS may swap the
  hold itself out, which says the machine was under pressure and that the hold no longer stood for anything.
  Whole is judged by what of it is resident (`own()["internal"]`), not by its footprint: the footprint counts
  pages compressed or swapped out as well. The hold is wired (`mlock`) unless told not to: held and left alone it
  is the coldest memory on the machine, and macOS compresses and swaps it out before anything else, so that a
  hold of 96 GiB kept 8.9 of them resident and the machine was not smaller at all (e47).
- `own()`: this process's resident, compressed and swapped memory (`task_info`), which tells whether the
  engine itself was pushed out where a count for the whole machine cannot. A mapped file's pages (the n-gram
  table) are dropped, not compressed, so they do not show here.

    uv run python bench/memory.py --hold 8      # hold 8 GiB until stopped, its footprint every second
"""

import ctypes
import os
import re
import subprocess
import sys
import threading
import time

GIB = 1 << 30
_RUSAGE_INFO_V4 = 4
# struct rusage_info_v4 (sys/resource.h): a 16-byte uuid, then 64-bit counters; these two are the 8th and the 29th
_PHYS_FOOTPRINT, _LIFETIME_MAX_PHYS_FOOTPRINT = 7, 28


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("counters", ctypes.c_uint64 * 35)]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
_libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RusageInfoV4)]


def footprint(pid: int | None = None) -> tuple:
    """(now, peak) bytes of a process's physical footprint, as Activity Monitor's "Memory" counts it."""
    info = _RusageInfoV4()
    if _libc.proc_pid_rusage(pid or os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage")
    return info.counters[_PHYS_FOOTPRINT], info.counters[_LIFETIME_MAX_PHYS_FOOTPRINT]


class _TaskVmInfo(ctypes.Structure):
    # struct task_vm_info (mach/task_info.h): then 64-bit fields from resident_size on; room for later revisions
    _fields_ = [("virtual_size", ctypes.c_uint64), ("region_count", ctypes.c_int32), ("page_size", ctypes.c_int32),
                ("fields", ctypes.c_uint64 * 60)]


_TASK_VM_INFO = 22
_VM_FIELDS = {"resident": 0, "resident_peak": 1, "internal": 4, "internal_peak": 5, "compressed": 13,
              "compressed_peak": 14, "compressed_lifetime": 15, "footprint": 16}
_task_self = ctypes.c_uint.in_dll(_libc, "mach_task_self_")


def own() -> dict:
    """Bytes of this process: resident, anonymous and resident (internal), in the compressor or swapped out
    (compressed), the most ever there (compressed_peak) and all that ever went there (compressed_lifetime)."""
    info = _TaskVmInfo()
    count = ctypes.c_uint(ctypes.sizeof(info) // 4)
    if _libc.task_info(_task_self, _TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count)) != 0:
        raise OSError("task_info")
    return {k: info.fields[i] for k, i in _VM_FIELDS.items()}


def swapped() -> tuple:
    """(pages swapped out since boot, bytes of swap in use) for the whole machine."""
    out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    swapouts = int(re.search(r"Swapouts:\s+(\d+)", out).group(1))
    usage = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = re.search(r"used = ([\d.]+)([MG])", usage)
    return swapouts * page, int(float(used.group(1)) * (GIB if used.group(2) == "G" else 1 << 20)) if used else 0


def settle(within: float = 1e9, over: float = 60.0, most: float = 300.0) -> tuple:
    """Wait until what is free or in the file cache has moved by less than `within` bytes for `over` seconds, at
    most `most` seconds: after a hold is let go the programs it pushed out come back in, and the next run would
    start on another machine (e51: 10.7 GB apart 30 s apart). Returns (seconds waited, bytes available, steady)."""
    def available():
        out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        free = int(re.search(r"Pages free:\s+(\d+)", out).group(1))
        cached = int(re.search(r"File-backed pages:\s+(\d+)", out).group(1))
        return (free + cached) * page

    t0, seen = time.time(), []
    while True:
        seen = [(t, a) for t, a in seen if t >= time.time() - over] + [(time.time(), available())]
        if time.time() - t0 >= over and max(a for _, a in seen) - min(a for _, a in seen) < within:
            return time.time() - t0, seen[-1][1], True
        if time.time() - t0 >= most:
            return time.time() - t0, seen[-1][1], False
        time.sleep(5)


def machine_gib() -> float:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB


class Hold:
    """Hold `gib` GiB in a process of its own while the `with` block runs."""

    def __init__(self, gib: float, wired: bool = True):
        self.gib, self.wired, self.samples, self.proc = gib, wired, [], None

    def __enter__(self):
        self.proc = subprocess.Popen([sys.executable, "-I", __file__, "--hold", str(self.gib)] + ([] if self.wired else ["--unwired"]),
                                     stdout=subprocess.PIPE, text=True)
        for line in self.proc.stdout:  # it says so once every byte is written
            if line.startswith("held"):
                break
        else:
            raise RuntimeError("the hold did not start")
        threading.Thread(target=self._watch, daemon=True).start()
        return self

    def _watch(self):
        for line in self.proc.stdout:
            if line.startswith("internal"):
                self.samples.append((time.time(), int(line.split()[1])))

    def __exit__(self, *exc):
        self.alive = self.proc.poll() is None
        self.proc.terminate()
        self.proc.wait()

    @property
    def least(self) -> int:
        return min((b for _, b in self.samples), default=0)

    @property
    def whole(self) -> bool:
        """The hold lived through the measurement and none of it left memory: resident, not compressed or out."""
        return getattr(self, "alive", self.proc.poll() is None) and self.least >= 0.98 * self.gib * GIB

    def report(self) -> dict:
        return {"hold_gib": self.gib, "hold_wired": self.wired, "hold_least_gib": round(self.least / GIB, 2), "hold_samples": len(self.samples),
                "hold_whole": self.whole}


def _hold(gib: float, wired: bool = True):
    import numpy as np

    _libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    rng, kept, left = np.random.default_rng(), [], int(gib * GIB)
    while left > 0:
        n = min(left, 256 << 20)
        kept.append(np.frombuffer(rng.bytes(n), dtype=np.uint8).copy())  # random bytes: nothing to compress
        if wired and _libc.mlock(kept[-1].ctypes.data, n) != 0:  # vm.user_wire_limit bounds it
            print("not wired:", os.strerror(ctypes.get_errno()), flush=True)
            sys.exit(1)
        left -= n
    print("held", sum(a.nbytes for a in kept), flush=True)
    while True:
        print("internal", own()["internal"], flush=True)
        time.sleep(1)


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--hold":
        _hold(float(sys.argv[2]), wired="--unwired" not in sys.argv)
    else:
        print(__doc__)
