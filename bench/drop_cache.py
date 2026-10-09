"""Ask the OS to forget its cached pages of the files given, and say how much of each is in memory before and
after. For measurements that are meant to read from the disk: a machine with a lot of memory otherwise answers
them from its page cache, F_NOCACHE or not."""

import ctypes
import mmap
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)
MS_INVALIDATE = 0x0002


def in_memory(path: str, drop: bool = False) -> float:
    """GB of the file's pages in memory, after dropping them if asked."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
    base, length = ctypes.c_void_p(), ctypes.c_ssize_t()
    ctypes.pythonapi.PyObject_AsReadBuffer(ctypes.py_object(m), ctypes.byref(base), ctypes.byref(length))
    if drop and libc.msync(base, ctypes.c_size_t(size), MS_INVALIDATE) != 0:
        raise OSError(ctypes.get_errno(), "msync")
    vec = (ctypes.c_ubyte * ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()
    if libc.mincore(base, ctypes.c_size_t(size), vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    return sum(v & 1 for v in vec) * mmap.PAGESIZE / 1e9


if __name__ == "__main__":
    for path in sys.argv[1:]:
        before = in_memory(path)
        print(f"{os.path.basename(path)}: {before:.1f} GB in memory, {in_memory(path, drop=True):.1f} after")
