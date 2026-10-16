"""Experiment 52: under pressure, does letting go of the least recently used arrays take compression off a process?

Before the expert store follows the pressure at run time, the lever it would pull: macOS compresses a process's
pages by their age, and arrays that only the GPU reads may not age as the CPU's do. So: arrays the size of a
store's experts, a wired hold that leaves the machine some gigabytes short, half of the arrays read by the GPU
every second and half left alone; once the process is compressed, the idle half is let go, MLX's cache emptied,
and the process watched again.

    uv run python bench/experiments/e52_shed_compressed.py --arrays-gb 45 --short-gb 25   # the run in e52
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory import Hold, own  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.paged import available_memory  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--arrays-gb", type=float, default=20)
p.add_argument("--short-gb", type=float, default=15, help="how far short of memory the hold leaves the machine")
p.add_argument("--piece-mib", type=int, default=10)
p.add_argument("--wait", type=float, default=90, help="seconds under pressure before letting go")
p.add_argument("--after", type=float, default=40, help="seconds watched after letting go")
p.add_argument("--json")
args = p.parse_args()

GB = 1e9
import subprocess  # noqa: E402

import numpy as np  # noqa: E402

limit = int(subprocess.run(["/usr/sbin/sysctl", "-n", "vm.user_wire_limit"], capture_output=True, text=True).stdout)
vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
wired = int(vm.split("Pages wired down:")[1].split(".")[0]) * 16384
free, cached = available_memory()
# the hold first, as much as the machine may still wire, so that the arrays made after it are what is short
hold_gib = min(free + cached - args.arrays_gb * GB + args.short_gb * GB, limit - wired - 2 * GB) / 2**30
rows = []


def sample(phase):
    mine = own()
    row = {"t": round(time.time() - t0, 1), "phase": phase, **{k: round(v / GB, 2) for k, v in mine.items()}}
    rows.append(row)
    print(f"{row['t']:6.1f}s {phase:7} footprint {row['footprint']:6.2f}  resident {row['resident']:6.2f}  "
          f"compressed {row['compressed']:6.2f} GB", flush=True)


print(f"{free / GB:.1f} GB free and {cached / GB:.1f} of file cache; holding {hold_gib:.1f} GiB wired, then "
      f"{args.arrays_gb} GB of arrays as the store makes them (read by the CPU, copied into MLX)", flush=True)
t0 = time.time()
with Hold(hold_gib) as h:
    sample("held")
    n = int(args.arrays_gb * GB / (args.piece_mib << 20))
    rng, arrays = np.random.default_rng(), []
    for i in range(n):  # as ExpertStore._read: bytes from the CPU, an MLX array of them
        arrays.append(mx.array(np.frombuffer(rng.bytes(args.piece_mib << 20), dtype=np.uint8)))
        if i % 200 == 199:
            mx.eval(arrays[-200:])
    mx.eval(arrays)
    warm, cold = arrays[: n // 2], arrays[n // 2 :]
    sample("made")
    while time.time() - t0 < args.wait:
        mx.eval([a[::4096].sum() for a in warm])  # the half in use, read by the GPU
        time.sleep(1)
        sample("before")
    del cold, arrays
    mx.clear_cache()
    sample("let go")
    t1 = time.time()
    while time.time() - t1 < args.after:
        mx.eval([a[::4096].sum() for a in warm])
        time.sleep(1)
        sample("after")
print(f"the hold: {h.report()}", flush=True)
before = [r for r in rows if r["phase"] == "before"][-1]
after = rows[-1]
print(f"compressed {before['compressed']} -> {after['compressed']} GB, footprint {before['footprint']} -> "
      f"{after['footprint']} GB", flush=True)
if args.json:
    Path(args.json).write_text(json.dumps({"args": vars(args), "hold": h.report(), "rows": rows}, indent=1) + "\n")
