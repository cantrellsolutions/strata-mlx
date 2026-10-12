"""Experiment 37: what does a prompt of several chunks read from the disk, where experts are read from the SSD?

A chunk of a prompt uses nearly every expert of a layer, so each layer's three tensors are read whole and not
kept (paged.ExpertStore.whole). A prompt of four chunks then reads every layer four times. Before anything is
built to read a layer once for a whole prompt, this measures what there is to gain:

    1  the bytes a prompt reads, and how many of them it had read before in the same prompt;
    2  for every layer of every chunk, the wait for the disk and the time of everything else (the GPU's work
       on that layer and the mixer of the next, the wait for the router);
    3  the arrays at the peak of reading, above what is held at rest.

As a machine of some memory (`--machine-memory`, as bench/paged.py), the file dropped from the page cache first,
each prompt read twice from an empty conversation. The prompts are the first N tokens of the project's own
records. Nothing is timed with the page cache answering: the last line says how much of the file is in it.

What it is for, set before the run (the rule is applied elsewhere, to a design's estimate, not here): reading
each layer once is worth building if a prompt of 8K tokens as a 32 GB machine would be read 1.5 times as fast,
and a prompt of one chunk no slower. From this run, two estimates for a prompt of C chunks, with D the disk's
time and G everything else: no layer read twice, the disk still waited for, G + D / C; and the next layer read
while this one is computed, the longer of the two for each layer, summed.

    uv run python bench/experiments/e37_prompt_reads.py --model <shard 1> --mtp <folder> --machine-memory 32
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))
from drop_cache import in_memory  # noqa: E402
from spec_decode import tokenize  # noqa: E402

from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402
from strata_mlx.paged import SEPARATE_UPTO, WHOLE_FROM, PagedSwitchGLU  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--machine-memory", type=float, required=True, help="GiB of the machine to stand for")
    p.add_argument("--tokens", type=int, nargs="+", default=[1537, 4096, 8192, 16384])
    p.add_argument("--json")
    args = p.parse_args()

    model, _ = load(args.model, expert_memory=0.0)
    store = model.expert_store
    engine = Engine(model, load_mtp(args.mtp, model.args.text), 3, lookup=True)
    budget = store.fit(int(args.machine_memory * 2**30))
    engine.warm()
    text = "\n\n".join((ROOT / name).read_text() for name in ("ROADMAP.md", "docs/OPTIMIZATION_RECORD.md", "README.md"))
    ids = tokenize(args.model, text)
    if len(ids) < max(args.tokens):
        raise SystemExit(f"the records are {len(ids)} tokens, {max(args.tokens)} were asked for")
    once = sum(t.data.nbytes for tensors in store.layers.values() for t in tensors)
    print(f"as a machine of {args.machine_memory:g} GiB: {budget / 1e9:.1f} GB of experts in memory; every layer's "
          f"experts once are {once / 1e9:.2f} GB in {len(store.layers)} layers", flush=True)

    # every call of a layer's experts: when the layer had its router's answer, what it then waited for the disk,
    # the bytes it read and how many of them this prompt had not read yet
    calls, seen, call = [], {}, PagedSwitchGLU.__call__

    def timed(self, x, indices):
        used = np.unique(np.array(indices)).tolist()  # waits for the router, as the call itself does
        t0, read_s, read = time.perf_counter(), store.read_s, store.bytes_read
        whole = len(used) >= WHOLE_FROM and np.array(indices).size > SEPARATE_UPTO
        asked = range(self.tensors[0].shape[0]) if whole else [e for e in used if (self.layer, e) not in store.lru]
        out = call(self, x, indices)
        one = sum(t.data.nbytes // t.shape[0] for t in self.tensors)
        new = set(asked) - seen.setdefault(self.layer, set())
        seen[self.layer] |= new
        calls.append((t0, store.read_s - read_s, store.bytes_read - read, len(new) * one, whole))
        return out

    PagedSwitchGLU.__call__ = timed
    files = sorted({t.path for tensors in store.layers.values() for t in tensors})
    for path in files:
        in_memory(path, drop=True)
    rows = []
    for n in args.tokens:
        for round_ in ("first", "again"):
            engine.reset()
            mx.clear_cache()
            rest = mx.get_active_memory()
            mx.reset_peak_memory()
            store.reset_stats()
            del calls[:]
            seen.clear()
            ends = []
            t0 = time.perf_counter()
            engine.read(ids[:n], on_chunk=lambda done, total: ends.append((done, time.perf_counter())))
            t1 = time.perf_counter()
            # a layer's other time: from its router's answer to the next layer's, less its wait for the disk
            chunks, start, at = [], t0, 0
            for done, end in ends:
                mine = [c for c in calls if start <= c[0] < end]
                disk = [c[1] for c in mine]
                other = [(mine[j + 1][0] if j + 1 < len(mine) else end) - c[0] - c[1] for j, c in enumerate(mine)]
                before = (mine[0][0] - start) if mine else end - start  # the embedding, the first mixer and router
                chunks.append({"tokens": done - at, "s": round(end - start, 3),
                               "layers_read_whole": sum(c[4] for c in mine), "read_gb": round(sum(c[2] for c in mine) / 1e9, 2),
                               "disk_s": round(sum(disk), 3), "other_s": round(sum(other) + before, 3),
                               "disk_ms_a_layer": round(statistics.median(disk) * 1e3, 1),
                               "other_ms_a_layer": round(statistics.median(other) * 1e3, 1),
                               "longer_of_the_two_s": round(sum(max(d, o) for d, o in zip(disk, other)) + before, 3)})
                start, at = end, done
            total, disk_s, new = t1 - t0, store.read_s, sum(c[3] for c in calls)
            # two estimates of a read that takes each layer from the disk once: what was read again is not read
            # (at the speed measured), and then the disk still waited for, or working beside everything else
            once = disk_s * new / max(store.bytes_read, 1)
            row = {"machine_gib": args.machine_memory, "experts_gb": round(budget / 1e9, 2), "tokens": n, "round": round_,
                   "s": round(total, 2), "tok_s": round(n / total, 1), "read_gb": round(store.bytes_read / 1e9, 2),
                   "read_again_gb": round((store.bytes_read - new) / 1e9, 2),
                   "disk_s": round(disk_s, 2), "disk_share": round(disk_s / total, 2),
                   "read_gb_s": round(store.bytes_read / 1e9 / disk_s, 2) if disk_s else None,
                   "once_still_waited_for_s": round(total - disk_s + once, 2),
                   "once_beside_the_gpu_s": round(max(total - disk_s, once), 2),
                   "next_layer_ahead_s": round(sum(c["longer_of_the_two_s"] for c in chunks), 2),
                   "rest_gb": round(rest / 1e9, 2), "peak_above_rest_gb": round((mx.get_peak_memory() - rest) / 1e9, 2),
                   "chunks": chunks}
            rows.append(row)
            print(f"{n:6} tokens {round_:5}: {row['s']:6.1f} s, {row['tok_s']:5.0f} tok/s; {row['read_gb']:6.1f} GB read "
                  f"({row['read_again_gb']:.1f} of it again) at {row['read_gb_s']} GB/s, the disk {row['disk_share']:.0%} of "
                  f"the time; peak {row['peak_above_rest_gb']:.1f} GB above {row['rest_gb']:.1f}; each layer once: "
                  f"{row['once_still_waited_for_s']:.1f} s waited for, {row['once_beside_the_gpu_s']:.1f} s beside the GPU; "
                  f"as now with the next layer read ahead: {row['next_layer_ahead_s']:.1f} s", flush=True)
            for c in chunks:
                print(f"         a chunk of {c['tokens']:5} tokens: {c['s']:5.1f} s, {c['read_gb']:5.1f} GB, {c['layers_read_whole']:2} "
                      f"layers read whole; a layer waits {c['disk_ms_a_layer']:5.0f} ms for the disk and takes "
                      f"{c['other_ms_a_layer']:5.0f} ms besides (medians)", flush=True)
            time.sleep(4)
    print(f"{sum(in_memory(path) for path in files):.1f} GB of the experts' files in the page cache at the end", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
