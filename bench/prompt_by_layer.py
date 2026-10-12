"""A long prompt read through the engine's line protocol, chunk by chunk and a layer at a time, as a machine of
some memory: what a server that starts this engine would see.

For each way the engine is started as its own process (`--machine-memory`, `--prompt-by-layer on|off`,
`--read-ahead on|off`) and asked for a few different prompts of the same length, none of which continues
another. Before each, the model's files are dropped from the page cache (`--drop all`): the experts' shard, and
the other one, which is the n-gram table (28.8 GB with Q2_0) that the engine leaves on the disk and reads by
page faults. `--drop experts` leaves that table in the cache, as a machine with memory to spare has it and a
32 GB one has not; `--drop none` leaves everything. The ways are taken in turns, `--rounds` times. The engine
is given a profile of its own, so the one this machine's answers have written is neither read nor changed.

    uv run python bench/prompt_by_layer.py --model <shard 1> --mtp <folder> [--machine-memory 32] [--tokens 8192]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

WAYS = {"chunks": ["--prompt-by-layer", "off"], "layers": ["--read-ahead", "off"], "ahead": []}


def ask(engine, line: str) -> list:
    engine.stdin.write(line + "\n")
    engine.stdin.flush()
    out = []
    while True:
        got = engine.stdout.readline()
        if not got:
            raise RuntimeError("the engine ended")
        out.append(got.strip())
        if got.startswith(("DONE", "ERR")):
            return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp")
    p.add_argument("--machine-memory", type=float, default=32)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--prompts", type=int, default=3, help="different prompts asked of each engine started")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--ways", nargs="*", default=list(WAYS), choices=list(WAYS))
    p.add_argument("--drop", choices=["all", "experts", "none"], default="all", help="what leaves the page cache before a prompt")
    p.add_argument("--json")
    args = p.parse_args()
    from drop_cache import in_memory
    from spec_decode import tokenize

    from strata_mlx.check import shards

    files = sorted({t.path for f in shards(args.model) for name, t in f.tensors.items()
                    if args.drop == "all" or name.endswith("_exps.weight")})
    profile = os.path.join(tempfile.mkdtemp(prefix="strata-mlx-bench-"), "expert-use.npy")  # not there: the shipped one
    text = "\n\n".join((ROOT / name).read_text() for name in ("ROADMAP.md", "docs/OPTIMIZATION_RECORD.md", "README.md"))
    ids = tokenize(args.model, text)
    while len(ids) < args.tokens + 997 * args.prompts:
        ids = ids + ids[1:]
    prompts = [ids[997 * j : 997 * j + args.tokens] for j in range(args.prompts)]  # no one begins as another
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_DEBUG_LAYER")}
    rows = []
    for round_ in range(args.rounds):
        for way in args.ways:
            cmd = [str(ROOT / "strata-mlx"), "--serve", "--native", args.model, "--machine-memory", str(args.machine_memory),
                   "--expert-profile", profile, *(["--mtp", args.mtp] if args.mtp else []), *WAYS[way]]
            engine = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True, env=env)
            try:
                while not engine.stdout.readline().startswith("READY"):
                    pass
                for j, prompt in enumerate(prompts):
                    if os.path.exists(profile):
                        os.remove(profile)  # what the request before wrote: every prompt starts from the shipped one
                    cached = sum(in_memory(path, drop=args.drop != "none") for path in files)
                    lines = ask(engine, f"GEN 8 {','.join(map(str, prompt))}")
                    done = lines[-1].split()
                    if done[0] != "DONE" or lines[0] != "RESUME 0":
                        raise RuntimeError(f"{lines[0]} ... {lines[-1]}")
                    s = float(done[3]) / 1e3
                    rows.append({"way": way, "round": round_ + 1, "prompt": j + 1, "tokens": args.tokens, "s": round(s, 2),
                                 "tok_s": round(args.tokens / s, 1), "lines_of_progress": sum(l.startswith("PP ") for l in lines),
                                 "in_page_cache_gb": round(cached, 1), "answer": [l.split()[1] for l in lines if l.startswith("T ")]})
                    print(f"  {way:<7} round {round_ + 1}, prompt {j + 1}: {s:6.1f} s {args.tokens / s:6.0f} tok/s  "
                          f"({rows[-1]['lines_of_progress']} lines of progress, {cached:.1f} GB of the files in the page cache)", flush=True)
            finally:
                engine.stdin.close()
                engine.wait(timeout=60)
    for j in range(args.prompts):
        answers = {tuple(r["answer"]) for r in rows if r["prompt"] == j + 1}
        print(f"  prompt {j + 1}: {'the same answer every time' if len(answers) == 1 else 'ANSWERS DIFFER'}")
    med = {w: statistics.median(r["s"] for r in rows if r["way"] == w) for w in args.ways}
    for w in args.ways:
        mine = [r["s"] for r in rows if r["way"] == w]
        print(f"  {w:<7} median {med[w]:6.1f} s ({min(mine):.1f}-{max(mine):.1f}), {args.tokens / med[w]:5.0f} tok/s"
              + (f", {med['chunks'] / med[w]:.2f} of chunk by chunk" if "chunks" in med and w != "chunks" else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
