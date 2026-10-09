"""llama.cpp (Metal) and the MLX graph, measured in turns on the same machine.

Other programs using the GPU move both engines by a large factor, so one run of each says little. This runs them
alternately for a few rounds and records how busy the GPU was just before each run.

    uv run bench/ab_baseline.py --model <shard 1 of the GGUF> --rounds 4 --json bench/results/ab_baseline.json
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent


def gpu_busy(samples: int = 5) -> int:
    """Median GPU utilisation in percent over a few seconds, from ioreg (no root needed)."""
    seen = []
    for _ in range(samples):
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        if m:
            seen.append(int(m.group(1)))
        time.sleep(0.6)
    return round(statistics.median(seen)) if seen else -1


def clean_env():
    # the Metal debug layer validates every call and aborts llama.cpp on this model
    return {k: v for k, v in os.environ.items() if not k.startswith("MTL_DEBUG_LAYER")}


def llama(model: str, prompt: int, gen: int) -> dict:
    out = subprocess.run(["llama-bench", "-m", model, "-p", str(prompt), "-n", str(gen), "-r", "1", "-o", "jsonl"],
                         capture_output=True, text=True, env=clean_env()).stdout
    res = {}
    for line in out.splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            res["prefill_tok_s" if d["n_prompt"] else "decode_tok_s"] = round(d["avg_ts"], 1)
    return res


def mlx(prompt: int, steps: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        subprocess.run([sys.executable, str(HERE / "mlx_overhead.py"), "--windows", "1", "4", "--steps", str(steps),
                        "--chunks", str(prompt), "--json", f.name], capture_output=True, env=clean_env(), check=True)
        d = json.load(open(f.name))
    by_t = {r["T"]: r["step_ms"] for r in d["decode"]}
    return {"prefill_tok_s": d["prefill"][0]["tok_s"], "decode_tok_s": round(1e3 / by_t[1], 1),
            "step_ms_T1": by_t[1], "step_ms_T4": by_t[4]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--prompt", type=int, default=2048)
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--json")
    args = p.parse_args()

    rounds = []
    print(f"{'round':>5} {'engine':<10} {'GPU busy before':>16} {'prefill tok/s':>14} {'decode tok/s':>13} {'T=4 step ms':>12}")
    for i in range(args.rounds):
        for name, fn in (("llama.cpp", lambda: llama(args.model, args.prompt, args.gen)),
                         ("mlx", lambda: mlx(args.prompt, args.gen // 4))):
            busy = gpu_busy()
            r = {"round": i + 1, "engine": name, "gpu_busy_before": busy, **fn()}
            rounds.append(r)
            print(f"{r['round']:>5} {name:<10} {busy:>15}% {r.get('prefill_tok_s', '-'):>14} "
                  f"{r.get('decode_tok_s', '-'):>13} {r.get('step_ms_T4', '-'):>12}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rounds, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
