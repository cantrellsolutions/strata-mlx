"""Tokens a second of this engine against llama.cpp's server and LM Studio: one file, the same prompts.

Each engine in turn loads the model alone, answers the four prompts of `spec_decode.py` (256 tokens each, the
most likely token every time) and reads a prompt of 1,180 tokens, and is stopped again; then the engines
once more in the opposite order, since this machine slows down under load (see quiet.py) and whoever runs
first has it coolest. Every prompt goes in as text with its chat markers, so no engine's template is in play,
and every engine reports its own timings: prompt and answer apart, nothing of HTTP in them.

    llama.cpp     `llama-server` as installed (Homebrew), all layers on the GPU, one slot
    LM Studio     its llama.cpp runtime, the model made known to it by symbolic links (`--lms-key`), one slot
    strata-mlx    `--spec 1` (a token a pass, no draft head) and as it serves by default

Ollama is not here: its GGUF reader (0.35.1) refuses both files at the Q2_0 tensors, which it does not know.

    uv run python bench/compare_servers.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp --lms-key <key> --json out.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spec_decode import EDIT_ME, PROMPTS, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENV = {k: v for k, v in os.environ.items() if not k.startswith("MTL_DEBUG_LAYER")}  # it crashes llama.cpp on this model
REST = 8.0  # seconds between requests: the GPU is back at speed after about ten


def chat(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def long_prompt(n: int) -> str:
    """1,180 tokens, and different from the first token on each time, so that nothing of it is reused."""
    return chat(f"Review number {n}. Say in two sentences what this module is for.\n\n```python\n" + EDIT_ME * 4 + "```")


def post(url: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class LlamaCpp:
    name = "llama.cpp (llama-server)"

    def __init__(self, args):
        self.url = f"http://127.0.0.1:{args.port}"
        self.log = open(args.log_dir / "llama-server.log", "w")
        self.p = subprocess.Popen(["llama-server", "-m", args.model, "-c", "8192", "-ngl", "999", "-np", "1",
                                   "--port", str(args.port)], stdout=self.log, stderr=subprocess.STDOUT, env=ENV)
        for _ in range(900):
            try:
                if json.loads(urllib.request.urlopen(self.url + "/health", timeout=2).read()).get("status") == "ok":
                    return
            except OSError:
                pass
            if self.p.poll() is not None:
                raise RuntimeError("llama-server ended while loading; see its log")
            time.sleep(1)
        raise RuntimeError("llama-server did not come up")

    def run(self, text: str, tokens: int) -> dict:
        t = post(self.url + "/completion", {"prompt": text, "n_predict": tokens, "temperature": 0, "cache_prompt": False})["timings"]
        return {"prompt_tokens": t["prompt_n"], "prompt_tps": t["prompt_per_second"], "tokens": t["predicted_n"],
                "tps": t["predicted_per_second"]}

    def close(self):
        self.p.terminate()
        self.p.wait(timeout=60)
        self.log.close()


class LMStudio:
    name = "LM Studio (llama.cpp runtime)"

    def __init__(self, args):
        self.id, self.url = "strata-bench", "http://127.0.0.1:1234/api/v0/completions"
        subprocess.run(["lms", "unload", self.id], capture_output=True)
        out = subprocess.run(["lms", "load", args.lms_key, "--gpu", "max", "-c", "8192", "--parallel", "1",
                              "--identifier", self.id, "-y"], capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError("lms load: " + (out.stderr or out.stdout)[-300:])

    def run(self, text: str, tokens: int) -> dict:
        d = post(self.url, {"model": self.id, "prompt": text, "max_tokens": tokens, "temperature": 0, "stream": False})
        s, u = d["stats"], d["usage"]
        self.runtime = d["runtime"]
        return {"prompt_tokens": u["prompt_tokens"], "prompt_tps": u["prompt_tokens"] / s["time_to_first_token"],
                "tokens": u["completion_tokens"], "tps": s["tokens_per_second"]}

    def close(self):
        subprocess.run(["lms", "unload", self.id], capture_output=True)


class Strata:
    def __init__(self, args, head: bool):
        self.name = "strata-mlx" + ("" if head else ", a token a pass (--spec 1)")
        self.model = args.model
        self.log = open(args.log_dir / "strata-mlx.log", "w")
        self.p = subprocess.Popen([str(ROOT / "strata-mlx"), "--serve", "--native", args.model, "--mtp", args.mtp,
                                   *([] if head else ["--spec", "1"]), *args.engine_args.split()],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, env=ENV)
        self.info = next(l for l in self._until("READY") if l.startswith("INFO"))

    def _until(self, prefix: str) -> list:
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("strata-mlx ended; see its log")
            lines.append(line.rstrip())
            if line.startswith(prefix):
                return lines
            if line.startswith(("ERR", "FATAL")):
                raise RuntimeError(line)

    def run(self, text: str, tokens: int) -> dict:
        ids = tokenize(self.model, text)
        self.p.stdin.write(f"GEN {tokens} " + ",".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        done = self._until("DONE")[-1].split()
        n, read = int(done[1]), int(done[14])
        return {"prompt_tokens": len(ids), "prompt_tps": read / float(done[3]) * 1e3, "tokens": n, "tps": n / float(done[4]) * 1e3,
                "guesses_kept": int(done[6]), "guesses_made": int(done[7])}

    def close(self):
        self.p.stdin.write("QUIT\n")
        self.p.stdin.flush()
        self.p.wait(timeout=120)
        self.log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True)
    p.add_argument("--lms-key", help="the model's key in `lms ls`; without it LM Studio is left out")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--log-dir", type=Path, default=Path("."))
    p.add_argument("--only", nargs="*", default=[], help="engines to run, by a word of their name")
    p.add_argument("--engine-args", default="", help="more arguments for this engine, in one string: "
                                                     "'--machine-memory 16' runs it as a Mac of 16 GB would")
    p.add_argument("--json")
    args = p.parse_args()
    makers = [("llama.cpp", lambda: LlamaCpp(args))] + ([("LM Studio", lambda: LMStudio(args))] if args.lms_key else []) + [
        ("strata-mlx a token a pass", lambda: Strata(args, False)), ("strata-mlx default", lambda: Strata(args, True))]
    makers = [make for label, make in makers if not args.only or any(w in label for w in args.only)]
    rows, n = [], 0
    for order in (makers, makers[::-1]):
        for make in order:
            t0 = time.perf_counter()
            engine = make()
            print(f"{engine.name}: up in {time.perf_counter() - t0:.0f} s", flush=True)
            if isinstance(engine, Strata) and args.engine_args:
                print(f"  {engine.info}", flush=True)
            try:
                engine.run(chat("Say hi."), 8)  # the first request pays for what is compiled once
                for name, text in PROMPTS.items():
                    time.sleep(REST)
                    r = engine.run(chat(text), args.tokens)
                    rows.append({"engine": engine.name, "prompt": name, **r})
                    print(f"  {name:10} {r['tps']:6.1f} tok/s  ({r['tokens']} tokens)", flush=True)
                time.sleep(REST)
                n += 1
                r = engine.run(long_prompt(n), 16)
                rows.append({"engine": engine.name, "prompt": "reading", **r})
                print(f"  reading    {r['prompt_tps']:6.0f} tok/s  ({r['prompt_tokens']} tokens of prompt)", flush=True)
            finally:
                engine.close()
            time.sleep(2 * REST)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
    print()
    names = list(dict.fromkeys(r["engine"] for r in rows))
    prompts = [*PROMPTS, "reading"]
    print(f"{'':42}" + "".join(f"{q:>14}" for q in prompts))
    for e in names:
        cells = []
        for q in prompts:
            v = [r["prompt_tps" if q == "reading" else "tps"] for r in rows if r["engine"] == e and r["prompt"] == q]
            cells.append(f"{min(v):.0f}-{max(v):.0f}" if v else "")
        print(f"{e:42}" + "".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
