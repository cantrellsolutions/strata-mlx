"""Draft-and-verify against plain greedy decoding, on the real model.

For each prompt: decode the answer one token per pass, then with the draft head guessing ahead, then with lookup
guesses as well, and report the speed of each, how many tokens a pass settles, how many guesses are kept, and
where (if anywhere) an answer parts from the first one.

    uv run python bench/spec_decode.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402
from strata_mlx.sampler import Sampling  # noqa: E402

EDIT_ME = '''\
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Item:
    name: str
    price: float
    quantity: int = 1

    def total(self) -> float:
        return self.price * self.quantity


class Cart:
    def __init__(self):
        self.items = []

    def add(self, name: str, price: float, quantity: int = 1) -> None:
        for item in self.items:
            if item.name == name:
                item.quantity += quantity
                return
        self.items.append(Item(name, price, quantity))

    def remove(self, name: str) -> bool:
        for i, item in enumerate(self.items):
            if item.name == name:
                del self.items[i]
                return True
        return False

    def total(self) -> float:
        return sum(item.total() for item in self.items)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps([vars(item) for item in self.items], indent=2))

    @classmethod
    def load(cls, path: Path) -> "Cart":
        cart = cls()
        for row in json.loads(path.read_text()):
            cart.items.append(Item(**row))
        return cart
'''
PROMPTS = {
    "prose": "Explain in a few paragraphs why the sky is blue during the day and red at sunset.",
    "code": "Write a Python function that parses an ISO 8601 date string without using any libraries, with tests.",
    "reasoning": "A farmer has 17 sheep. All but 9 run away, then he buys twice as many as he has left and sells 5. "
                 "How many sheep does he have? Show your work.",
    # an answer that repeats its prompt: what lookup guesses are for
    "edit": "Add a `discount: float = 0.0` field to Item (a fraction of the price taken off in total()) and print the "
            "whole file again with that change and no others.\n\n```python\n" + EDIT_ME + "```",
}
STOP = (248044, 248046)  # <|endoftext|>, <|im_end|>


def tokenize(model: str, text: str) -> list:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_DEBUG_LAYER")}
    out = subprocess.run(["llama-tokenize", "-m", model, "-p", text, "--ids", "--log-disable"],
                         capture_output=True, text=True, env=env, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def decode(engine: Engine, prompt: list, max_tokens: int, sampling: dict):
    engine.reset()
    engine.sampling = Sampling(**sampling)
    t0 = time.perf_counter()
    out = [engine.read(prompt)]
    t1 = time.perf_counter()
    while len(out) < max_tokens and out[-1] not in STOP:
        out += engine.step(max_tokens - len(out), STOP)
    return out, len(prompt) / (t1 - t0), (len(out) - 1) / (time.perf_counter() - t1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mtp", required=True, help="the folder tools/mtp_fetch.py wrote")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--drafts", type=int, default=3)
    p.add_argument("--no-think", action="store_true", help="start the answer after an empty thinking block")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--temperature", type=float, default=0.0, help="0: greedy; else top_p 0.95, top_k 20")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    p.add_argument("--json")
    args = p.parse_args()

    import mlx.core as mx

    dtype = getattr(mx, args.dtype)
    model, _ = load(args.model, dtype)
    mtp = load_mtp(args.mtp, model.args.text, dtype)
    sampling = dict(temperature=args.temperature, top_p=0.95, top_k=20, seed=args.seed) if args.temperature else {}
    engines = {"one token a pass": Engine(model), "draft head": Engine(model, mtp, args.drafts),
               "draft head + lookup": Engine(model, mtp, args.drafts, lookup=True)}
    for engine in engines.values():
        engine.warm()
    rows = []
    for name in args.prompts:
        text = f"<|im_start|>user\n{PROMPTS[name]}<|im_end|>\n<|im_start|>assistant\n"
        if args.no_think:
            text += "<think>\n\n</think>\n\n"
        prompt = tokenize(args.model, text)
        ref = None
        for label, engine in engines.items():
            out, _, tps = decode(engine, prompt, args.tokens, sampling)
            ref = ref or out
            s = engine.stats
            same = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), min(len(out), len(ref)))
            row = {"prompt": name, "engine": label, "prompt_tokens": len(prompt), "tokens": len(out),
                   "tok_s": round(tps, 1), "tokens_per_pass": round(s.tokens / max(s.steps, 1), 2),
                   "guesses_kept": round(s.accepted / max(s.drafted, 1), 2), "lookup_passes": s.lookup_steps,
                   "passes": s.steps, "same_as_first_until": same}
            rows.append(row)
            parted = "" if same == len(ref) == len(out) else f" | parts from the first at token {same}"
            print(f"{name:<10} {label:<20} {row['tok_s']:>6} tok/s  {row['tokens_per_pass']:>4} tokens/pass  "
                  f"{row['guesses_kept']:>4.0%} of guesses kept  {s.lookup_steps:>3}/{s.steps} lookup passes"
                  f"{parted}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
