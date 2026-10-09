"""Which experts a conversation uses, and how many it would miss if only some were kept in memory.

Records the ten experts every layer routes each token to, over a few conversations of different kinds, then
replays each conversation against a set of resident experts of a given size: the most used ones in the *other*
conversations, kept fixed, or the same set as a start with the least recently used one replaced at every miss.
A miss is an expert that would have to be read from the SSD (three tensors) before the token can go on.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spec_decode import PROMPTS, STOP, tokenize  # noqa: E402

from strata_mlx.blocks import BlockSwitchGLU  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

MORE = {
    "chinese": "用中文解释一下什么是复利,并举一个每月定投的例子,给出计算过程。",
    "summary": None,  # a long document to summarize: --document
    "sql": "Write a SQL query that returns, for each customer, their three most recent orders with the order total, "
           "given tables customers(id, name), orders(id, customer_id, created_at) and order_items(order_id, price, qty). "
           "Explain the query.",
}


def record(args) -> dict:
    model, _ = load(args.model)
    layers = len(model.model.layers)
    seen, call = [], BlockSwitchGLU.__call__

    def spy(self, x, indices):
        seen.append(indices)
        return call(self, x, indices)

    BlockSwitchGLU.__call__ = spy
    engine = Engine(model, fused=False)  # the plain pass: nothing compiled, so the spy sees every call
    texts = dict(PROMPTS, **MORE)
    texts["summary"] = Path(args.document).read_text() + "\n\nSummarize the text above in five sentences."
    out = {}
    for name, text in texts.items():
        prompt = tokenize(args.model, f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        engine.reset()
        del seen[:]
        produced = [engine.read(prompt)]
        while len(produced) < args.tokens and produced[-1] not in STOP:
            produced += engine.step(1, STOP)
        mx.eval(seen)
        passes = [np.stack([np.array(a)[0] for a in seen[i : i + layers]], axis=1) for i in range(0, len(seen), layers)]
        use = np.concatenate(passes).astype(np.int16)  # (tokens, layers, experts per token)
        out[name] = {"prompt_tokens": len(prompt), "use": use}
        print(f"{name}: {len(prompt)} prompt tokens, {len(use) - len(prompt)} answer tokens", flush=True)
    return out


def replay(use: np.ndarray, resident: set, lru: bool, experts: int) -> np.ndarray:
    """Misses at every token. `resident`: keys layer * experts + expert."""
    order = dict.fromkeys(resident)  # insertion order is recency
    misses = np.zeros(len(use), dtype=np.int32)
    for t, token in enumerate(use):
        for layer, row in enumerate(token):
            for e in row:
                key = layer * experts + int(e)
                if key in order:
                    if lru:
                        order[key] = order.pop(key)
                else:
                    misses[t] += 1
                    if lru:
                        order.pop(next(iter(order)))
                        order[key] = None
    return misses


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--budgets", nargs="*", type=int, default=[1500, 3000, 6000, 12000])
    p.add_argument("--json")
    p.add_argument("--counts", help="write how often each expert was used, (layers, experts), to this .npy: the "
                                    "profile a small machine loads its first experts by (strata_mlx/expert_use.npy)")
    args = p.parse_args()
    talks = record(args)
    layers = next(iter(talks.values()))["use"].shape[1]
    total = layers * args.experts
    counts = {name: np.bincount((t["use"] + np.arange(layers)[None, :, None] * args.experts).reshape(-1), minlength=total)
              for name, t in talks.items()}
    if args.counts:
        np.save(args.counts, sum(counts.values()).reshape(layers, args.experts).astype(np.uint16))
    rows = []
    print(f"\n{total} experts in all; a token uses {layers * next(iter(talks.values()))['use'].shape[2]}")
    for budget in args.budgets:
        line = {"resident": budget, "share": round(budget / total, 3), "fixed": {}, "lru": {}}
        for name, t in talks.items():
            others = sum(c for n, c in counts.items() if n != name)
            resident = set(np.argsort(-others)[:budget].tolist())
            n = t["prompt_tokens"]
            for kind, lru in (("fixed", False), ("lru", True)):
                m = replay(t["use"], resident, lru, args.experts)
                line[kind][name] = {"prompt": round(float(m[:n].mean()), 1), "answer": round(float(m[n:].mean()), 1)}
        rows.append(line)
        for kind in ("fixed", "lru"):
            a = [v["answer"] for v in line[kind].values()]
            pr = [v["prompt"] for v in line[kind].values()]
            print(f"{budget:6} resident ({budget / total:5.1%}), {kind:5}: misses per answer token {min(a):5.1f} - {max(a):5.1f} "
                  f"(mean {np.mean(a):5.1f}); per prompt token {np.mean(pr):5.1f}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
