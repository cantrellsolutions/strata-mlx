"""What a long conversation costs: passes and a chunk of prompt with N tokens already in the attention caches.

The caches are filled with random cells instead of reading N tokens (two minutes for 32K, and the cost of a pass
does not depend on what the cells hold), then timed: passes of one token, passes with the draft head, and
reading a chunk of prompt, with the memory each reaches above what was held before.

    uv run python bench/long_context.py --model <shard 1> --mtp ~/.cache/strata-mlx/mtp
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402
from strata_mlx.mtp import load_mtp  # noqa: E402


def fill(engine: Engine, n: int, dtype):
    """A conversation of n tokens, as far as the attention caches and the token count go."""
    engine.reset()
    a = engine.m.args
    caches = list(engine.attn.values()) + ([engine.mtp_cache] if engine.mtp_cache is not None else [])
    for cache in caches:
        kv = (1, a.num_key_value_heads, n, a.head_dim)
        cache.state = (mx.random.normal(kv).astype(dtype), mx.random.normal(kv).astype(dtype),
                       mx.random.normal((1, n, a.indexer_head_dim)).astype(dtype))
        mx.eval(cache.keys, cache.values, cache.indexer.keys)
    engine.tokens = [1] * n
    if engine.lookup:
        engine.lookup.extend(engine.tokens)
    engine.next_token, engine.guesses = 1, []


def passes(engine: Engine, count: int) -> dict:
    """Milliseconds and tokens in the window of each of `count` passes, the first two left out."""
    ms, sizes = [], []
    for _ in range(count + 2):
        size = 1 + len(engine.guesses)
        t0 = time.perf_counter()
        engine.step()
        ms.append((time.perf_counter() - t0) * 1e3)
        sizes.append(size)
    ms, sizes = ms[2:], sizes[2:]
    return {"ms": round(sorted(ms)[len(ms) // 2], 1), "window": round(sum(sizes) / len(sizes), 2)}


def above(held: int) -> float:
    return round((mx.get_peak_memory() - held) / 2**30, 2)


p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--mtp", required=True)
p.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192, 32768, 65536, 131072])
p.add_argument("--chunk", type=int, default=2048)
p.add_argument("--chunk-upto", type=int, default=1 << 30, help="read a chunk only at lengths up to this")
p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
p.add_argument("--json")
args = p.parse_args()
dtype = getattr(mx, args.dtype)
model, _ = load(args.model, dtype)
mtp = load_mtp(args.mtp, model.args.text, dtype)
plain, drafting = Engine(model, chunk=args.chunk), Engine(model, mtp, 3, chunk=args.chunk)
drafting.warm()
rows = []
for n in args.lengths:
    row = {"tokens": n}
    for name, engine in (("one token", plain), ("draft head", drafting)):
        fill(engine, n, dtype)
        mx.clear_cache()
        held = mx.get_active_memory()
        mx.reset_peak_memory()
        row[name] = {**passes(engine, 8), "peak_gb": above(held)}
        engine.reset()
    if n <= args.chunk_upto:
        fill(drafting, n, dtype)
        mx.clear_cache()
        held = mx.get_active_memory()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        drafting.read(drafting.tokens + [2 + i % 1000 for i in range(args.chunk)])
        s = time.perf_counter() - t0
        row["chunk"] = {"tok_s": round(args.chunk / s), "peak_gb": above(held)}
        drafting.reset()
    mx.clear_cache()
    rows.append(row)
    print(json.dumps(row), flush=True)
if args.json:
    Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
