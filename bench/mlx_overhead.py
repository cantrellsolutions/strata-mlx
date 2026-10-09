"""What one step of the Qwen3.8-Flash-Next graph costs in MLX, before any weights are downloaded.

The model is built at its real shapes with random weights: 2-bit experts, 4-bit dense projections, the router in
full precision. The 28.8 GB n-gram table is replaced by a small stub (the engine reads it on the CPU). Speed
depends on the shapes and bit widths, not on the values, so this answers two questions:

  * decode: the time of a step that pushes T tokens through all 48 layers, for T = 1, 2, 3, 4, ...
    T = 1 is plain decoding; T > 1 is what a draft-and-verify window costs.
  * prefill: tokens per second when a prompt is read in chunks of S tokens.

    uv run bench/mlx_overhead.py --json bench/results/mlx_overhead.json
"""

import argparse
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen4_exp
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

GROUP = 64
STUB_ROWS = 1 << 16


class StubTable(nn.Module):
    """Stands in for the sharded n-gram table: the same rows-in, vectors-out contract, 65,536 rows."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.random.normal((STUB_ROWS, dim)).astype(mx.float16) * 0.02

    def __call__(self, gid: mx.array) -> mx.array:
        return self.weight[gid % STUB_ROWS]


def random_switch(n_expert: int, out_dims: int, in_dims: int, bits: int) -> QuantizedSwitchLinear:
    """A quantized expert stack filled with random codes, without ever holding it in float."""
    q = QuantizedSwitchLinear(GROUP, 1, 1, bias=False, group_size=GROUP, bits=bits)
    words = in_dims * bits // 32
    hi = mx.random.randint(0, 1 << 16, (n_expert, out_dims, words)).astype(mx.uint32)
    lo = mx.random.randint(0, 1 << 16, (n_expert, out_dims, words)).astype(mx.uint32)
    scale = 0.5 / in_dims**0.5
    q.weight = (hi << 16) | lo
    q.scales = mx.full((n_expert, out_dims, in_dims // GROUP), scale, dtype=mx.float16)
    q.biases = mx.full((n_expert, out_dims, in_dims // GROUP), -scale * ((1 << bits) - 1) / 2, dtype=mx.float16)
    q.group_size, q.bits = GROUP, bits
    return q


def build(expert_bits: int, dense_bits: int) -> nn.Module:
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config={}))
    a = model.args.text
    for layer in model.layers:
        if layer.ple is not None:
            emb = layer.ple.ple_embedding
            emb.ngram_embedding = StubTable(emb.ngram_embedding.dim)
        sw = layer.mlp.switch_mlp
        sw.gate_proj = random_switch(a.num_experts, a.moe_intermediate_size, a.hidden_size, expert_bits)
        sw.up_proj = random_switch(a.num_experts, a.moe_intermediate_size, a.hidden_size, expert_bits)
        sw.down_proj = random_switch(a.num_experts, a.hidden_size, a.moe_intermediate_size, expert_bits)

    def dense(path, module):
        return isinstance(module, (nn.Linear, nn.Embedding)) and not path.endswith("mlp.gate")

    # one layer at a time, so only one layer's float weights exist at once
    for part in [*model.layers, model.model.hyper_connection_mixer, model.model.embed_tokens, model.lm_head]:
        holder = nn.Module()
        holder.m = part
        nn.quantize(holder, group_size=GROUP, bits=dense_bits, class_predicate=dense)
        if holder.m is not part:
            if part is model.lm_head:
                model.lm_head = holder.m
            else:
                model.model.embed_tokens = holder.m
        holder.set_dtype(mx.float16)
        mx.eval(holder.parameters())
    model.eval()
    return model


def arrays(tree):
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, (list, tuple)):
        for t in tree:
            yield from arrays(t)
    elif isinstance(tree, dict):
        yield from arrays(list(tree.values()))


def run(model, cache, ids):
    """One forward over ids (1, T): returns (seconds building the graph, seconds evaluating it)."""
    t0 = time.perf_counter()
    tok = mx.argmax(model(ids, cache), axis=-1)
    t1 = time.perf_counter()
    mx.eval(tok, *arrays([c.state for c in cache]))
    return t1 - t0, time.perf_counter() - t1


def fresh(model, vocab: int, n: int):
    cache = model.make_cache()
    if n:
        run(model, cache, mx.random.randint(0, vocab, (1, n)))
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expert-bits", type=int, default=2)
    p.add_argument("--dense-bits", type=int, default=4)
    p.add_argument("--context", type=int, default=1024, help="tokens already in the conversation when decoding")
    p.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--chunks", type=int, nargs="*", default=[512, 2048, 8192])
    p.add_argument("--json", help="also write the results to this file")
    args = p.parse_args()

    mx.random.seed(0)
    t0 = time.perf_counter()
    model = build(args.expert_bits, args.dense_bits)
    vocab = model.args.text.vocab_size
    info = mx.device_info()
    out = {
        "device": info["device_name"],
        "mlx": mx.__version__,
        "expert_bits": args.expert_bits,
        "dense_bits": args.dense_bits,
        "weights_gb": round(mx.get_active_memory() / 1e9, 1),
        "build_s": round(time.perf_counter() - t0, 1),
        "context": args.context,
        "decode": [],
        "prefill": [],
    }
    print(f"{out['device']}, MLX {out['mlx']}: {out['weights_gb']} GB of weights built in {out['build_s']} s")

    print(f"\ndecode at a {args.context}-token context, medians of {args.steps} steps")
    print(f"{'T':>3} {'ms/step':>9} {'graph ms':>9} {'ms/token':>9} {'vs T=1':>7}")
    base = None
    for T in args.windows:
        cache = fresh(model, vocab, args.context)
        times = [run(model, cache, mx.random.randint(0, vocab, (1, T))) for _ in range(args.steps + 5)][5:]
        build_ms = statistics.median(b for b, _ in times) * 1e3
        step_ms = statistics.median(b + e for b, e in times) * 1e3
        base = base or step_ms
        row = {"T": T, "step_ms": round(step_ms, 2), "graph_ms": round(build_ms, 2),
               "ms_per_token": round(step_ms / T, 2), "vs_T1": round(step_ms / base, 2)}
        out["decode"].append(row)
        print(f"{T:>3} {row['step_ms']:>9} {row['graph_ms']:>9} {row['ms_per_token']:>9} {row['vs_T1']:>7}")
        del cache

    if args.chunks:
        print("\nprefill, one chunk into an empty conversation, best of 3")
        print(f"{'S':>6} {'tok/s':>8} {'peak GB':>8}")
    for S in args.chunks:
        best = 0.0
        mx.reset_peak_memory()
        for _ in range(3):
            cache = model.make_cache()
            b, e = run(model, cache, mx.random.randint(0, vocab, (1, S)))
            best = max(best, S / (b + e))
            del cache
        row = {"S": S, "tok_s": round(best), "peak_gb": round(mx.get_peak_memory() / 1e9, 1)}
        out["prefill"].append(row)
        print(f"{S:>6} {row['tok_s']:>8} {row['peak_gb']:>8}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
