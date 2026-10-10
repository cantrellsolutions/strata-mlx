"""Experiment 35: can a layer's experts be known before its router has run, so that the disk reads them early?

With part of the experts in memory (paged.py) a layer waits for its router and then for the disk, and the GPU
has nothing to do meanwhile. Reading ahead needs a guess of the experts, made a layer or half a layer before.
Others guess with the model's own routers on an earlier state (Eliseev and Mazur, arXiv:2312.17238; slotstream;
mlx-stream). Here: how good are such guesses for this model, and of the experts a small memory would have to
read, how many would they have asked for in time?

Nothing is timed and nothing is read from the disk. With every expert in memory, a token at a time, the pass
records the ten experts each layer routes to and three guesses of them, each a router's 32 best:

    a   layer L's router on what its hyper-connection reads before L's mixer has written   (the mixer ahead)
    b   layer L's router and hyper-connection on the residual as layer L-1's experts find it
    c   layer L's router on layer L-1's experts' input                                     (b, c: a layer ahead)

and, from the record alone, `last`: the experts the token before went to at the same layer.

Then the conversations are replayed against a memory of some experts (those most used in the other
conversations, the least recently used replaced at a miss: bench/expert_use.py), and for the answers' tokens:
of the experts that had to be read, the share a guess's best N held (they could have been read ahead), and the
reads a guess would add for nothing (among its best N, not in memory, not used).

What was to decide it, set before the run. From bench/results/paged_*.json, with the draft head, the disk is
65-71% of an answer's time at 2 GB of experts (a 16 GB Mac), 43-45% at 8 GB (a 24 GB Mac), 6-14% at 16 GB (a
32 GB one). Reads asked for ahead overlap the GPU's work and no more: a token then takes the longer of the two,
plus the reads not guessed. Build it only if that comes to 1.5 times at 2 GB or at 8 GB with a guess that adds
no more reads than it spares. slotstream, on a 48 GB Mac with 20-22 GB for this model, measured 1.1 times.

    uv run python bench/experiments/e35_route_ahead.py --model <shard 1> --document <text to summarize>
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from expert_use import MORE  # noqa: E402
from spec_decode import PROMPTS, STOP, tokenize  # noqa: E402

from strata_mlx.blocks import BlockSwitchGLU  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402

BEST = 32  # of a guess that are recorded
# (share of an answer's time the disk takes, tokens a second) with the draft head: bench/results/paged_*.json
MEASURED = {1500: (0.68, 9.3, "2 GB, a 16 GB Mac"), 3000: (0.59, 12.5, "4 GB"), 6000: (0.44, 20.3, "8 GB, a 24 GB Mac"),
            12000: (0.10, 37.0, "16 GB, a 32 GB Mac")}


def record(args) -> dict:
    model, _ = load(args.model)
    layers = model.model.layers
    engine = Engine(model, fused=False)  # the plain pass: nothing compiled, so the spies see every call
    seen = {"use": [], "a": [], "b": [], "c": []}
    call = BlockSwitchGLU.__call__

    def spy(self, x, indices):
        seen["use"].append(indices)
        return call(self, x, indices)

    BlockSwitchGLU.__call__ = spy

    def best(layer, x):
        """The router's 32 best for an input of the experts, best first."""
        return mx.argsort(-layer.mlp.gate(x.astype(mx.float32)), axis=-1)[..., :BEST]

    reads = [layer.mlp_hyper_connection for layer in layers]
    for i, layer in enumerate(layers):
        def before_mixer(h, i=i, layer=layer, inner=layer.attn_hyper_connection):
            seen["a"].append(best(layer, reads[i](h)[0]))
            return inner(h)

        def before_experts(h, i=i):
            out = reads[i](h)
            if i + 1 < len(layers):
                seen["b"].append(best(layers[i + 1], reads[i + 1](h)[0]))
                seen["c"].append(best(layers[i + 1], out[0]))
            return out

        layer.attn_hyper_connection, layer.mlp_hyper_connection = before_mixer, before_experts

    def taken() -> dict:
        """What the passes since the last call recorded, a row to each token fed: (tokens, layers, experts)."""
        mx.eval(*[a for v in seen.values() for a in v])
        n, out = len(layers), {}
        for key, arrays in seen.items():
            per = n if key in ("use", "a") else n - 1  # b and c have no guess of the first layer
            rows = []
            for at in range(0, len(arrays), per):
                piece = np.stack([np.array(a)[0] for a in arrays[at : at + per]], axis=1).astype(np.int16)
                if per < n:
                    piece = np.concatenate([np.full_like(piece[:, :1], -1), piece], axis=1)
                rows.append(piece)
            out[key] = np.concatenate(rows)
            del arrays[:]
        return out

    texts = dict(PROMPTS, **MORE)
    texts["summary"] = Path(args.document).read_text() + "\n\nSummarize the text above in five sentences."
    talks = {}
    for name, text in texts.items():
        prompt = tokenize(args.model, f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        engine.reset()
        produced, parts = [engine.read(prompt)], [taken()]
        while len(produced) < args.tokens and produced[-1] not in STOP:
            produced += engine.step(1, STOP)
            parts.append(taken())
        talks[name] = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        talks[name]["prompt_tokens"] = len(prompt)
        assert all(len(talks[name][key]) == len(talks[name]["use"]) for key in ("a", "b", "c"))
        print(f"{name}: {len(prompt)} prompt tokens, {len(talks[name]['use']) - len(prompt)} answer tokens", flush=True)
    return talks


def replay(talk: dict, resident: list, experts: int, guesses: dict, counts: tuple) -> dict:
    """An answer's misses against a memory of `resident` experts (the last is the last to go), and for each
    guess and each number of its best: the misses it held, the reads it would add for nothing."""
    use, n = talk["use"], talk["prompt_tokens"]
    order = dict.fromkeys(resident)  # insertion order is recency
    out = {"misses": 0, "tokens": len(use) - n,
           **{(g, c): {"held": 0, "added": 0} for g in guesses for c in counts}}
    for t, token in enumerate(use):
        for layer, row in enumerate(token):
            keys = [layer * experts + int(e) for e in row]
            missed = {int(e) for e, key in zip(row, keys) if key not in order}
            if t >= n:
                out["misses"] += len(missed)
                routed = {int(e) for e in row}
                for g, guess in guesses.items():
                    ahead = guess[t, layer]
                    if ahead[0] < 0:
                        continue
                    for c in counts if g != "last" else counts[:1]:  # the token before went to ten: all of them
                        asked = {int(e) for e in (ahead[:c] if g != "last" else ahead) if layer * experts + int(e) not in order}
                        out[(g, c)]["held"] += len(asked & missed)
                        out[(g, c)]["added"] += len(asked - routed)
            for key in keys:
                if key in order:
                    order[key] = order.pop(key)
                else:
                    order.pop(next(iter(order)))
                    order[key] = None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--budgets", nargs="*", type=int, default=[1500, 3000, 6000, 12000])
    p.add_argument("--counts", nargs="*", type=int, default=[10, 12, 16, 24, 32])
    p.add_argument("--save", help="keep the record here (.npz), or replay the one that is there")
    p.add_argument("--json")
    args = p.parse_args()
    if args.save and Path(args.save).exists():
        z = np.load(args.save)
        talks = {}
        for key in z.files:
            name, field = key.split("__", 1)
            talks.setdefault(name, {})[field] = z[key] if field != "prompt_tokens" else int(z[key])
    else:
        talks = record(args)
        if args.save:
            np.savez_compressed(args.save, **{f"{name}__{field}": v for name, t in talks.items() for field, v in t.items()})
    layers = next(iter(talks.values()))["use"].shape[1]
    total = layers * args.experts
    counts = tuple(args.counts)
    names = ("a", "b", "c", "last")
    for t in talks.values():  # the token before, at the same layer: known a whole pass ahead
        t["last"] = np.concatenate([np.full_like(t["use"][:1], -1), t["use"][:-1]])

    # how good the guesses are, whatever is in memory: of the ten experts routed to, those among a guess's best
    print(f"\nof the ten experts a layer routes to, among a guess's best N (answers' tokens, {len(talks)} conversations):")
    print(f"{'':6}" + "".join(f"{'best ' + str(c):>10}" for c in counts))
    agree = {}
    for g in names:
        row = []
        for c in counts:
            hit = n = 0
            for t in talks.values():
                use, guess = t["use"][t["prompt_tokens"]:], t[g][t["prompt_tokens"]:, :, : c if g != "last" else None]
                ok = guess[:, :, 0] >= 0
                hit += int(((use[:, :, :, None] == guess[:, :, None, :]).any(-1) & ok[:, :, None]).sum())
                n += int(ok.sum()) * use.shape[2]
            row.append(hit / n)
            if g == "last":
                break
        agree[g] = row
        print(f"{g:6}" + "".join(f"{v:10.1%}" for v in row) + ("   (its ten)" if g == "last" else ""))

    use_counts = {name: np.bincount((t["use"] + np.arange(layers)[None, :, None] * args.experts).reshape(-1), minlength=total)
                  for name, t in talks.items()}
    rows = []
    for budget in args.budgets:
        sums = {}
        for name, t in talks.items():
            others = sum(c for n, c in use_counts.items() if n != name)
            resident = np.argsort(-others)[:budget][::-1].tolist()  # the most used are the last to go
            r = replay(t, resident, args.experts, {g: t[g] for g in names}, counts)
            for key, v in r.items():
                if isinstance(v, dict):
                    for field, x in v.items():
                        sums[(key, field)] = sums.get((key, field), 0) + x
                else:
                    sums[key] = sums.get(key, 0) + v
        misses, tokens = sums["misses"], sums["tokens"]
        share, tok_s, what = MEASURED.get(budget, (None, None, ""))
        print(f"\n{budget} experts in memory ({budget / total:.0%}; {what}): {misses / tokens:.1f} read a token of an answer")
        print(f"{'':6}" + "".join(f"{'best ' + str(c):>22}" for c in counts))
        line = {"resident": budget, "misses_per_token": misses / tokens, "guesses": {}}
        for g in names:
            cells, best = [], None
            for c in counts if g != "last" else counts[:1]:
                held, added = sums[((g, c), "held")] / misses, sums[((g, c), "added")] / misses
                cell = {"best": c, "held": held, "added": added}
                if share:
                    # the reads asked for ahead go on beside the GPU's work; what they take longer than it, and
                    # the reads not guessed, the token waits for
                    disk, gpu = share, 1 - share
                    cell["times"] = 1 / (gpu + (1 - held) * disk + max(0.0, (held + added) * disk - gpu))
                    if best is None or cell["times"] > best["times"]:
                        best = cell
                line["guesses"].setdefault(g, []).append(cell)
                cells.append(f"{held:6.0%} held +{added:5.0%}" + (f" {cell['times']:4.2f}x" if share else ""))
                if g == "last":
                    cell["best"] = 10
            print(f"{g:6}" + "".join(f"{c:>22}" for c in cells))
            if best:
                line.setdefault("at_best", {})[g] = best
        rows.append(line)
    print("\nheld: of the experts read, those among the guess's best N. +: the reads it adds, as a share of them.")
    print("x: what an answer would gain if every read asked for ahead went on beside the GPU's work (see the top).")
    if args.json:
        Path(args.json).write_text(json.dumps({"agreement": dict(zip(names, [agree[g] for g in names])), "counts": counts,
                                               "budgets": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
