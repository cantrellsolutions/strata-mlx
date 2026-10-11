"""Does the MLX model compute what llama.cpp computes from the same file?

Asks a running llama-server for a greedy continuation of a prompt and its top log-probabilities, then pushes the
same tokens through the MLX model twice: the whole sequence in one pass (how a prompt is read) and the
continuation one token at a time (how an answer is written).

    llama-server -m <shard 1> --port 8091 -c 8192 &
    uv run python bench/compare_llama.py --model <shard 1> --prompt-file some.txt

With `--ref <file>` the first run only saves llama.cpp's answer; stop the server and run again to compare.
With `--windows 1 4` the same tokens also go through the engine, which reads the prompt in chunks and then
decodes that many tokens a pass; `--engine-only` leaves out the model's own two passes, which past some tens of
thousands of tokens attend to everything at once and do not fit.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))
from strata_mlx.dense import AS_BLOCKS  # noqa: E402
from strata_mlx.engine import Engine  # noqa: E402
from strata_mlx.load import load  # noqa: E402


END_TOKENS = (248044, 248046)  # <|endoftext|>, <|im_end|>


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # a local server: no proxy
    with opener.open(req, timeout=600) as r:
        return json.load(r)


def report(name: str, logits: mx.array, ref: list) -> None:
    """`logits` (n, vocab): row i predicts the token llama.cpp produced at step i."""
    logp = logits.astype(mx.float32)
    # the server is asked to write past the end of an answer (ignore_eos), which it does by forbidding the
    # end tokens outright; forbid them here too, or every step where the model wants to stop looks like a mismatch
    logp[:, list(END_TOKENS)] = -mx.inf
    logp = logp - mx.logsumexp(logp, axis=-1, keepdims=True)
    top = mx.argmax(logp, axis=-1).tolist()
    same = sum(t == step["id"] for t, step in zip(top, ref))
    first = next((i for i, (t, step) in enumerate(zip(top, ref)) if t != step["id"]), None)
    worst = max(abs(logp[i, c["id"]].item() - c["logprob"])
                for i, step in enumerate(ref) for c in step["top_logprobs"]
                if c["logprob"] > -20 and c["id"] not in END_TOKENS)  # llama.cpp forbids one of the two, this both
    where = "never" if first is None else f"first at step {first}"
    print(f"{name}: top token equal at {same}/{len(ref)} steps (differs: {where}); "
          f"largest log-probability gap among llama.cpp's top {len(ref[0]['top_logprobs'])}: {worst:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8091")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--max-prompt", type=int, default=0, help="keep only this many prompt tokens (0: all)")
    p.add_argument("-n", type=int, default=48)
    p.add_argument("--probs", type=int, default=5)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--ref", help="llama.cpp's answer in a file: written on the first run, which then stops; read "
                                 "on the next, so the two models need not be in memory together")
    p.add_argument("--windows", nargs="*", type=int, default=[], help="also through the engine, this many tokens a pass")
    p.add_argument("--engine-only", action="store_true")
    p.add_argument("--expert-memory", type=float, help="GB of experts in memory, the others read from the SSD")
    p.add_argument("--dense", choices=["blocks", "arrays"], help="the dense tensors of two types as the file's blocks "
                                                                "or as MLX's arrays (default: as the loader chooses)")
    args = p.parse_args()

    if args.ref and Path(args.ref).exists():
        saved = json.loads(Path(args.ref).read_text())
        prompt, ref = saved["prompt"], saved["ref"]
    else:
        prompt = post(args.server + "/tokenize", {"content": Path(args.prompt_file).read_text()})["tokens"]
        if args.max_prompt:
            prompt = prompt[: args.max_prompt]
        out = post(args.server + "/completion", {
            "prompt": prompt, "n_predict": args.n, "temperature": 0, "n_probs": args.probs, "return_tokens": True,
            "ignore_eos": True, "cache_prompt": False})
        ref = out["completion_probabilities"]
        print(f"llama.cpp: {out['timings']['prompt_per_second']:.0f} tok/s prompt, "
              f"{out['timings']['predicted_per_second']:.1f} tok/s answer")
        if args.ref:
            Path(args.ref).write_text(json.dumps({"prompt": prompt, "ref": ref}))
            return
    gen = [step["id"] for step in ref]
    print(f"prompt {len(prompt)} tokens, continuation {len(gen)} tokens")

    model, _ = load(args.model, getattr(mx, args.dtype), expert_memory=args.expert_memory,
                    dense_blocks={"blocks": AS_BLOCKS, "arrays": (), None: None}[args.dense])
    n = len(prompt)

    if not args.engine_only:
        logits = model(mx.array([prompt + gen[:-1]]), model.make_cache())[0, n - 1 :]
        mx.eval(logits)
        report("one pass     ", logits, ref)

        cache = model.make_cache()
        rows = [model(mx.array([prompt]), cache)[0, -1]]
        mx.eval(rows[0])
        for t in gen[:-1]:
            rows.append(model(mx.array([[t]]), cache)[0, -1])
            mx.eval(rows[-1])
        report("token by token", mx.stack(rows), ref)

    engine = Engine(model)
    following = prompt[-1:] + gen[:-1]  # row i of their logits predicts llama.cpp's step i
    for size in args.windows:
        engine.reset()
        engine.read(prompt[:-1])
        rows = []
        for a in range(0, len(following), size):
            logits, w = engine._forward(mx.array([following[a : a + size]]))
            mx.eval(logits)
            engine._commit(w, logits.shape[1])
            rows.append(logits[0])
        report(f"engine, {size} a pass", mx.concatenate(rows), ref)


if __name__ == "__main__":
    main()
