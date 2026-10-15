"""Shuffled shared-weight comparisons with duplicate controls and raw evidence.

This experiment is Qwen4Exp-specific. Stop competing model workloads externally.
It neither owns nor changes serving services. Run with --help for explicit inputs.
MLX imports and all model work happen only in main(), on Apple hardware.
"""

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

SETTINGS = {
    "reference6": (False, 6),
    "new6": (True, 6),
    "new8": (True, 8),
    "control6": (True, 6),
}
METRICS = ("hits", "misses", "bytes_read", "read_s", "waited_s")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def windows(value):
    try:
        values = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated token windows") from error
    if (
        not values
        or len(values) != len(set(values))
        or any(v < 1 or v > 8 for v in values)
    ):
        raise argparse.ArgumentTypeError(
            "windows must be unique integers from 1 through 8"
        )
    return values


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "mtp", "tokenizer", "fixtures", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--micro-rounds", type=positive_int, default=20)
    p.add_argument("--answer-rounds", type=positive_int, default=3)
    p.add_argument("--windows", type=windows, default=(1, 4, 7, 8))
    p.add_argument("--max-tokens", type=positive_int, default=256)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument(
        "--answer-policy",
        choices=("reset", "preserve-history"),
        default="reset",
        help="reset adaptive draft policy per request, or reproduce legacy history retention",
    )
    p.add_argument("--max-prompt-tokens", type=positive_int, default=8192)
    p.add_argument(
        "--cache-mib",
        type=positive_int,
        default=512,
        help="MLX temporary allocation cache cap; not a conversation cache",
    )
    p.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    p.add_argument(
        "--stop-token",
        action="append",
        default=None,
        help="tokenizer token string; repeatable; defaults to Qwen end tokens",
    )
    return p


def json_equal(actual, expected):
    """JSON numbers compare numerically; booleans never equal numbers."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            json_equal(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            json_equal(x, y) for x, y in zip(actual, expected)
        )
    return actual == expected


def load_fixtures(path):
    fixtures = json.loads(path.read_text())
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("fixtures must be a nonempty JSON list")
    names = set()
    for case in fixtures:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("name"), str)
            or not case["name"]
            or not isinstance(case.get("prompt"), str)
            or not case["prompt"]
        ):
            raise ValueError("each fixture needs a nonempty name and prompt")
        if case["name"] in names:
            raise ValueError("fixture names must be unique")
        names.add(case["name"])
    return fixtures


def prepare_output(path):
    """Never append a second run to existing evidence, even an incomplete run."""
    path.mkdir(parents=True, exist_ok=False)
    return path


def stop_ids(tokenizer, strings):
    ids = [tokenizer.token_to_id(token) for token in strings]
    if not strings or any(value is None for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("stop tokens must exist in tokenizer and be distinct")
    return tuple(ids)


def clean(text):
    visible = (
        text.replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .split("</think>")[-1]
    )
    return "" if "<think>" in visible else visible.strip()


def main(argv=None):
    a = parser().parse_args(argv)
    for name in ("model", "tokenizer", "fixtures"):
        path = getattr(a, name).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"--{name} must name an existing file: {path}")
        setattr(a, name, path)
    a.mtp = a.mtp.expanduser().resolve()
    if not a.mtp.is_dir():
        raise ValueError("--mtp must name an existing draft-head directory")
    fixtures = load_fixtures(a.fixtures)
    # Validate paths and evidence ownership before importing MLX or allocating any model.
    out = prepare_output(a.output.expanduser().resolve())

    def save(name, data):
        (out / name).write_text(json.dumps(data, indent=2) + "\n")

    def append(name, data):
        with (out / name).open("a") as stream:
            stream.write(json.dumps(data) + "\n")

    def phase(name, **extra):
        append(
            "phase-events.jsonl",
            {
                "phase": name,
                "monotonic": time.monotonic(),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **extra,
            },
        )
        print(name, flush=True)

    phase("run-start")
    original_precision_env = os.environ.get("MLX_ENABLE_TF32")
    try:
        run(a, fixtures, out, save, append, phase)
    except BaseException as error:
        phase("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if original_precision_env is None:
            os.environ.pop("MLX_ENABLE_TF32", None)
        else:
            os.environ["MLX_ENABLE_TF32"] = original_precision_env
    phase("complete")
    return 0


def run(a, fixtures, out, save, append, phase):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    original_precision_env = os.environ.get("MLX_ENABLE_TF32")
    if a.dtype == "float32":
        os.environ.setdefault("MLX_ENABLE_TF32", "0")
    import numpy as np
    import mlx.core as mx
    from tokenizers import Tokenizer
    from strata_mlx import engine as eng, fused, blocks, attention
    from strata_mlx.mtp import load_mtp
    from strata_mlx.sampler import Sampling
    from strata_mlx.paged import machine_memory

    loadmod = importlib.import_module("strata_mlx.load")
    tok = Tokenizer.from_file(str(a.tokenizer))
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids
    stop_strings = a.stop_token or ["<|im_end|>", "<|endoftext|>"]
    stop = stop_ids(tok, stop_strings)
    encoded_fixtures = {case["name"]: encode(case["prompt"]) for case in fixtures}
    if any(
        (not ids or len(ids) > a.max_prompt_tokens for ids in encoded_fixtures.values())
    ):
        raise ValueError("fixture prompts must encode to 1..--max-prompt-tokens tokens")
    if tok.get_vocab_size() <= 1023:
        raise ValueError("Qwen4Exp experiment requires at least 1024 tokenizer entries")
    flags = [
        (fused, "ROUTER_TOPK", True),
        (fused, "ROUTER_AS_IS", True),
        (fused, "COMBINE_UPTO", 16),
        (fused, "GDN_PREP_UPTO", 16),
        (fused, "GATED_NORM_UPTO", 16),
        (blocks, "GLU_IN_DOWN", True),
        (blocks, "BLOCK_GLU", True),
        (attention, "QK_ROPE_UPTO", 16),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in flags]
    previous_cache_limit = None

    def configure(enabled):
        for module, name, value in flags:
            setattr(module, name, value if enabled else False)

    def expert_metrics(model):
        store = model.expert_store
        return {key: getattr(store, key) for key in METRICS} if store else {}

    try:
        phase("load")
        t = time.monotonic()
        model, meta = loadmod.load(
            str(a.model), getattr(mx, a.dtype), expert_memory="auto"
        )
        load_s = time.monotonic() - t
        if model.model.args.eos_token_id not in stop:
            raise ValueError(
                "Tokenizer stop IDs do not include model EOS; verify matching tokenizer"
            )
        threads = getattr(model, "table_warming", None)
        info = {
            "device": mx.device_info(),
            "auto_fuse_upto": eng.FUSE_UPTO,
            "model": str(a.model),
            "load_s": load_s,
            "table_warming_started": bool(threads),
            "active_bytes_after_load": mx.get_active_memory(),
            "cache_bytes_after_load": mx.get_cache_memory(),
            "settings": SETTINGS,
            "dtype": a.dtype,
            "precision_env": os.environ.get("MLX_ENABLE_TF32"),
            "original_precision_env": original_precision_env,
            "metal_debug_env": os.environ.get("MTL_DEBUG_LAYER"),
            "kernel_reference_limit": "Reference toggles eight trunk module switches and fuse limit; current shared draft-head and Q2 thread rule remain. Not an exact historical source comparison.",
            "seed": a.seed,
            "sampling_seed": a.seed,
            "mlx_version": mx.__version__,
            "python_version": sys.version,
            "source_sha256": {
                str(
                    path.relative_to(Path(eng.__file__).resolve().parent)
                ): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(eng.__file__).resolve().parent.glob("*.py"))
            },
            "experiment_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "input_file_metadata": {
                "model": {
                    "size_bytes": a.model.stat().st_size,
                    "mtime_ns": a.model.stat().st_mtime_ns,
                },
                "mtp": [
                    {
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                    }
                    for path in sorted(a.mtp.iterdir())
                    if path.is_file()
                ],
            },
            "physical_memory_bytes": machine_memory(),
            "windows": a.windows,
            "micro_rounds": a.micro_rounds,
            "answer_rounds": a.answer_rounds,
            "max_tokens": a.max_tokens,
            "stop_tokens": stop_strings,
            "stop_ids": stop,
            "paths": {
                key: str(getattr(a, key))
                for key in ["model", "mtp", "tokenizer", "fixtures"]
            },
            "tokenizer_sha256": hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
            "fixtures_sha256": hashlib.sha256(a.fixtures.read_bytes()).hexdigest(),
            "answer_policy": a.answer_policy,
            "state_policy": "Separate KV/recurrent state. Adaptive policy follows answer_policy; compiled-shape knowledge remains warmed. Expert and OS caches shared. Whole answers remain descriptive, not fixed-work comparisons.",
            "plan": {
                "micro_rounds": a.micro_rounds,
                "answer_rounds": a.answer_rounds,
                "windows": a.windows,
            },
            "fixtures": fixtures,
            "full_model_resident": model.expert_store is None,
        }
        phase("draft-load")
        t = time.monotonic()
        mtp = load_mtp(str(a.mtp), model.args.text, getattr(mx, a.dtype))
        info["draft_load_s"] = time.monotonic() - t
        if model.expert_store is not None:
            info["expert_budget_bytes"] = model.expert_store.fit(machine_memory())
        previous_cache_limit = mx.set_cache_limit(a.cache_mib * 2**20)
        settings = SETTINGS
        engines = {}
        for label, (enabled, limit) in settings.items():
            phase("warm-" + label)
            configure(enabled)
            e = eng.Engine(model, mtp, fuse_upto=limit)
            t = time.monotonic()
            e.warm()
            e.generate(
                list(range(1000, 1024)),
                12,
                sampling=Sampling(temperature=0, seed=a.seed),
            )
            e.reset()
            engines[label] = e
            info.setdefault("warm_s", {})[label] = time.monotonic() - t
        if threads:
            phase("wait-table-warming")
            for thread in threads:
                if thread:
                    thread.join(timeout=180)
            info["table_warming_finished"] = all(
                (not thread or not thread.is_alive() for thread in threads)
            )
            if not info["table_warming_finished"]:
                raise RuntimeError("Table warm exceeded bound")
        else:
            info["table_warming_finished"] = False
        save("configuration.json", info)
        rng = random.Random(a.seed)
        prefix = encode(
            "<|im_start|>user\nReturn exactly the word READY.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        phase("micro-start")
        for window in a.windows:
            ids = mx.array([[1000 + i for i in range(window)]], dtype=mx.int32)
            for label, e in engines.items():
                configure(settings[label][0])
                e.reset()
                e.read(prefix)
                logits, w = e._forward(ids)
                mx.eval(logits, *eng._arrays([list(w.gdn), w.ple_conv, w.residual]))
                for cache in e.attn.values():
                    cache.trim(window)
            for repetition in range(a.micro_rounds):
                order = list(settings)
                rng.shuffle(order)
                for position, label in enumerate(order):
                    configure(settings[label][0])
                    e = engines[label]
                    mx.synchronize()
                    t = time.perf_counter()
                    logits, w = e._forward(ids)
                    mx.eval(logits, *eng._arrays([list(w.gdn), w.ple_conv, w.residual]))
                    elapsed = (time.perf_counter() - t) * 1000
                    raw = np.array(logits)
                    logits_finite = bool(np.isfinite(raw).all())
                    append(
                        "micro.jsonl",
                        {
                            "round": repetition,
                            "window": window,
                            "position": position,
                            "setting": label,
                            "ms": elapsed,
                            "logits_finite": logits_finite,
                            "logits_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                            "argmax": raw.argmax(-1).tolist(),
                        },
                    )
                    if not logits_finite:
                        raise FloatingPointError(
                            f"Nonfinite logits at {label}, window {window}, round {repetition}"
                        )
                    for cache in e.attn.values():
                        cache.trim(window)
                    del logits, w, raw
        phase("micro-end")
        phase("answers-start")
        for repetition in range(a.answer_rounds):
            for case in fixtures:
                order = list(settings)
                rng.shuffle(order)
                for position, label in enumerate(order):
                    configure(settings[label][0])
                    e = engines[label]
                    e.reset()
                    e.rows = eng.GuessRows(model.lm_head, model.model.args.vocab_size)
                    e.sampling = Sampling(temperature=0, seed=a.seed)
                    if a.answer_policy == "reset":
                        e.policy = eng.Policy()
                    policy_before = json.loads(json.dumps(vars(e.policy)))
                    ids = encoded_fixtures[case["name"]]
                    metrics_before = expert_metrics(model)
                    phase(
                        "request-start",
                        round=repetition,
                        case=case["name"],
                        setting=label,
                    )
                    start = time.perf_counter()
                    tokens = [e.read(ids)]
                    ttft = time.perf_counter() - start
                    visible = None
                    if clean(tok.decode(tokens, skip_special_tokens=False)):
                        visible = ttft
                    while len(tokens) < a.max_tokens and tokens[-1] not in stop:
                        tokens += e.step(a.max_tokens - len(tokens), stop)
                        if visible is None and clean(
                            tok.decode(tokens, skip_special_tokens=False)
                        ):
                            visible = time.perf_counter() - start
                    wall = time.perf_counter() - start
                    text = clean(tok.decode(tokens, skip_special_tokens=False))
                    try:
                        correct = (
                            json_equal(json.loads(text), case["expected"])
                            if "expected" in case
                            else None
                        )
                    except (ValueError, TypeError):
                        correct = False
                    metrics_after = expert_metrics(model)
                    append(
                        "answers.jsonl",
                        {
                            "round": repetition,
                            "case": case["name"],
                            "position": position,
                            "setting": label,
                            "prompt_tokens": len(ids),
                            "prompt_ids": ids,
                            "output_ids": tokens,
                            "text": text,
                            "correct": correct,
                            "expected": case.get("expected"),
                            "policy_before": policy_before,
                            "policy_after": vars(e.policy),
                            "ttft_s": ttft,
                            "first_visible_s": visible,
                            "wall_s": wall,
                            "decode_tps": (len(tokens) - 1) / (wall - ttft)
                            if wall > ttft
                            else None,
                            "finish": "stop" if tokens[-1] in stop else "length",
                            "stats": vars(e.stats),
                            "expert_metrics": metrics_after,
                            "expert_metrics_delta": {
                                key: metrics_after[key] - value
                                for key, value in metrics_before.items()
                            },
                        },
                    )
                    phase(
                        "request-end",
                        round=repetition,
                        case=case["name"],
                        setting=label,
                    )
        phase("answers-end")
    finally:
        for module, name, value in originals:
            setattr(module, name, value)
        if previous_cache_limit is not None:
            mx.set_cache_limit(previous_cache_limit)


if __name__ == "__main__":
    raise SystemExit(main())
