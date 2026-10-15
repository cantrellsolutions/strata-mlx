"""Validate complete shared-weight runs before calculating descriptive paired summaries.

Legacy runs require explicit plan arguments; the analysis never infers a successful
plan from whatever rows happened to survive a crash.
"""

import argparse
import itertools
import json
import math
import random
import statistics
from pathlib import Path


def json_equal(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            json_equal(actual[k], expected[k]) for k in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            map(lambda pair: json_equal(*pair), zip(actual, expected))
        )
    return actual == expected


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def interval(values, seed=20261015):
    rng = random.Random(seed)
    bootstrap = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)
    ]
    return [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]


def read_rows(path):
    try:
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    except (ValueError, OSError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def unique(rows, fields, expected, name):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in indexed:
            raise ValueError(f"Duplicate {name} sample {key}")
        indexed[key] = row
    if set(indexed) != expected:
        missing, extra = expected - set(indexed), set(indexed) - expected
        raise ValueError(
            f"Incomplete {name}: {len(missing)} missing, {len(extra)} unexpected samples"
        )
    return indexed


def positive(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be finite and positive")


def summarize(
    path,
    *,
    micro_rounds=None,
    answer_rounds=None,
    windows=None,
    fixtures=None,
    baseline="new6",
    control="control6",
):
    path = Path(path)
    config = json.loads((path / "configuration.json").read_text())
    plan = config.get("plan", {})
    micro_rounds = (
        micro_rounds if micro_rounds is not None else plan.get("micro_rounds")
    )
    answer_rounds = (
        answer_rounds if answer_rounds is not None else plan.get("answer_rounds")
    )
    windows = windows if windows is not None else plan.get("windows")
    fixtures = fixtures if fixtures is not None else config.get("fixtures")
    if (
        not isinstance(micro_rounds, int)
        or isinstance(micro_rounds, bool)
        or micro_rounds < 2
    ):
        raise ValueError("Supply a micro-round plan of at least two rounds")
    if (
        not isinstance(answer_rounds, int)
        or isinstance(answer_rounds, bool)
        or answer_rounds < 1
    ):
        raise ValueError("Supply a positive answer-round plan")
    if (
        not windows
        or len(windows) != len(set(windows))
        or any(type(w) is not int or w < 1 for w in windows)
    ):
        raise ValueError("Supply distinct positive planned windows")
    if not fixtures or any("name" not in case for case in fixtures):
        raise ValueError("Supply the planned named fixtures")
    cases = {case["name"]: case for case in fixtures}
    if len(cases) != len(fixtures):
        raise ValueError("Duplicate planned fixture names")
    settings = list(config["settings"])
    if baseline not in settings or control not in settings or baseline == control:
        raise ValueError("Baseline and distinct duplicate control must exist")
    if config["settings"][baseline] != config["settings"][control]:
        raise ValueError("Control configuration must equal baseline")
    micro = unique(
        read_rows(path / "micro.jsonl"),
        ("round", "window", "setting"),
        set(itertools.product(range(micro_rounds), windows, settings)),
        "micro",
    )
    answers = unique(
        read_rows(path / "answers.jsonl"),
        ("round", "case", "setting"),
        set(itertools.product(range(answer_rounds), cases, settings)),
        "answer",
    )
    for row in micro.values():
        positive(row["ms"], "micro ms")
        if not row.get("logits_sha256") or "argmax" not in row:
            raise ValueError("Missing numerical comparison evidence")
        if row.get("logits_finite") is False:
            raise ValueError("Nonfinite logits in fixed-pass evidence")
    prompt_by_case = {}
    completion_checked = "max_tokens" in config and "stop_ids" in config
    for row in answers.values():
        positive(row["wall_s"], "answer wall time")
        positive(row["ttft_s"], "answer first-token time")
        if row["ttft_s"] > row["wall_s"] or row["finish"] not in ("stop", "length"):
            raise ValueError("Inconsistent answer timing or completion")
        if not row["output_ids"] or any(
            type(i) is not int or i < 0 for i in row["output_ids"]
        ):
            raise ValueError("Missing or invalid output token IDs")
        visible, rate = row["first_visible_s"], row["decode_tps"]
        if visible is not None and (
            isinstance(visible, bool)
            or not isinstance(visible, (int, float))
            or not math.isfinite(visible)
            or not 0 <= visible <= row["wall_s"]
        ):
            raise ValueError("Invalid first-visible timing")
        if rate is not None and (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate < 0
        ):
            raise ValueError("Invalid decode rate")
        prompt = row.get("prompt_ids")
        if (
            not prompt
            or any(type(i) is not int or i < 0 for i in prompt)
            or type(row.get("prompt_tokens")) is not int
            or row["prompt_tokens"] != len(prompt)
        ):
            raise ValueError("Invalid prompt token evidence")
        if row["case"] in prompt_by_case and prompt_by_case[row["case"]] != prompt:
            raise ValueError("Mismatched paired prompt IDs")
        prompt_by_case[row["case"]] = prompt
        if completion_checked:
            tokens, stops, cap = (
                row["output_ids"],
                config["stop_ids"],
                config["max_tokens"],
            )
            if (
                type(cap) is not int
                or cap < 1
                or not stops
                or any(type(i) is not int for i in stops)
            ):
                raise ValueError("Invalid completion protocol")
            if len(tokens) > cap or any(i in stops for i in tokens[:-1]):
                raise ValueError("Output violates completion protocol")
            if (row["finish"] == "stop" and tokens[-1] not in stops) or (
                row["finish"] == "length"
                and (len(tokens) != cap or tokens[-1] in stops)
            ):
                raise ValueError("Completion label contradicts token evidence")
        case = cases[row["case"]]
        if "expected" in case:
            try:
                valid = json_equal(json.loads(row["text"]), case["expected"])
            except ValueError:
                valid = False
            if row["correct"] is not valid:
                raise ValueError(
                    f"Stored grade differs from type-sensitive JSON validation: {row['case']}"
                )
        elif row["correct"] is not None:
            raise ValueError("Ungraded throughput case has a correctness score")
    result = {
        "completion_protocol_validated": completion_checked,
        "logits_finiteness_validated": all(
            row.get("logits_finite") is True for row in micro.values()
        ),
        "validation": "complete unique evidence; all planned cells present",
        "configuration": config,
        "interval_limit": "20-style repeated-pair bootstrap is descriptive for this fixed fixture; "
        "does not cover independent-prefix, thermal or cross-day uncertainty.",
        "micro": [],
        "answers": {},
        "answer_parity": [],
    }
    for window in windows:
        pairs = [
            {s: micro[n, window, s] for s in settings} for n in range(micro_rounds)
        ]
        noise = [p[baseline]["ms"] - p[control]["ms"] for p in pairs]
        for setting in settings:
            if setting == baseline:
                continue
            differences = [p[baseline]["ms"] - p[setting]["ms"] for p in pairs]
            result["micro"].append(
                {
                    "window": window,
                    "setting": setting,
                    "paired_rounds": micro_rounds,
                    "median_saving_ms": statistics.median(differences),
                    "bootstrap_95_median_interval_ms": interval(differences),
                    "duplicate_control_median_ms": statistics.median(noise),
                    "duplicate_control_abs_p95_ms": percentile(
                        list(map(abs, noise)), 0.95
                    ),
                    "logits_bit_equal_to_baseline": all(
                        p[baseline]["logits_sha256"] == p[setting]["logits_sha256"]
                        for p in pairs
                    ),
                    "argmax_equal_to_baseline": all(
                        p[baseline]["argmax"] == p[setting]["argmax"] for p in pairs
                    ),
                }
            )
    for setting in settings:
        rows = [r for r in answers.values() if r["setting"] == setting]
        graded = [r for r in rows if r["correct"] is not None]
        result["answers"][setting] = {
            "requests": len(rows),
            "graded": len(graded),
            "correct": sum(r["correct"] for r in graded),
            "completion_limits": sum(r["finish"] == "length" for r in rows),
            "per_case": {
                case: {
                    field: [
                        answers[n, case, setting][field] for n in range(answer_rounds)
                    ]
                    for field in ("wall_s", "ttft_s", "first_visible_s", "decode_tps")
                }
                for case in cases
            },
        }
    for n, case, setting in itertools.product(range(answer_rounds), cases, settings):
        if setting == baseline:
            continue
        a, b = answers[n, case, baseline], answers[n, case, setting]
        result["answer_parity"].append(
            {
                "round": n,
                "case": case,
                "setting": setting,
                "output_ids_equal_to_baseline": a["output_ids"] == b["output_ids"],
                "visible_text_equal_to_baseline": a["text"] == b["text"],
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        type=Path,
        help="One model run containing configuration and raw JSONL files",
    )
    parser.add_argument("--micro-rounds", type=int)
    parser.add_argument("--answer-rounds", type=int)
    parser.add_argument("--windows", nargs="+", type=int)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New summary path; never overwrites existing evidence",
    )
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text()) if args.fixtures else None
    result = summarize(
        args.run,
        micro_rounds=args.micro_rounds,
        answer_rounds=args.answer_rounds,
        windows=args.windows,
        fixtures=fixtures,
    )
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
