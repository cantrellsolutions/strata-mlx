"""CPU-only evidence/configuration contracts; no MLX imports or model loads."""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "bench/experiments/e49_shared_weight_controls.py"
)
spec = importlib.util.spec_from_file_location("shared_controls", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class TestSharedControls(unittest.TestCase):
    def test_import_and_help_need_no_accelerator(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--answer-policy", result.stdout)

    def test_invalid_rounds_windows_and_caps_fail_before_run(self):
        base = [
            part
            for name in ("model", "mtp", "tokenizer", "fixtures", "output")
            for part in ("--" + name, "/unused")
        ]
        for flag, value in [
            ("--micro-rounds", "0"),
            ("--answer-rounds", "-1"),
            ("--max-tokens", "0"),
            ("--windows", "1,1"),
            ("--windows", "9"),
            ("--windows", "one"),
            ("--cache-mib", "0"),
        ]:
            with (
                self.subTest(flag=flag, value=value),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    experiment.parser().parse_args(base + [flag, value])

    def test_existing_output_refused_even_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "run"
            experiment.prepare_output(output)
            with self.assertRaises(FileExistsError):
                experiment.prepare_output(output)
            (output / "micro.jsonl").write_text("prior evidence\n")
            with self.assertRaises(FileExistsError):
                experiment.prepare_output(output)
            self.assertEqual((output / "micro.jsonl").read_text(), "prior evidence\n")

    def test_fixture_names_unique_and_prompts_required(self):
        cases = [
            [],
            [{"name": "x", "prompt": ""}],
            [{"name": "x", "prompt": "ok"}, {"name": "x", "prompt": "ok"}],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixtures.json"
            for value in cases:
                path.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    experiment.load_fixtures(path)
            valid = [{"name": "x", "prompt": "ok", "expected": [False, 1]}]
            path.write_text(json.dumps(valid))
            self.assertEqual(experiment.load_fixtures(path), valid)

    def test_boolean_number_mismatches_never_pass(self):
        self.assertFalse(experiment.json_equal({"recursive": 0}, {"recursive": False}))
        self.assertFalse(experiment.json_equal([True, 1], [1, 1]))
        self.assertFalse(experiment.json_equal({"x": 1}, {"x": "1"}))
        self.assertTrue(experiment.json_equal({"x": [False, 32.0]}, {"x": [False, 32]}))

    def test_unclosed_thinking_is_not_visible_answer(self):
        self.assertEqual(experiment.clean("<think>hidden reasoning"), "")
        self.assertEqual(
            experiment.clean("<think>hidden</think> visible<|im_end|>"), "visible"
        )
        self.assertEqual(experiment.clean("plain answer"), "plain answer")

    def test_stop_ids_validated_by_token_strings(self):
        tokenizer = mock.Mock()
        tokenizer.token_to_id.side_effect = {"end": 42, "turn": 43}.get
        self.assertEqual(experiment.stop_ids(tokenizer, ["end", "turn"]), (42, 43))
        for values in [[], ["missing"], ["end", "end"]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                experiment.stop_ids(tokenizer, values)

    def test_failure_records_phase_and_protects_previous_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ["model", "tokenizer"]:
                (root / name).write_text("dummy")
            (root / "fixtures").write_text('[{"name":"x","prompt":"ok"}]')
            (root / "mtp").mkdir()
            args = [
                part
                for name in ["model", "tokenizer", "fixtures", "mtp", "output"]
                for part in ["--" + name, str(root / name)]
            ]
            with mock.patch.object(
                experiment, "run", side_effect=RuntimeError("synthetic failure")
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    experiment.main(args)
                rows = [
                    json.loads(line)
                    for line in (root / "output" / "phase-events.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual([r["phase"] for r in rows], ["run-start", "failed"])
                self.assertEqual(rows[-1]["error"], "synthetic failure")
                with self.assertRaises(FileExistsError):
                    experiment.main(args)
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
