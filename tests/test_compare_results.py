"""CPU-only regression cases for evidence loss, duplicate reruns and strict JSON types."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compare_results", Path(__file__).parents[1] / "bench/compare_results.py"
)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class EvidenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = {"new6": [True, 6], "control6": [True, 6], "new8": [True, 8]}
        self.config = {
            "settings": self.settings,
            "plan": {"micro_rounds": 2, "answer_rounds": 1, "windows": [1, 4]},
            "fixtures": [{"name": "tool", "expected": {"recursive": False}}],
        }
        self.micro = [
            {
                "round": n,
                "window": w,
                "setting": s,
                "ms": 10.0,
                "logits_sha256": "fixed-hash",
                "argmax": [[7]],
            }
            for n in range(2)
            for w in (1, 4)
            for s in self.settings
        ]
        self.answers = [
            {
                "round": 0,
                "case": "tool",
                "setting": s,
                "wall_s": 1.0,
                "ttft_s": 0.1,
                "first_visible_s": 0.1,
                "decode_tps": 2.0,
                "output_ids": [7, 8],
                "finish": "stop",
                "text": '{"recursive":false}',
                "correct": True,
                "prompt_ids": [1, 2],
                "prompt_tokens": 2,
            }
            for s in self.settings
        ]

    def write(self):
        (self.root / "configuration.json").write_text(json.dumps(self.config))
        for name, rows in [("micro", self.micro), ("answers", self.answers)]:
            (self.root / (name + ".jsonl")).write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )

    def summary(self):
        self.write()
        return compare.summarize(self.root)

    def test_complete_identical_controls_have_zero_saving(self):
        result = self.summary()
        self.assertTrue(all(x["median_saving_ms"] == 0 for x in result["micro"]))
        self.assertEqual(result["answers"]["new6"]["correct"], 1)

    def test_duplicate_append_is_rejected(self):
        self.micro.append(self.micro[0].copy())
        with self.assertRaisesRegex(ValueError, "Duplicate micro"):
            self.summary()

    def test_incomplete_round_is_rejected(self):
        self.micro.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete micro"):
            self.summary()

    def test_missing_answer_cannot_disappear_from_pairs(self):
        self.answers.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete answer"):
            self.summary()

    def test_nonfinite_timing_is_rejected(self):
        for value in [float("nan"), float("inf"), -1.0, 0.0]:
            with self.subTest(value=value):
                self.micro[0]["ms"] = value
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    self.summary()

    def test_legacy_plan_must_be_explicit(self):
        self.config.pop("plan")
        with self.assertRaisesRegex(ValueError, "micro-round plan"):
            self.summary()
        result = compare.summarize(
            self.root, micro_rounds=2, answer_rounds=1, windows=[1, 4]
        )
        self.assertEqual(len(result["micro"]), 4)

    def test_boolean_numeric_confusion_does_not_pass(self):
        self.answers[0]["text"] = '{"recursive":0}'
        with self.assertRaisesRegex(ValueError, "Stored grade differs"):
            self.summary()
        self.answers[0]["correct"] = False
        self.assertEqual(self.summary()["answers"]["new6"]["correct"], 0)

    def test_markdown_json_remains_a_failure(self):
        self.answers[0].update(text='```json\n{"recursive":false}\n```', correct=False)
        self.assertEqual(self.summary()["answers"]["new6"]["correct"], 0)

    def test_json_number_equivalence_without_boolean_equivalence(self):
        self.assertTrue(compare.json_equal({"x": [1.0, None]}, {"x": [1, None]}))
        self.assertFalse(compare.json_equal({"x": [True]}, {"x": [1]}))
        self.assertFalse(compare.json_equal([False], [0]))

    def test_invalid_visible_and_decode_evidence_is_rejected(self):
        for field, value in [
            ("first_visible_s", -1.0),
            ("first_visible_s", 2.0),
            ("first_visible_s", float("nan")),
            ("decode_tps", -1.0),
            ("decode_tps", float("inf")),
        ]:
            with self.subTest(field=field, value=value):
                old = self.answers[0][field]
                self.answers[0][field] = value
                with self.assertRaises(ValueError):
                    self.summary()
                self.answers[0][field] = old

    def test_different_prompt_work_cannot_be_paired(self):
        self.answers[1]["prompt_ids"] = [3, 4]
        with self.assertRaisesRegex(ValueError, "Mismatched paired prompt"):
            self.summary()

    def test_early_length_completion_is_rejected(self):
        self.config.update(max_tokens=10, stop_ids=[8])
        self.answers[0].update(output_ids=[7], finish="length")
        with self.assertRaisesRegex(ValueError, "Completion label"):
            self.summary()

    def test_legacy_validation_limits_are_explicit(self):
        result = self.summary()
        self.assertFalse(result["completion_protocol_validated"])
        self.assertFalse(result["logits_finiteness_validated"])

    def test_empty_or_mismatched_control_is_rejected(self):
        self.config["settings"]["control6"] = [True, 8]
        with self.assertRaisesRegex(ValueError, "Control configuration"):
            self.summary()


if __name__ == "__main__":
    unittest.main()
