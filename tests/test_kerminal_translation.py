"""Offline checks for preservation of the synthetic five-round translation."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.kerminal_translate_smoke import validate_translation


FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "jev_smoke"


class KerminalTranslationTests(unittest.TestCase):
    def setUp(self):
        self.chinese = (FIXTURE / "evidence.zh.md").read_text(encoding="utf-8-sig")
        self.evidence = json.loads((FIXTURE / "evidence.en.json").read_text(encoding="utf-8-sig"))
        # The translation protocol requires an explicit limitation field, beyond
        # the older English fixture's combined finding/uncertainty description.
        self.evidence["language"] = "en"
        self.evidence["sample_kind"] = "synthetic_connectivity_fixture_not_real_performance_evidence"
        self.evidence["evidence_status"] = (
            "Synthetic records, not executed on an NPU, and not real operator performance; "
            "for translation and connectivity testing only."
        )
        self.evidence["profiling_analysis"]["limitation"] = (
            "These synthetic observations do not establish that one kernel is faster."
        )
        self.evidence["measurement_protocol"] = {
            "provider": "cann-bench",
            "strategy": "kernel_details",
            "scope": "Sum of per-kernel median execution times; excludes gaps between kernels.",
            "metric_policy": "Preserve the tool's baseline, speedup, HAP and aggregate scores without recomputing.",
        }
        launch = patch("tools.kerminal_translate_smoke.subprocess.Popen")
        self.cli = launch.start()
        self.addCleanup(launch.stop)
        self.addCleanup(self.cli.assert_not_called)

    def test_valid_fixture_passes_without_claiming_semantic_review(self):
        original = copy.deepcopy(self.evidence)
        result = validate_translation(self.evidence, self.chinese)
        self.assertEqual(result, {
            "structure": "passed",
            "numeric_values": "passed",
            "iteration_count": 5,
            "chinese_text_absent": True,
            "semantic_review": "required_before_jev",
        })
        self.assertEqual(self.evidence, original)

    def test_missing_added_duplicate_or_reordered_rounds_are_rejected(self):
        history = self.evidence["iteration_history"]
        invalid_histories = (
            history[:-1],
            history + [history[-1]],
            [history[1], history[0], *history[2:]],
            [history[0], history[0], *history[2:]],
        )
        for records in invalid_histories:
            with self.subTest(iterations=[item["iteration"] for item in records]):
                evidence = copy.deepcopy(self.evidence)
                evidence["iteration_history"] = copy.deepcopy(records)
                with self.assertRaises(ValueError):
                    validate_translation(evidence, self.chinese)

    def test_changed_numbers_and_booleans_masquerading_as_one_are_rejected(self):
        mutations = (
            lambda data: data["iteration_history"][0].update(iteration=True),
            lambda data: data["iteration_history"][3]["measured_effect"].update(min_case_speedup=True),
            lambda data: data["iteration_history"][0]["measured_effect"].update(geomean_speedup=1.11),
            lambda data: data["iteration_history"][4]["measured_effect"].update(min_case_speedup=1.02),
            lambda data: data["iteration_history"][2]["measured_effect"].update(geomean_speedup="1.14"),
            lambda data: data["iteration_history"][1]["measured_effect"].update(geomean_speedup=float("nan")),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                evidence = copy.deepcopy(self.evidence)
                mutate(evidence)
                with self.assertRaises(ValueError):
                    validate_translation(evidence, self.chinese)

    def test_chinese_in_nested_values_or_keys_is_rejected(self):
        mutations = (
            lambda data: data["iteration_history"][4].update(adoption_outcome="Accepted as 当前版本"),
            lambda data: data["profiling_analysis"].update(limitation="尚未证明单 kernel 更快"),
            lambda data: data.update({"中文字段": "English value"}),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                evidence = copy.deepcopy(self.evidence)
                mutate(evidence)
                with self.assertRaisesRegex(ValueError, "Chinese text remains"):
                    validate_translation(evidence, self.chinese)

    def test_missing_or_empty_measurement_and_uncertainty_fields_are_rejected(self):
        for value in (None, "", " \n\t", False, 1, {}):
            with self.subTest(evidence_status=value):
                evidence = copy.deepcopy(self.evidence)
                if value is None:
                    evidence.pop("evidence_status")
                else:
                    evidence["evidence_status"] = value
                with self.assertRaisesRegex(ValueError, "explicit evidence-status explanation"):
                    validate_translation(evidence, self.chinese)
        required = {
            "profiling_analysis": ("finding", "limitation", "next_validation"),
            "measurement_protocol": ("scope", "metric_policy"),
        }
        for section, fields in required.items():
            for field in fields:
                for missing in (True, False):
                    with self.subTest(section=section, field=field, missing=missing):
                        evidence = copy.deepcopy(self.evidence)
                        if missing:
                            evidence[section].pop(field)
                        else:
                            evidence[section][field] = " \n\t"
                        with self.assertRaisesRegex(ValueError, "omitted a measurement or profiling explanation"):
                            validate_translation(evidence, self.chinese)
        for field, replacement in (("provider", "other-tool"), ("strategy", "trace_view")):
            with self.subTest(protocol_field=field):
                evidence = copy.deepcopy(self.evidence)
                evidence["measurement_protocol"][field] = replacement
                with self.assertRaisesRegex(ValueError, "measurement provider or strategy"):
                    validate_translation(evidence, self.chinese)

    def test_translator_cannot_add_or_override_kernel_source(self):
        for kernel in (None, {}, {"source": "def changed_kernel(): pass"}):
            with self.subTest(kernel=kernel):
                evidence = copy.deepcopy(self.evidence)
                evidence["kernel"] = kernel
                with self.assertRaisesRegex(ValueError, "must not replace source code"):
                    validate_translation(evidence, self.chinese)


if __name__ == "__main__":
    unittest.main()
