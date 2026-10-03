"""Offline source/evidence binding and fusion-choice integrity checks."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

from lib.fusion_evidence import (
    DECISION_FILE, begin_development, development_prompt, finalize_development,
    format_evidence_for_prompt, implementation_hash, load_evidence,
)


class FusionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.develop = self.work / "develop/iter1"
        self.develop.mkdir(parents=True)
        (self.work / "impl").mkdir()
        (self.work / "impl/kernel.py").write_text("def kernel(x): return x + 1\n", encoding="utf-8")
        self.initial = {"schema_version": 1, "candidates": [
            {"rank": 1, "method": {"id": "pipeline", "name": "Pipeline"}, "probability": 0.8},
            {"rank": 2, "method": {"id": "separate", "name": "Separate kernels"}, "probability": 0.6},
        ]}
        self.library = {**deepcopy(self.initial), "selection": {
            "method_ids": ["pipeline"], "implementation_plan": "A tiled pipeline.",
            "reason": "The hardware supports it.", "target_cases": ["all"],
            "actual_changes": ["Tile the reduction."], "expected_benefits": ["Less traffic."],
        }}
        self.self_test = {"schema_version": 1,
            "provided_cases": {"executed": True, "passed": True, "total": 2,
                               "passed_cases": 2, "evidence_path": "self_test.log"},
            "continuous_calls": {"executed": True, "passed": True, "same_shape": True,
                "call_count": 2, "reference_checked_each_call": True, "evidence_path": "self_test.log",
                "changes": {"inputs": {"applicable": True, "changed": True, "passed": True},
                    "weights": {"applicable": False, "reason": "The interface has no weight argument."},
                    "bias": {"applicable": False, "reason": "The interface has no bias argument."}}}}
        self.write_json(self.work / "fusion/fusion_library.json", self.initial)
        self.write_artifacts()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def write_artifacts(self):
        self.write_json(self.develop / "fusion_library.json", self.library)
        self.write_json(self.develop / "self_test_result.json", self.self_test)
        for name in (DECISION_FILE, "design_rationale.md", "self_test_report.md", "self_test.log"):
            (self.develop / name).write_text("Actual decision and test evidence.\n", encoding="utf-8")

    def finish(self, stage=3, **kwargs):
        return finalize_development(self.work, self.develop, stage, **kwargs)

    def test_success_binds_current_source_without_creating_a_best_record(self):
        result = self.finish()
        self.assertTrue(result["eligible"])
        self.assertEqual(result["impl_sha256"], implementation_hash(self.work / "impl"))
        self.assertEqual(load_evidence(self.work)["implementation_plan"], "A tiled pipeline.")
        self.assertTrue(result["evidence_paths"]["decision_rationale"].endswith(DECISION_FILE))
        self.assertFalse((self.work / "selection/best.json").exists())
        self.assertIn("Actual decision", format_evidence_for_prompt(self.work))

    def test_first_implementation_uses_existing_design_rationale(self):
        (self.develop / DECISION_FILE).unlink()
        result = self.finish(stage=2)
        self.assertTrue(result["eligible"])
        self.assertTrue(result["evidence_paths"]["decision_rationale"].endswith("design_rationale.md"))
        self.assertFalse(self.finish(stage=3)["eligible"])

    def test_new_method_is_allowed_only_without_an_invented_probability(self):
        self.library["candidates"].append({"method": {"id": "new", "name": "Measured hybrid"}, "probability": None})
        self.library["selection"]["method_ids"] = ["new"]
        self.write_artifacts()
        self.assertTrue(self.finish()["eligible"])
        for probability in (0.9, True, "unscored"):
            with self.subTest(probability=probability):
                self.library["candidates"][-1]["probability"] = probability
                self.write_artifacts()
                self.assertFalse(self.finish()["eligible"])

    def test_initial_probabilities_and_method_definitions_cannot_be_rewritten(self):
        original = deepcopy(self.library)
        for mutation in ("probability", "method", "missing", "duplicate"):
            with self.subTest(mutation=mutation):
                self.library = deepcopy(original)
                if mutation == "probability":
                    self.library["candidates"][0]["probability"] = 0.99
                elif mutation == "method":
                    self.library["candidates"][0]["method"]["name"] = "Different design"
                elif mutation == "missing":
                    self.library["candidates"].pop()
                else:
                    self.library["candidates"].append(deepcopy(self.library["candidates"][0]))
                self.write_artifacts()
                self.assertFalse(self.finish()["eligible"])

    def test_self_test_booleans_are_strict_and_missing_runs_are_not_success(self):
        original = deepcopy(self.self_test)
        for field, value in (("executed", False), ("executed", "true"), ("passed", 1),
                             ("same_shape", False), ("reference_checked_each_call", False),
                             ("call_count", True), ("call_count", 1)):
            with self.subTest(field=field, value=value):
                self.self_test = deepcopy(original)
                self.self_test["continuous_calls"][field] = value
                self.write_artifacts()
                self.assertFalse(self.finish()["eligible"])

    def test_applicable_parameters_need_new_values_and_inapplicable_need_evidence(self):
        self.self_test["continuous_calls"]["changes"]["weights"] = {"applicable": True, "changed": False, "passed": True}
        self.write_artifacts()
        self.assertFalse(self.finish()["eligible"])
        self.self_test["continuous_calls"]["changes"]["weights"] = {"applicable": False}
        self.write_artifacts()
        self.assertFalse(self.finish()["eligible"])
        self.self_test["continuous_calls"]["changes"]["weights"] = {"applicable": True, "changed": True, "passed": True}
        self.write_artifacts()
        self.assertTrue(self.finish()["eligible"])

    def test_code_changes_after_development_invalidate_self_tests(self):
        self.assertTrue(self.finish()["eligible"])
        (self.work / "impl/kernel.py").write_text("def kernel(x): return x + 2\n", encoding="utf-8")
        self.assertFalse(load_evidence(self.work)["eligible"])
        self.assertIn("Stage3", format_evidence_for_prompt(self.work))

    def test_build_outputs_do_not_invalidate_unchanged_source(self):
        self.assertTrue(self.finish()["eligible"])
        for folder in ("build", "dist", "__pycache__", "pkg.egg-info"):
            target = self.work / "impl" / folder
            target.mkdir()
            (target / "artifact").write_bytes(b"generated")
        self.assertTrue(load_evidence(self.work)["eligible"])
        (self.work / "impl/dispatcher.py").write_text("def dispatch(): pass\n", encoding="utf-8")
        self.assertFalse(load_evidence(self.work)["eligible"])

    def test_changed_rationale_log_or_initial_library_invalidates_binding(self):
        for path in (self.develop / DECISION_FILE, self.develop / "self_test.log",
                     self.work / "fusion/fusion_library.json"):
            with self.subTest(path=path.name):
                self.write_json(self.work / "fusion/fusion_library.json", self.initial)
                self.write_artifacts()
                self.assertTrue(self.finish()["eligible"])
                path.write_text("Changed\n", encoding="utf-8")
                self.assertFalse(load_evidence(self.work)["eligible"])

    def test_failed_agent_and_missing_log_invalidate_prior_success(self):
        self.assertTrue(self.finish()["eligible"])
        self.assertFalse(self.finish(agent_ok=False)["eligible"])
        self.assertFalse(load_evidence(self.work)["eligible"])
        (self.develop / "self_test.log").unlink()
        self.assertFalse(self.finish()["eligible"])

    def test_invalid_self_test_keeps_bound_choice_available_to_reviewers(self):
        self.self_test["continuous_calls"]["passed"] = False
        self.write_artifacts()
        self.assertFalse(self.finish()["eligible"])
        result = load_evidence(self.work)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["fusion_scheme"]["method_ids"], ["pipeline"])
        self.assertIn("Actual decision", format_evidence_for_prompt(self.work))
        self.assertIn("False", format_evidence_for_prompt(self.work))

    def test_initial_library_cache_metadata_does_not_change_semantic_binding(self):
        self.assertTrue(self.finish()["eligible"])
        self.initial.update(cache_reused=True, request_budget={"request_bytes": 1234})
        self.write_json(self.work / "fusion/fusion_library.json", self.initial)
        self.assertTrue(load_evidence(self.work)["eligible"])

    def test_prompt_separates_initial_and_round_libraries_and_no_run_claims(self):
        prompt = development_prompt(self.work, self.develop, 3)
        self.assertIn(str(self.work / "fusion/fusion_library.json"), prompt)
        self.assertIn(str(self.develop), prompt)
        self.assertIn(DECISION_FILE, prompt)
        self.assertIn("probability 必须为 null", prompt)
        self.assertIn("未运行或失败如实写 false", prompt)

    def test_nonfinite_extra_metadata_is_invalid_without_breaking_failure_routes(self):
        self.library["selection"]["untrusted_metric"] = float("nan")
        self.write_artifacts()
        self.assertFalse(self.finish()["eligible"])
        self.assertFalse(load_evidence(self.work)["eligible"])

    def test_rerun_cannot_bind_old_reports_to_changed_code(self):
        self.assertTrue(self.finish()["eligible"])
        # A previous log can live outside this develop directory, within work.
        outside_log = self.work / "eval/old_self_test.log"
        outside_log.parent.mkdir()
        outside_log.write_text("Old executed tests.\n", encoding="utf-8")
        for field in ("provided_cases", "continuous_calls"):
            self.self_test[field]["evidence_path"] = str(outside_log)
        self.write_artifacts()
        token = begin_development(self.work, self.develop)
        self.assertFalse(load_evidence(self.work)["eligible"])
        self.assertTrue((self.develop / "self_test_report.md").is_file())
        (self.work / "impl/kernel.py").write_text("def kernel(x): return x + 2\n", encoding="utf-8")
        rejected = self.finish(previous_revisions=token)
        self.assertFalse(rejected["eligible"])
        self.assertEqual(rejected["fusion_scheme"], {})
        self.assertIn("Stale", rejected["reason"])
        # Updating docs alone still cannot reuse their prior external log.
        self.write_artifacts()
        for path in self.develop.iterdir():
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        rejected = self.finish(previous_revisions=token)
        self.assertFalse(rejected["eligible"])
        self.assertIn("old_self_test.log", rejected["reason"])

    def test_new_artifacts_and_rewritten_prior_evidence_pass_run_boundary(self):
        fresh = self.work / "develop/iter2"
        token = begin_development(self.work, fresh)
        self.develop = fresh
        self.develop.mkdir()
        self.write_artifacts()
        self.assertTrue(self.finish(previous_revisions=token)["eligible"])
        token = begin_development(self.work, self.develop)
        (self.work / "impl/kernel.py").write_text("def kernel(x): return x + 2\n", encoding="utf-8")
        self.write_artifacts()
        # Explicitly advance timestamps to avoid filesystem clock granularity
        # affecting this fixture when a new real run produces identical text.
        for path in self.develop.iterdir():
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertTrue(self.finish(previous_revisions=token)["eligible"])
        self.assertTrue(load_evidence(self.work)["eligible"])

    def test_legacy_without_stage15_accepts_new_unscored_choices(self):
        (self.work / "fusion/fusion_library.json").unlink()
        (self.work / "fusion").rmdir()
        for candidate in self.library["candidates"]:
            candidate["probability"] = None
        self.write_artifacts()
        self.assertTrue(self.finish()["eligible"])
        self.assertTrue(load_evidence(self.work)["eligible"])
        self.assertIn("旧任务", development_prompt(self.work, self.develop, 3))
        self.assertIn("不补调 Jev", development_prompt(self.work, self.develop, 3))
        self.library["candidates"][0]["probability"] = 0.8
        self.write_artifacts()
        self.assertFalse(self.finish()["eligible"])

    def test_missing_initial_library_is_not_legacy_if_stage15_was_started(self):
        (self.work / "fusion/fusion_library.json").unlink()
        for candidate in self.library["candidates"]:
            candidate["probability"] = None
        self.write_artifacts()
        self.assertFalse(self.finish()["eligible"])
        (self.work / "fusion").rmdir()
        (self.work / "fusion_requirements.en.json").write_text("{}", encoding="utf-8")
        self.assertFalse(self.finish()["eligible"])


if __name__ == "__main__":
    unittest.main()
