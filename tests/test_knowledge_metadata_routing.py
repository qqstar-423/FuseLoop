"""Knowledge environment labels must survive actual Stage9 storage and logging."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from lib.history_manager import (load_history, load_pitfalls, load_proven_patterns,
                                 load_regression_patterns)
import test_semantic_routing as semantic_routing_tests


class KnowledgeMetadataRoutingTests(unittest.TestCase):
    def setUp(self):
        self.routing = semantic_routing_tests.SemanticRoutingTests(methodName="runTest")
        self.routing.setUp()
        self.addCleanup(self.routing.doCleanups)
        self.work = self.routing.work
        self.routing.config["workflow"]["semantic_exit"].update(
            passed_window=8, underperforming_window=8)

    def read_json(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def ledger(self, iteration):
        return next(entry for entry in load_history(str(self.work))["ledger"]
                    if entry["iter"] == iteration)

    def assert_environment(self, environment, *, source_kind, source_path):
        self.assertEqual(environment["framework"], "Triton")
        self.assertEqual(environment["framework_source"],
                         "performance_report" if source_kind == "performance_report" else "workflow_target")
        self.assertEqual(environment["chip_model"], "offline_test")
        self.assertEqual(environment["soc_version"], "offline_test")
        self.assertEqual(environment["source_kind"], source_kind)
        self.assertEqual(Path(environment["source_path"]), Path(source_path).resolve())

    def assert_write_logged(self, *, label, filename, iteration, decision_path,
                          detail=()):
        matches = []
        for log_name in ("workflow.log", "state_transitions.log"):
            text = (self.work / "log" / log_name).read_text(encoding="utf-8")
            lines = [line for line in text.splitlines()
                     if f"[knowledge accumulation] {label} saved; iter={iteration}; " in line]
            self.assertEqual(len(lines), 1, (log_name, label, lines))
            line = lines[0]
            for token in ("framework=Triton", "chip=offline_test", "SoC=offline_test",
                          f"output directory={self.work / 'knowledge'}",
                          f"file={self.work / 'knowledge' / filename}",
                          f"original decision={decision_path}", *detail):
                self.assertIn(token, line)
            matches.append(line)
        self.assertEqual(matches[0], matches[1], "Both audit logs must identify the same write")

    def test_real_improvement_and_regression_record_environment_and_both_logs(self):
        self.routing.perfs = {1: (2.0,), 2: (1.8,), 3: (2.1,)}
        self.routing.run_workflow(max_iterations=3, file_logs=True)

        for iteration, loader, filename, label, delta in (
            (2, load_regression_patterns, "regression_patterns.md", "regression lesson", -10.0),
            (3, load_proven_patterns, "proven_patterns.md", "success experience", 16.7),
        ):
            with self.subTest(label=label):
                records = loader(str(self.work))
                self.assertEqual(len(records), 1)
                record = records[0]
                entry = self.ledger(iteration)
                report = self.work / f"eval/iter{iteration}/perf_result.json"
                self.assert_environment(record["environment"], source_kind="performance_report",
                                        source_path=report)
                self.assertEqual(entry["environment"], record["environment"])
                self.assertEqual(record["delta_pct"], delta)
                self.assertEqual(record["decision_path"], entry["stage9_decision_path"])
                self.assertTrue(Path(record["decision_path"]).is_file())
                self.assert_write_logged(
                    label=label, filename=filename, iteration=iteration,
                    decision_path=record["decision_path"],
                    detail=(f"change={delta}%", "avg_speedup=", str(report)))
                self.assert_write_logged(
                    label="history ledger", filename="history.json", iteration=iteration,
                    decision_path=record["decision_path"])

    def test_model_environment_and_changed_device_file_cannot_relabel_measured_gain(self):
        self.routing.perfs = {1: (1.5,), 2: (1.8,)}
        forged = {"framework": "MODEL_FAKE_FRAMEWORK", "chip_model": "MODEL_FAKE_CHIP",
                  "soc_version": "MODEL_FAKE_SOC", "source_path": "MODEL_FAKE_SOURCE"}

        def stage9(output, payload, iteration, _prompt):
            if iteration == 2:
                # The measured report remains frozen while a live file changes.
                device = self.read_json(self.work / "device_info.json")
                device.update(chip_model="LIVE_CHANGED_CHIP", soc_version="LIVE_CHANGED_SOC")
                self.routing.write_json("device_info.json", device)
                payload["proven_pattern"]["environment"] = deepcopy(forged)
            self.routing.write_json(output, payload)
            return True

        self.routing.stage9_callback = stage9
        self.routing.run_workflow(max_iterations=2, file_logs=True)

        record, = load_proven_patterns(str(self.work))
        entry = self.ledger(2)
        report = self.work / "eval/iter2/perf_result.json"
        self.assert_environment(record["environment"], source_kind="performance_report",
                                source_path=report)
        self.assertEqual(entry["environment"], record["environment"])
        self.assertEqual(self.read_json(record["decision_path"])
                         ["proven_pattern"]["environment"], forged,
                         "Preserve original model output as evidence, not accepted metadata")
        self.assertEqual(self.read_json(self.work / "device_info.json")["chip_model"],
                         "LIVE_CHANGED_CHIP")
        self.assertEqual(self.read_json(report)["comparison_context"]["hardware"]["chip_model"],
                         "offline_test")
        text = (self.work / "knowledge/proven_patterns.md").read_text(encoding="utf-8")
        for invalid in ("MODEL_FAKE", "LIVE_CHANGED"):
            self.assertNotIn(invalid, text)
        self.assert_write_logged(label="success experience", filename="proven_patterns.md",
                                 iteration=2, decision_path=record["decision_path"])

    def test_compile_ruling_records_environment_paths_without_a_performance_report(self):
        self.routing.perfs = {1: (1.6,)}
        self.routing.build_failures = {1}
        question = self.work / "develop/iter0/question.md"
        question.parent.mkdir(parents=True, exist_ok=True)
        question.write_text("# Question\nThe suggested intrinsic is unavailable.\n", encoding="utf-8")

        def stage9(output, payload, _iteration, _prompt):
            payload["pitfall"] = {
                "verdict": "confirmed", "topic": "Unsupported intrinsic",
                "target_advice": "Use an intrinsic unavailable on this chip",
                "feedback": "Compilation rejects the suggested intrinsic",
                "root_cause": "The advice used a different hardware generation",
                "correct_approach": "Use the supported primitive for the detected SoC",
                "environment": {"framework": "MODEL_FAKE", "chip_model": "MODEL_FAKE"},
            }
            self.routing.write_json(output, payload)
            return True

        self.routing.stage9_callback = stage9
        self.routing.run_workflow(max_iterations=1, file_logs=True)

        self.assertNotIn("stage6", self.routing.fixture.events)
        self.assertFalse((self.work / "eval/iter1/perf_result.json").exists())
        record, = load_pitfalls(str(self.work))
        entry = self.ledger(1)
        self.assert_environment(record["environment"], source_kind="workflow_context",
                                source_path=self.work / "device_info.json")
        self.assertEqual(entry["environment"], record["environment"])
        self.assertEqual(record["decision_path"], entry["stage9_decision_path"])
        self.assertEqual(Path(record["question_path"]), question)
        self.assert_write_logged(label="pitfall log", filename="tech_lead_pitfalls.md",
                                 iteration=1, decision_path=record["decision_path"],
                                 detail=("verdict=confirmed", str(question)))
        self.assert_write_logged(label="history ledger", filename="history.json",
                                 iteration=1, decision_path=record["decision_path"])


if __name__ == "__main__":
    unittest.main()
