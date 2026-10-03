"""Regression handling against real routing, knowledge commits and code snapshots."""

import json
from pathlib import Path
import unittest

from lib.history_manager import load_history
import test_fusion_routing as fusion_routing
import test_semantic_routing as semantic_routing


class RegressionRoutingTests(unittest.TestCase):
    def setUp(self):
        # Compose the existing transport fixture, without inheriting its test cases.
        self.routing = semantic_routing.SemanticRoutingTests(methodName="runTest")
        self.routing.setUp()
        self.addCleanup(self.routing.doCleanups)
        self.fixture = self.routing.fixture
        self.work = self.routing.work
        self.routing.config["workflow"]["semantic_exit"]["passed_window"] = 8
        self.original_agent = self.fixture.agent
        self.fixture.agent = self.agent
        self.entries = []
        self.on_entry = None

    def agent(self, agent, role, work, prompt, **kwargs):
        stage = fusion_routing.ROLES[Path(role).name]
        iteration = self.fixture.read_state()["iteration"]
        if stage in {"stage9", "stage3", "stage10"}:
            entry = {
                "stage": stage, "iteration": iteration, "prompt": prompt,
                "code": (self.work / "impl/synthetic_op.py").read_text(encoding="utf-8"),
                "knowledge_exists": (self.work / "knowledge/regression_patterns.md").is_file(),
                "record_exists": self.record_path(iteration).is_file(),
            }
            if entry["record_exists"]:
                record = self.record(iteration)
                entry["record_status"] = record["status"]
                entry["failed_code_exists"] = Path(record["failed_implementation_dir"]).exists()
            self.entries.append(entry)
            if self.on_entry:
                self.on_entry(entry)
        return self.original_agent(agent, role, work, prompt, **kwargs)

    def record_path(self, iteration):
        return self.work / "selection/rollbacks" / f"iter{iteration}" / "record.json"

    def record(self, iteration):
        return json.loads(self.record_path(iteration).read_text(encoding="utf-8"))

    def entry(self, stage, iteration):
        matches = [entry for entry in self.entries
                   if entry["stage"] == stage and entry["iteration"] == iteration]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def assert_code(self, directory, revision):
        self.assertEqual((Path(directory) / "synthetic_op.py").read_text(encoding="utf-8"),
                         f"# implementation revision {revision}\n")

    def test_passed_regression_commits_knowledge_before_restoring_best_for_stage3(self):
        self.routing.perfs = {1: (2.0,), 2: (1.5,)}
        self.routing.run_workflow(max_iterations=2, file_logs=True)

        review = self.entry("stage9", 2)
        develop = self.entry("stage3", 2)
        self.assertEqual(review["code"], "# implementation revision 1\n")
        self.assertEqual(review["record_status"], "planned")
        self.assertFalse(review["failed_code_exists"])
        self.assertEqual(develop["code"], "# implementation revision 0\n")
        self.assertTrue(develop["knowledge_exists"])
        self.assertTrue(develop["record_exists"])
        self.assertIn("性能退步处置", develop["prompt"])

        record = self.record(2)
        self.assertEqual(record["action"], "restore_best")
        self.assertEqual(record["status"], "restored")
        self.assertEqual(record["source_iteration"], 1)
        self.assert_code(record["best_implementation_dir"], 0)
        self.assert_code(record["failed_implementation_dir"], 1)
        rounds = load_history(str(self.work))["rounds"]
        self.assertEqual([row["avg_speedup"] for row in rounds], [2.0, 1.5])
        for filename in ("workflow.log", "state_transitions.log"):
            log = (self.work / "log" / filename).read_text(encoding="utf-8")
            for token in ("===== 性能退步处置：iter2 =====", "avg_speedup 2.0→1.5",
                          "变化=-25.0%", "当前全部达标=True",
                          "===== 最佳实现恢复完成：iter2 → 最佳 iter1 =====",
                          record["best_implementation_dir"], record["failed_implementation_dir"],
                          record["knowledge_path"], record["decision_path"], str(self.record_path(2))):
                with self.subTest(log=filename, token=token):
                    self.assertIn(token, log)

    def test_regression_with_one_slow_case_keeps_current_code_and_records_knowledge(self):
        self.routing.perfs = {1: (2.0, 2.0), 2: (2.0, 0.8)}
        self.routing.run_workflow(max_iterations=2, file_logs=True)

        self.assertEqual(self.entry("stage9", 2)["code"], "# implementation revision 1\n")
        develop = self.entry("stage3", 2)
        self.assertEqual(develop["code"], "# implementation revision 1\n")
        self.assertTrue(develop["knowledge_exists"])
        self.assertIn("性能退步处置", develop["prompt"])
        record = self.record(2)
        self.assertEqual(record["action"], "keep_current")
        self.assertEqual(record["status"], "recorded")
        self.assertEqual(self.routing.selection_status()["best"]["iteration"], 1)
        for filename in ("workflow.log", "state_transitions.log"):
            log = (self.work / "log" / filename).read_text(encoding="utf-8")
            for token in ("===== 性能退步处置：iter2 =====", "avg_speedup 2.0→1.4",
                          "变化=-30.0%", "当前全部达标=False", "保留当前实现，只记录退步教训",
                          record["best_implementation_dir"], record["knowledge_path"],
                          str(self.record_path(2))):
                with self.subTest(log=filename, token=token):
                    self.assertIn(token, log)
            self.assertNotIn("===== 最佳实现恢复完成", log)
        # Both logs identify record.json; its final decision links the knowledge
        # commit even though the branch notification precedes the Stage9 call.
        decision = Path(record["decision_path"])
        self.assertTrue(record["knowledge_written"])
        self.assertTrue(decision.is_file())
        self.assertTrue(json.loads((decision.parent / "commit.json").read_text(encoding="utf-8"))["committed"])
        self.assertIn(str(decision), (self.work / "log/workflow.log").read_text(encoding="utf-8"))

    def test_exact_five_percent_regression_and_case_equal_one_restore_best(self):
        self.routing.perfs = {1: (2.0, 2.0), 2: (2.8, 1.0)}
        self.routing.run_workflow(max_iterations=2)

        self.assertEqual(self.entry("stage3", 2)["code"], "# implementation revision 0\n")
        self.assertEqual(self.record(2)["action"], "restore_best")
        knowledge = (self.work / "knowledge/regression_patterns.md").read_text(encoding="utf-8")
        self.assertIn("-5.0", knowledge)
        self.assertFalse(any(stage in {"stage7", "stage8"}
                             for iteration, stage, _ in self.routing.seen if iteration == 2))

    def test_smaller_regression_does_not_restore_or_create_regression_knowledge(self):
        self.routing.perfs = {1: (2.0,), 2: (1.92,)}
        self.routing.run_workflow(max_iterations=2)

        self.assertEqual(self.entry("stage3", 2)["code"], "# implementation revision 1\n")
        self.assertFalse(self.record_path(2).exists())
        self.assertFalse((self.work / "knowledge/regression_patterns.md").exists())

    def test_semantic_exit_restores_only_after_current_regression_is_recorded(self):
        self.routing.config["workflow"]["semantic_exit"]["passed_window"] = 2
        self.routing.perfs = {1: (2.0,), 2: (2.0,), 3: (1.5,)}
        self.routing.run_workflow(max_iterations=8)

        self.assertEqual(self.entry("stage9", 3)["code"], "# implementation revision 2\n")
        summary = self.entry("stage10", 3)
        self.assertEqual(summary["code"], "# implementation revision 0\n")
        self.assertTrue(summary["knowledge_exists"])
        self.assertEqual(self.fixture.read_state()["stopped_by"], "semantic_stagnation")
        self.assertFalse(any(entry["stage"] == "stage3" and entry["iteration"] == 3
                             for entry in self.entries))
        record = self.record(3)
        self.assertEqual(record["action"], "restore_best")
        self.assert_code(record["failed_implementation_dir"], 2)

    def test_stage3_retry_preserves_partial_changes_and_original_failed_snapshot(self):
        self.routing.perfs = {1: (2.0,), 2: (1.5,)}
        interrupted = False

        def interrupt_development(entry):
            nonlocal interrupted
            if entry["stage"] == "stage3" and entry["iteration"] == 2 and not interrupted:
                self.assertEqual(entry["code"], "# implementation revision 0\n")
                interrupted = True
                (self.work / "impl/synthetic_op.py").write_text(
                    "# partial Stage3 change after rollback\n", encoding="utf-8")
                raise RuntimeError("interrupt development after best rollback")

        self.on_entry = interrupt_development
        with self.assertRaisesRegex(RuntimeError, "interrupt development after best rollback"):
            self.routing.run_workflow(max_iterations=2)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter2_stage3")
        before = self.record_path(2).read_bytes()
        self.routing.run_workflow()

        attempts = [entry for entry in self.entries
                    if entry["stage"] == "stage3" and entry["iteration"] == 2]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[1]["code"], "# partial Stage3 change after rollback\n")
        self.assertEqual(self.record_path(2).read_bytes(), before)
        self.assert_code(self.record(2)["failed_implementation_dir"], 1)
        self.assertEqual(self.fixture.events.count("stage6"), 2)

    def test_failed_stage9_cannot_restore_before_required_knowledge_commits(self):
        self.routing.perfs = {1: (2.0,), 2: (1.5,)}

        def omit_required_knowledge(output, payload, iteration, _prompt):
            if iteration == 2:
                payload.pop("regression_pattern")
            self.routing.write_json(output, payload)
            return True

        self.routing.stage9_callback = omit_required_knowledge
        with self.assertRaises(RuntimeError):
            self.routing.run_workflow(max_iterations=2)

        self.assert_code(self.work / "impl", 1)
        self.assertEqual(self.record(2)["status"], "planned")
        self.assertFalse(Path(self.record(2)["failed_implementation_dir"]).exists())
        self.assertFalse((self.work / "knowledge/regression_patterns.md").exists())
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter2_stage9")

    def test_corrupt_best_snapshot_blocks_restore_without_overwriting_live_code(self):
        self.routing.perfs = {1: (2.0,), 2: (1.5,)}

        def corrupt_snapshot_after_review(output, payload, iteration, _prompt):
            self.routing.write_json(output, payload)
            if iteration == 2:
                best = self.routing.selection_status()["best"]
                (Path(best["implementation_dir"]) / "synthetic_op.py").write_text(
                    "# corrupted historical snapshot\n", encoding="utf-8")
            return True

        self.routing.stage9_callback = corrupt_snapshot_after_review
        with self.assertRaisesRegex(ValueError, "失效|恢复"):
            self.routing.run_workflow(max_iterations=2)

        self.assert_code(self.work / "impl", 1)
        self.assertEqual(self.record(2)["status"], "planned")
        self.assertFalse(Path(self.record(2)["failed_implementation_dir"]).exists())
        self.assertTrue((self.work / "knowledge/regression_patterns.md").is_file())
        self.assertFalse(any(entry["stage"] == "stage3" and entry["iteration"] == 2
                             for entry in self.entries))


if __name__ == "__main__":
    unittest.main()
