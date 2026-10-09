"""Offline checks for the shared per-iteration profile report and its consumers."""

import json
from pathlib import Path
import unittest

from lib.history_manager import load_history, save_history
import test_fusion_routing as fusion
import test_semantic_routing as semantic


class ProfileReportRoutingTests(unittest.TestCase):
    def setUp(self):
        self.flow = semantic.SemanticRoutingTests(methodName="runTest")
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)
        self.fixture = self.flow.fixture
        self.work = self.flow.work
        self.flow.perfs = {1: (1.6,), 2: (1.7,)}
        self.flow.config["workflow"]["semantic_exit"]["passed_window"] = 8

    def report(self, iteration):
        return self.work / f"profile/iter{iteration}/bottleneck_analysis.md"

    def assert_report_hint(self, prompt, iteration):
        relative = f"profile/iter{iteration}/bottleneck_analysis.md"
        hints = [line for line in prompt.splitlines()
                 if f"relative to working directory: `{relative}`" in line]
        self.assertTrue(hints, f"Missing report input description: {relative}")
        self.assertTrue(any(str(self.report(iteration)) in line for line in hints))
        for token in ("Purpose:", "how to read:", "profile/<iter>/bottleneck_analysis.md"):
            self.assertTrue(any(token in line for line in hints), token)

    def assert_stage9_report(self, iteration, decision_path=None):
        report = self.report(iteration)
        self.assertTrue(report.is_file(), f"Missing current report: {report}")
        text = report.read_text(encoding="utf-8")
        self.assertIn("Stage9", text)
        if decision_path is None:
            ledger = next(entry for entry in load_history(str(self.work))["ledger"]
                          if entry["iter"] == iteration)
            decision_path = Path(ledger["stage9_decision_path"])
        decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
        self.assertIn(decision["ledger_entry"]["evaluation_summary"], text)
        for analysis in decision["ledger_entry"]["case_analysis"]:
            for field in ("case_id", "observation", "explanation", "evidence", "next_action"):
                self.assertIn(analysis[field], text)
        return text

    def test_underperforming_stage7_report_is_preserved_for_stage8_stage9_and_stage3(self):
        self.flow.perfs = {1: (0.8, 1.6)}
        snapshots = {}
        original_agent = self.fixture.agent

        def agent(agent_name, role, work, prompt, **kwargs):
            stage = fusion.ROLES[Path(role).name]
            if stage in {"stage8", "stage9", "stage3"}:
                snapshots[stage] = self.report(1).read_text(encoding="utf-8")
            return original_agent(agent_name, role, work, prompt, **kwargs)

        self.fixture.agent = agent
        self.flow.run_workflow(max_iterations=1)
        report = self.report(1).read_text(encoding="utf-8")
        self.assertIn("Stage7", report)
        self.assertIn("The slow case is improving with local tile tuning.", report)
        self.assertNotIn("Reviewed the actual implementation", report)
        self.assertEqual(set(snapshots), {"stage8", "stage9", "stage3"})
        for stage, snapshot in snapshots.items():
            with self.subTest(stage=stage):
                self.assertEqual(snapshot, report)
                self.assert_report_hint(self.fixture.prompts[stage], 1)

    def assert_unusable_stage7_report_stops_workflow(self, content):
        self.flow.perfs = {1: (0.8,)}
        original_agent = self.fixture.agent

        def agent(agent_name, role, work, prompt, **kwargs):
            if fusion.ROLES[Path(role).name] == "stage7":
                if content is not None:
                    self.report(1).write_text(content, encoding="utf-8")
                return True
            return original_agent(agent_name, role, work, prompt, **kwargs)

        self.fixture.agent = agent
        with self.assertRaisesRegex(RuntimeError, "Stage7"):
            self.flow.run_workflow(max_iterations=1)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter1_stage7")
        self.assertFalse(any(stage in {"stage8", "stage9", "stage3", "stage10"}
                             for _, stage, _ in self.flow.seen))

    def test_successful_stage7_without_report_stops_before_consumers(self):
        self.assert_unusable_stage7_report_stops_workflow(None)

    def test_successful_stage7_with_empty_report_stops_before_consumers(self):
        self.assert_unusable_stage7_report_stops_workflow(" \n\t")

    def test_all_passed_program_renders_validated_case_analysis_before_stage3(self):
        self.flow.perfs = {1: (1.6, 1.8)}
        snapshots = {}
        original_agent = self.fixture.agent

        def agent(agent_name, role, work, prompt, **kwargs):
            stage = fusion.ROLES[Path(role).name]
            if stage == "stage3":
                snapshots[stage] = self.report(1).read_text(encoding="utf-8")
            return original_agent(agent_name, role, work, prompt, **kwargs)

        def decision(output, payload, iteration, _prompt):
            self.assertFalse(self.report(iteration).exists())
            # The agent writes only its decision; the orchestrator owns the report.
            self.flow.write_json(output, payload)
            return True

        self.fixture.agent = agent
        self.flow.stage9_callback = decision
        self.flow.run_workflow(max_iterations=1)
        self.assertFalse(any(stage in {"stage7", "stage8"} for _, stage, _ in self.flow.seen))
        text = self.assert_stage9_report(1)
        self.assertEqual(snapshots["stage3"], text)
        self.assertIn("synthetic_op_1", text)
        self.assertIn("synthetic_op_2", text)
        self.assert_report_hint(self.fixture.prompts["stage3"], 1)
        self.assertIn("Stage9", self.fixture.prompts["stage3"])
        self.assertEqual(self.fixture.read_state()["stage9_context"]["development_reason"],
                         "perf_pass_optimize")

    def test_semantic_exit_iteration_still_renders_stage9_report(self):
        self.flow.perfs = {1: (1.6,), 2: (1.6,), 3: (1.6,)}
        self.flow.config["workflow"]["semantic_exit"]["passed_window"] = 2
        self.flow.run_workflow(max_iterations=8)
        self.assertEqual(self.fixture.read_state()["stopped_by"], "semantic_stagnation")
        self.assertEqual(self.fixture.read_state()["iteration"], 3)
        self.assertEqual([iteration for iteration, stage, _ in self.flow.seen
                          if stage == "stage3"], [1, 2])
        for iteration in (1, 2, 3):
            with self.subTest(iteration=iteration):
                self.assert_stage9_report(iteration)

    def test_failed_stage9_transport_cannot_publish_an_unaccepted_report(self):
        prior_report = []
        outputs = []

        def decision(output, payload, iteration, _prompt):
            self.flow.write_json(output, payload)
            if iteration == 1:
                return True
            prior_report.append(self.report(1).read_bytes())
            outputs.append(output)
            self.assertFalse(self.report(2).exists())
            return False

        self.flow.stage9_callback = decision
        with self.assertRaises(RuntimeError):
            self.flow.run_workflow(max_iterations=2)
        self.assertEqual(len(outputs), 1)
        self.assertFalse(self.report(2).exists())
        self.assertEqual(self.report(1).read_bytes(), prior_report[0])
        self.assertFalse((outputs[0].parent / "commit.json").exists())
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter2_stage9")
        self.assertEqual([iteration for iteration, stage, _ in self.flow.seen
                          if stage == "stage3"], [1])

    def test_exhausted_corrections_do_not_publish_a_new_report(self):
        outputs = []
        prior_report = []

        def decision(output, payload, iteration, _prompt):
            if iteration == 2:
                outputs.append(output)
                if not prior_report:
                    prior_report.append(self.report(1).read_bytes())
                self.assertFalse(self.report(2).exists())
                payload["ledger_entry"]["readonly_files"] = ["impl/synthetic_op.py"]
                payload["ledger_entry"]["case_analysis"][0]["observation"] = "REJECTED_PROFILE_ANALYSIS"
            self.flow.write_json(output, payload)
            return True

        self.flow.stage9_callback = decision
        with self.assertRaises(RuntimeError):
            self.flow.run_workflow(max_iterations=2)
        self.assertEqual(len(outputs), 3)
        self.assertFalse(self.report(2).exists())
        self.assertEqual(self.report(1).read_bytes(), prior_report[0])
        self.assertTrue(all((output.parent / "validation_error.json").is_file() for output in outputs))
        self.assertTrue(all(not (output.parent / "commit.json").exists() for output in outputs))
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        self.assertEqual([iteration for iteration, stage, _ in self.flow.seen
                          if stage == "stage3"], [1])

    def test_successful_correction_renders_only_the_accepted_decision(self):
        outputs = []

        def decision(output, payload, iteration, _prompt):
            outputs.append(output)
            self.assertFalse(self.report(iteration).exists())
            if len(outputs) == 1:
                payload["ledger_entry"]["readonly_files"] = ["impl/synthetic_op.py"]
                observation = "REJECTED_PROFILE_ANALYSIS"
            else:
                observation = "ACCEPTED_PROFILE_ANALYSIS"
            payload["ledger_entry"]["case_analysis"][0]["observation"] = observation
            self.flow.write_json(output, payload)
            return True

        self.flow.stage9_callback = decision
        self.flow.run_workflow(max_iterations=1)
        self.assertEqual(len(outputs), 2)
        text = self.assert_stage9_report(1, outputs[-1])
        self.assertIn("ACCEPTED_PROFILE_ANALYSIS", text)
        self.assertNotIn("REJECTED_PROFILE_ANALYSIS", text)
        self.assertFalse((outputs[0].parent / "commit.json").exists())
        self.assertTrue((outputs[1].parent / "commit.json").exists())
        self.assertEqual(self.fixture.events.count("stage6"), 1)

    def test_legacy_stage3_handoff_backfills_report_without_repeating_evaluation_or_review(self):
        original_agent = self.fixture.agent

        def interrupted(agent_name, role, work, prompt, **kwargs):
            if fusion.ROLES[Path(role).name] == "stage3":
                raise RuntimeError("synthetic interruption before development")
            return original_agent(agent_name, role, work, prompt, **kwargs)

        self.fixture.agent = interrupted
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption before development"):
            self.flow.run_workflow(max_iterations=1)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter1_stage3")
        before = load_history(str(self.work))
        accepted_decision = Path(before["ledger"][-1]["stage9_decision_path"])
        self.assertTrue((accepted_decision.parent / "commit.json").is_file())
        # Simulate a valid checkpoint produced before unified reports existed.
        self.report(1).unlink(missing_ok=True)
        seen_before = len(self.flow.seen)
        self.fixture.agent = original_agent
        self.flow.run_workflow()
        self.assert_stage9_report(1, accepted_decision)
        self.assert_report_hint(self.fixture.prompts["stage3"], 1)
        self.assertEqual([stage for _, stage, _ in self.flow.seen[seen_before:]],
                         ["stage3", "stage10"])
        self.assertEqual(self.fixture.events.count("stage6"), 1)
        self.assertEqual(self.fixture.events.count("stage9"), 1)
        self.assertEqual(load_history(str(self.work)), before)


class ProfileReportIntegrityTests(unittest.TestCase):
    def make_flow(self):
        flow = semantic.SemanticRoutingTests(methodName="runTest")
        flow.setUp()
        self.addCleanup(flow.doCleanups)
        flow.perfs = {1: (0.8,)}
        return flow

    def test_stage7_rerun_cannot_reuse_an_interrupted_attempts_report(self):
        flow = self.make_flow()
        report = flow.work / "profile/iter1/bottleneck_analysis.md"
        original_agent = flow.fixture.agent
        attempts = []

        def agent(agent_name, role, work, prompt, **kwargs):
            if fusion.ROLES[Path(role).name] == "stage7":
                attempts.append(report.exists())
                if len(attempts) == 1:
                    report.write_text("UNFINISHED_STAGE7_REPORT", encoding="utf-8")
                    raise RuntimeError("synthetic interruption during profiling")
                return True
            return original_agent(agent_name, role, work, prompt, **kwargs)

        flow.fixture.agent = agent
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption during profiling"):
            flow.run_workflow(max_iterations=1)
        self.assertEqual(report.read_text(encoding="utf-8"), "UNFINISHED_STAGE7_REPORT")
        with self.assertRaisesRegex(RuntimeError, "Stage7"):
            flow.run_workflow()
        self.assertEqual(attempts, [False, False])
        self.assertFalse(report.exists())
        self.assertEqual(flow.fixture.read_state()["current_stage"], "iter1_stage7")
        self.assertEqual(flow.fixture.events.count("stage6"), 1)
        self.assertFalse(any(stage in {"stage8", "stage9", "stage3", "stage10"}
                             for _, stage, _ in flow.seen))

    def test_stage8_and_stage9_resume_keep_the_completed_stage7_report(self):
        for interrupted_stage in ("stage8", "stage9"):
            with self.subTest(resume=interrupted_stage):
                flow = self.make_flow()
                report = flow.work / "profile/iter1/bottleneck_analysis.md"
                original_agent = flow.fixture.agent

                def interrupted(agent_name, role, work, prompt, **kwargs):
                    if fusion.ROLES[Path(role).name] == interrupted_stage:
                        raise RuntimeError("synthetic interruption after profiling")
                    return original_agent(agent_name, role, work, prompt, **kwargs)

                flow.fixture.agent = interrupted
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption after profiling"):
                    flow.run_workflow(max_iterations=1)
                before = report.read_bytes()
                self.assertEqual(flow.fixture.read_state()["current_stage"], f"iter1_{interrupted_stage}")
                seen_before = len(flow.seen)
                flow.fixture.agent = original_agent
                flow.run_workflow()
                self.assertEqual(report.read_bytes(), before)
                self.assertNotIn("stage7", [stage for _, stage, _ in flow.seen[seen_before:]])
                self.assertEqual(flow.fixture.events.count("stage6"), 1)

    def test_stage9_cannot_overwrite_stage7_report_even_when_its_decision_fails(self):
        for outcome in ("accepted", "transport_failure", "invalid_decision"):
            with self.subTest(outcome=outcome):
                flow = self.make_flow()
                report = flow.work / "profile/iter1/bottleneck_analysis.md"
                original_reports = []
                outputs = []

                def decision(output, payload, _iteration, _prompt):
                    outputs.append(output)
                    current = report.read_bytes()
                    if not original_reports:
                        original_reports.append(current)
                    self.assertEqual(current, original_reports[0])
                    report.write_text("UNAUTHORIZED_STAGE9_REPLACEMENT", encoding="utf-8")
                    if outcome == "invalid_decision":
                        payload["ledger_entry"]["readonly_files"] = ["impl/synthetic_op.py"]
                    flow.write_json(output, payload)
                    return outcome != "transport_failure"

                flow.stage9_callback = decision
                if outcome == "accepted":
                    flow.run_workflow(max_iterations=1)
                else:
                    with self.assertRaises(RuntimeError):
                        flow.run_workflow(max_iterations=1)
                    self.assertNotIn("stage3", flow.fixture.events)
                    self.assertTrue(all(not (output.parent / "commit.json").exists() for output in outputs))
                self.assertEqual(len(outputs), 3 if outcome == "invalid_decision" else 1)
                self.assertEqual(report.read_bytes(), original_reports[0])
                self.assertIn(b"Stage7", report.read_bytes())
                self.assertNotIn(b"UNAUTHORIZED_STAGE9_REPLACEMENT", report.read_bytes())

    def test_invalid_all_passed_decisions_cannot_leave_their_direct_report_writes(self):
        for has_accepted_report in (False, True):
            with self.subTest(previous_accepted=has_accepted_report):
                flow = self.make_flow()
                flow.perfs = {1: (1.6,)}
                report = flow.work / "profile/iter1/bottleneck_analysis.md"
                accepted_report = None
                if has_accepted_report:
                    original_agent = flow.fixture.agent

                    def interrupted(agent_name, role, work, prompt, **kwargs):
                        if fusion.ROLES[Path(role).name] == "stage3":
                            raise RuntimeError("synthetic interruption before development")
                        return original_agent(agent_name, role, work, prompt, **kwargs)

                    flow.fixture.agent = interrupted
                    with self.assertRaisesRegex(RuntimeError, "before development"):
                        flow.run_workflow(max_iterations=1)
                    accepted_report = report.read_bytes()
                    self.assertIn(b"Stage9", accepted_report)
                    # An old, incomplete scope must be reviewed autonomously
                    # before Stage3 resumes. Preserve the accepted report while
                    # checking whether replacement decisions are admissible.
                    history = load_history(str(flow.work))
                    history["ledger"][-1].pop("action_plan")
                    save_history(str(flow.work), history)
                    flow.fixture.agent = original_agent
                outputs = []

                def decision(output, payload, _iteration, _prompt):
                    outputs.append(output)
                    if has_accepted_report:
                        self.assertEqual(report.read_bytes(), accepted_report)
                    else:
                        self.assertFalse(report.exists())
                    report.write_text("UNACCEPTED_DIRECT_STAGE9_REPORT", encoding="utf-8")
                    payload["ledger_entry"]["readonly_files"] = ["impl/synthetic_op.py"]
                    flow.write_json(output, payload)
                    return True

                flow.stage9_callback = decision
                with self.assertRaises(RuntimeError):
                    flow.run_workflow(**({} if has_accepted_report else {"max_iterations": 1}))
                self.assertEqual(len(outputs), 3)
                self.assertTrue(all((output.parent / "validation_error.json").exists() for output in outputs))
                self.assertTrue(all(not (output.parent / "commit.json").exists() for output in outputs))
                self.assertNotIn("stage3", flow.fixture.events)
                self.assertEqual(flow.fixture.events.count("stage6"), 1)
                self.assertFalse((flow.work / "human_review").exists())
                if has_accepted_report:
                    self.assertEqual(report.read_bytes(), accepted_report)
                    self.assertNotIn(b"UNACCEPTED_DIRECT_STAGE9_REPORT", report.read_bytes())
                else:
                    self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
