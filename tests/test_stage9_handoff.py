"""Offline Stage9 output, history protection and knowledge-consumption regressions."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock
from uuid import uuid4

import orchestrator
from lib.history_manager import (load_history, save_history, load_proven_patterns,
                                 load_regression_patterns)
from lib.state import State
import test_semantic_routing as semantic_routing_tests


class Stage9HandoffTests(unittest.TestCase):
    def setUp(self):
        self.routing = semantic_routing_tests.SemanticRoutingTests(methodName="runTest")
        self.routing.setUp()
        self.addCleanup(self.routing.doCleanups)
        self.fixture = self.routing.fixture
        self.work = self.routing.work
        self.routing.perfs = {1: (1.6,), 2: (1.7,)}
        self.routing.config["workflow"]["semantic_exit"]["passed_window"] = 8

    def write_decision(self, output, payload):
        self.routing.write_json(output, payload)

    def assert_second_review_rejected(self, alter, *, tamper_history=False):
        before = {}

        def agent(output, payload, iteration, _prompt):
            if iteration == 1:
                self.write_decision(output, payload)
                return True
            before.update(deepcopy(load_history(str(self.work))))
            if tamper_history:
                # A model bypassing its output contract must not destroy durable history.
                (self.work / "knowledge/history.json").write_text(
                    '{"ledger": [], "rounds": [], "suggest_next": "UNSAFE_DIRECT_EDIT"}',
                    encoding="utf-8")
            return alter(output, payload)

        self.routing.stage9_callback = agent
        with self.assertRaises(RuntimeError):
            self.routing.run_workflow(max_iterations=2)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter2_stage9")
        self.assertEqual([iteration for iteration, stage, _ in self.routing.seen
                          if stage == "stage3"], [1])
        self.assertNotIn("stage10", self.fixture.events)
        self.assertEqual(load_history(str(self.work)), before)
        self.assertEqual(before["ledger"][0]["direction"], "LOCAL_TILE_ITER_1")
        self.assertTrue(before["suggest_next"], "Stale suggestions exist but must not reach Stage3")
        return before

    def test_missing_output_cannot_reuse_previous_advice(self):
        self.assert_second_review_rejected(lambda _output, _payload: True)

    def test_agent_false_cannot_publish_even_a_valid_output(self):
        def false_response(output, payload):
            self.write_decision(output, payload)
            return False
        self.assert_second_review_rejected(false_response)

    def test_wrong_iteration_is_rejected_before_development(self):
        def stale_iteration(output, payload):
            payload["iteration"] -= 1
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(stale_iteration)

    def test_wrong_request_id_is_rejected_before_development(self):
        def stale_request(output, payload):
            payload["request_id"] = str(uuid4())
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(stale_request)

    def test_malformed_json_is_rejected_before_development(self):
        def malformed(output, _payload):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"iteration": 2,', encoding="utf-8")
            return True
        self.assert_second_review_rejected(malformed)

    def test_agent_exit_control_field_is_rejected(self):
        def attempted_control(output, payload):
            payload["exit_decision"] = True
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(attempted_control)

    def test_nonterminal_empty_suggestions_cannot_reuse_previous_advice(self):
        def empty_advice(output, payload):
            payload["suggest_next"] = []
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(empty_advice)

    def test_missing_required_improvement_reason_is_not_replaced_by_a_placeholder(self):
        def missing_pattern(output, payload):
            payload.pop("proven_pattern")
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(missing_pattern)
        path = self.work / "knowledge/proven_patterns.md"
        self.assertFalse(path.exists(), "An unreviewed improvement must not create placeholder knowledge")

    def test_missing_required_regression_reason_is_not_replaced_by_a_placeholder(self):
        self.routing.perfs = {1: (2.0,), 2: (1.9,)}

        def missing_pattern(output, payload):
            payload.pop("regression_pattern")
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(missing_pattern)
        self.assertFalse((self.work / "knowledge/regression_patterns.md").exists())

    def test_missing_current_case_analysis_cannot_reuse_previous_conclusion(self):
        def missing_analysis(output, payload):
            payload["ledger_entry"].pop("case_analysis")
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(missing_analysis)

    def test_next_plan_cannot_replace_the_current_evaluation_summary(self):
        def missing_summary(output, payload):
            payload["ledger_entry"].pop("evaluation_summary")
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(missing_summary)

    def test_triggered_experience_without_evidence_is_not_published(self):
        def missing_evidence(output, payload):
            payload["proven_pattern"].pop("evidence")
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(missing_evidence)
        self.assertFalse((self.work / "knowledge/proven_patterns.md").exists())

    def test_successful_output_merges_into_original_history_despite_agent_tampering(self):
        before = {}

        def agent(output, payload, iteration, _prompt):
            if iteration == 2:
                before.update(deepcopy(load_history(str(self.work))))
                tampered = deepcopy(before)
                tampered["rounds"] = []
                tampered["ledger"][0]["direction"] = "CORRUPTED_OLD_DIRECTION"
                tampered["ledger"][0]["avg_speedup_after"] = 999
                tampered["ledger"][-1]["verdict"] = "MODEL_INVENTED_VERDICT"
                tampered["insights"] = ["UNAUTHORIZED_HISTORY_WRITE"]
                save_history(str(self.work), tampered)
                payload["suggest_next"][0]["action"] = "VALIDATED_CURRENT_ADVICE"
            self.write_decision(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=2)
        after = load_history(str(self.work))
        self.assertEqual(after["rounds"], before["rounds"])
        self.assertEqual(after["ledger"][0], before["ledger"][0])
        for field in ("iter", "avg_speedup_before", "avg_speedup_after", "delta", "verdict"):
            self.assertEqual(after["ledger"][-1][field], before["ledger"][-1][field])
        self.assertEqual(after["ledger"][-1]["reason"], "perf_pass_semantic_review")
        self.assertEqual(after["ledger"][-1]["direction"], "LOCAL_TILE_ITER_2")
        self.assertEqual(after["insights"], before["insights"])
        self.assertIn("VALIDATED_CURRENT_ADVICE", self.fixture.prompts["stage3"])
        self.assertIn("LOCAL_TILE_ITER_1", self.fixture.prompts["stage3"])
        self.assertIn("LOCAL_TILE_ITER_2", self.fixture.prompts["stage3"])
        self.assertNotIn("CORRUPTED_OLD_DIRECTION", self.fixture.prompts["stage3"])
        self.assertNotIn("MODEL_INVENTED_VERDICT", self.fixture.prompts["stage3"])

    def test_history_is_restored_when_tampering_agent_returns_false(self):
        def failed(output, payload):
            self.write_decision(output, payload)
            return False
        self.assert_second_review_rejected(failed, tamper_history=True)

    def test_history_is_restored_when_tampering_agent_raises(self):
        def failed(_output, _payload):
            raise RuntimeError("synthetic Stage9 transport interruption")
        self.assert_second_review_rejected(failed, tamper_history=True)

    def test_history_is_restored_when_tampering_agent_returns_invalid_output(self):
        def invalid(output, payload):
            payload["iteration"] = 1
            self.write_decision(output, payload)
            return True
        self.assert_second_review_rejected(invalid, tamper_history=True)

    def repeat_current_review(self):
        state = State.load_or_create(str(self.work))
        directories = {name: str(self.work / name / f"iter{state.iteration}")
                       for name in ("eval", "profile", "search", "build", "develop")}
        with self.fixture.patches():
            orchestrator.run_tech_lead(
                Mock(), str(Path(orchestrator.__file__).parent / "roles"), str(self.work),
                "synthetic_op", state, Mock(), {"N4": Mock()}, directories,
                fail_reason="perf_pass_optimize",
                perf_diff=orchestrator._compute_perf_diff(str(self.work), state.iteration))

    def assert_repeated_pattern_is_not_duplicated(self, filename, why):
        requests = []

        def agent(output, payload, _iteration, _prompt):
            requests.append(payload["request_id"])
            self.write_decision(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=2)
        path = self.work / "knowledge" / filename
        original = path.read_text(encoding="utf-8")
        history = load_history(str(self.work))
        loader = load_proven_patterns if filename == "proven_patterns.md" else load_regression_patterns
        original_pattern = next(record for record in loader(str(self.work)) if record["iter"] == 2)
        original_decision = history["ledger"][-1]["stage9_decision_path"]
        self.assertIn(why, original)
        self.assertEqual(original.count("## iter2:"), 1)
        for _ in range(2):
            self.repeat_current_review()
        updated_text = path.read_text(encoding="utf-8")
        self.assertEqual(updated_text.count("## iter2:"), 1)
        self.assertIn(why, updated_text)
        updated_patterns = loader(str(self.work))
        self.assertEqual(len(updated_patterns), 1)
        updated_pattern = updated_patterns[0]
        for field in ("iter", "what_changed", "why_it_worked", "why_it_failed",
                      "speedup_before", "speedup_after", "delta_pct", "case_analysis", "evidence"):
            if field in original_pattern:
                self.assertEqual(updated_pattern[field], original_pattern[field])
        after = load_history(str(self.work))
        self.assertEqual(after["ledger"][:-1], history["ledger"][:-1])
        for field in ("iter", "direction", "modify_files", "readonly_files", "avg_speedup_before",
                      "avg_speedup_after", "delta", "verdict"):
            self.assertEqual(after["ledger"][-1][field], history["ledger"][-1][field])
        self.assertEqual(after["rounds"], history["rounds"])
        newest_decision = Path(after["ledger"][-1]["stage9_decision_path"])
        self.assertNotEqual(str(newest_decision), original_decision)
        self.assertTrue(newest_decision.is_file())
        self.assertEqual(newest_decision.parent.name, requests[-1])
        self.assertEqual(updated_pattern["decision_path"], str(newest_decision))
        self.assertEqual(len(requests), 4)
        self.assertEqual(len(set(requests)), 4, "Every invocation needs a fresh request binding")

    def test_repeated_stage9_consumption_keeps_one_detailed_improvement_record(self):
        self.assert_repeated_pattern_is_not_duplicated("proven_patterns.md", "Less launch overhead")

    def test_repeated_stage9_consumption_keeps_one_detailed_regression_record(self):
        self.routing.perfs = {1: (2.0,), 2: (1.9,)}
        self.assert_repeated_pattern_is_not_duplicated("regression_patterns.md", "More padding overhead")

    def prepare_question(self):
        path = self.work / "develop/iter0/question.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Question\nThe forced larger tile increases tail padding.\n", encoding="utf-8")
        return path

    def test_public_pitfall_marks_the_original_question_and_writes_the_mistake_book_once(self):
        question = self.prepare_question()
        original = question.read_text(encoding="utf-8")
        pitfall = {
            "verdict": "confirmed", "topic": "Unnecessary large-tile requirement",
            "target_advice": "Force one large tile for every input shape",
            "feedback": "The small tail case performs extra padded work",
            "root_cause": "The prior advice ignored measured padding overhead",
            "correct_approach": "Keep the small tile for tail-heavy shapes",
        }

        def agent(output, payload, _iteration, prompt):
            request = json.loads((output.parent / "request.json").read_text(encoding="utf-8"))
            # A later review sees the question as already adjudicated and must
            # not resubmit a now-inapplicable pitfall merely to repeat history.
            if request.get("question_path"):
                payload["pitfall"] = pitfall
            self.write_decision(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=1)
        self.assertIn(str(question), self.fixture.prompts["stage9"])
        text = question.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(original))
        self.assertEqual(text.count("## tech_lead 裁定"), 1)
        self.assertIn(pitfall["root_cause"], text)
        self.assertIn(pitfall["correct_approach"], text)
        book = self.work / "knowledge/tech_lead_pitfalls.md"
        first_book = book.read_text(encoding="utf-8")
        for field in ("topic", "target_advice", "feedback", "root_cause", "correct_approach"):
            self.assertIn(pitfall[field], first_book)
        self.assertEqual(first_book.count("## iter1 "), 1)
        self.repeat_current_review()
        self.assertEqual(book.read_text(encoding="utf-8"), first_book)
        self.assertEqual(question.read_text(encoding="utf-8"), text)

    def test_unanswered_question_cannot_pass_review_without_a_pitfall(self):
        question = self.prepare_question()
        original = question.read_text(encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.routing.run_workflow(max_iterations=1)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter1_stage9")
        self.assertNotIn("stage3", self.fixture.events)
        self.assertNotIn("stage10", self.fixture.events)
        self.assertEqual(question.read_text(encoding="utf-8"), original)
        self.assertFalse((self.work / "knowledge/tech_lead_pitfalls.md").exists())

    def test_program_semantic_exit_allows_empty_advice_without_developing_again(self):
        self.routing.config["workflow"]["semantic_exit"]["passed_window"] = 2
        self.routing.perfs = {1: (1.6,), 2: (1.6,), 3: (1.6,)}

        def agent(output, payload, iteration, _prompt):
            if iteration == 3:
                payload["suggest_next"] = []
            self.write_decision(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=8)
        self.assertEqual(self.fixture.read_state()["stopped_by"], "semantic_stagnation")
        self.assertEqual([iteration for iteration, stage, _ in self.routing.seen
                          if stage == "stage3"], [1, 2])
        self.assertEqual(load_history(str(self.work))["suggest_next"], [])
        self.assertIn("stage10", self.fixture.events)


if __name__ == "__main__":
    unittest.main()
