"""Exercise the real workflow with a synthetic agent and deterministic human clock."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import orchestrator
from lib.human_review import HumanReview, submit_message
from lib.history_manager import load_history
from lib.tech_lead import validate_human_responses
import test_semantic_routing as routing
import test_fusion_routing as fusion


class HumanRoutingTests(unittest.TestCase):
    def setUp(self):
        self.routing = routing.SemanticRoutingTests(methodName="runTest")
        self.routing.setUp()
        self.addCleanup(self.routing.doCleanups)
        self.work = self.routing.work
        self.fixture = self.routing.fixture
        self.now = 1000.0
        self.sleep_hook = None
        self.human = HumanReview(str(self.work), clock=lambda: self.now, sleeper=self.sleep)
        self.routing.config["workflow"]["human_review"] = {
            "proactive_enabled": True, "consultation_enabled": False}
        self.routing.config["workflow"]["semantic_exit"]["passed_window"] = 8
        self.routing.perfs = {i: (1.6,) for i in range(1, 9)}
        self.original_agent = self.fixture.agent
        self.fixture.agent = self.agent
        self.routing.stage9_callback = self.decision
        self.on_stage = None
        self.on_decision = None
        self.omit_receipt = False
        self.questions = []
        self.question_prompts = []
        self.decisions = []
        self.role_texts = []

    def sleep(self, seconds):
        self.now += seconds
        if self.sleep_hook:
            self.sleep_hook()

    def submit(self, text, **kwargs):
        return submit_message(str(self.work), text, now=self.now, **kwargs)

    def agent(self, agent, role, work, prompt, **kwargs):
        stage = fusion.ROLES[Path(role).name]
        iteration = self.fixture.read_state()["iteration"]
        if self.on_stage:
            self.on_stage(stage, iteration, prompt)
        if stage == "stage9":
            self.role_texts.append(Path(role).read_text(encoding="utf-8"))
        if "Stage9 consultation output file:" in prompt:
            question_path = Path(next(line.split(":", 1)[1].strip() for line in prompt.splitlines()
                                      if line.startswith("Stage9 consultation output file:")))
            request_id = next(line.split(":", 1)[1].strip() for line in prompt.splitlines()
                              if line.startswith("Stage9 consultation request ID:"))
            question = {
                "request_id": request_id, "question": "keep optimizing the slow cases or switch schemes?",
                "difficulty": "one case is still slow", "current_scheme": "F2 vertical fusion",
                "attempts": "tile adjusted; average gain is limited", "evidence": "see the per-case performance and scheme rationale in the version manifest",
                "options": [{"id": name, "title": title, "benefit": "improve the slow cases", "cost": "one implementation round to verify", "risk": "may bring no gain"}
                            for name, title in (("A", "local tuning"), ("B", "shape partial fusion"))],
                "recommended_option": "A", "recommendation_reason": "the slow cases are still improving",
            }
            self.fixture.write_json(question_path, question)
            self.questions.append((iteration, request_id, len(load_history(work).get("ledger", []))))
            self.question_prompts.append(prompt)
            return True
        result = self.original_agent(agent, role, work, prompt, **kwargs)
        if stage == "stage3" and not self.omit_receipt:
            ids = self.fixture.read_state()["stage9_context"].get("message_ids", [])
            messages = [item for item in self.human.all_messages()
                        if item["id"] in ids and item["kind"] == "direction" and item["status"] != "executed"]
            if messages:
                self.fixture.write_json(f"develop/iter{iteration}/human_feedback.json", [
                    {"message_id": item["id"], "status": "implemented", "details": "adjusted the slow cases' tile per the feedback"}
                    for item in messages])
                path = self.work / f"develop/iter{iteration}/fusion_scheme_rationale.md"
                path.write_text(path.read_text(encoding="utf-8") + "\n" + "\n".join(item["id"] for item in messages), encoding="utf-8")
        return result

    def decision(self, output, payload, iteration, prompt):
        request = json.loads((output.parent / "request.json").read_text(encoding="utf-8"))
        ids = set(request["human_message_ids"])
        messages = [item for item in self.human.all_messages() if item["id"] in ids]
        payload["human_responses"] = []
        for message in messages:
            payload["human_responses"].append({"message_id": message["id"], "kind": message["kind"], "answer": "handled with the evidence"})
            if message["kind"] == "direction":
                task = fusion.stage9_plan_task(request.get("known_case_ids"),
                                               task_id=f"T{len(payload['suggest_next']) + 1}")
                task.update(priority="P0", source="human", human_message_id=message["id"],
                            action="P0 (human suggestion): optimize the slow cases", reason=message["text"])
                payload["suggest_next"].append(task)
        self.decisions.append((iteration, ids, request, prompt))
        if self.on_decision:
            self.on_decision(output, payload, iteration, prompt)
        self.fixture.write_json(output, payload)
        return True

    def run_flow(self, max_iterations=None):
        with patch.object(orchestrator, "HumanReview", return_value=self.human):
            self.routing.run_workflow(max_iterations=max_iterations)

    def test_proactive_direction_during_build_reaches_stage3_as_p0(self):
        sent = []
        def send(stage, _iteration, _prompt):
            if stage == "stage4" and not sent:
                sent.append(self.submit("keep the fusion direction first; optimize the slow cases"))
        self.on_stage = send
        self.run_flow(1)
        message = self.human.all_messages()[0]
        self.assertEqual(message["status"], "executed")
        self.assertIn("P0 (human suggestion)", self.fixture.prompts["stage3"])
        self.assertIn(sent[0]["id"], self.fixture.prompts["stage3"])
        self.assertEqual(len(self.decisions), 1)
        manifest = list((self.work / "human_review").glob("iter*/*/evidence_manifest.json"))[0]
        self.assertTrue(json.loads(manifest.read_text(encoding="utf-8"))["files"])

    def test_new_opinion_during_stage9_reopens_review_and_keeps_both_p0s(self):
        first = self.submit("keep the fusion scheme")
        second = []
        def arriving(*_args):
            if not second:
                second.append(self.submit("prioritize case2"))
        self.on_decision = arriving
        self.run_flow(1)
        self.assertEqual(len(self.decisions), 2)
        self.assertEqual(self.decisions[-1][1], {first["id"], second[0]["id"]})
        self.assertTrue(all(item["status"] == "executed" for item in self.human.all_messages()))
        self.assertEqual(len(load_history(str(self.work))["ledger"]), 1)

    def test_question_is_answered_without_becoming_p0(self):
        message = self.submit("why not switch fusion schemes?", kind="question")
        self.run_flow(1)
        suggestions = load_history(str(self.work))["suggest_next"]
        self.assertFalse(any(item.get("human_message_id") == message["id"] for item in suggestions))
        self.assertEqual(self.human.all_messages()[0]["status"], "processed")

    def test_third_stagnation_times_out_without_human_approval(self):
        self.routing.config["workflow"]["human_review"]["consultation_enabled"] = True
        self.routing.config["workflow"]["semantic_exit"]["underperforming_window"] = 1
        self.routing.perfs = {i: (0.8, 1.5) for i in range(1, 5)}
        self.run_flow(4)
        self.assertEqual([item[0] for item in self.questions], [4])
        self.assertEqual(self.now, 1120.0)
        self.assertEqual(self.human.state()["stagnation"]["count"], 0)
        request = self.human.get_consultation(self.questions[0][1])
        self.assertEqual(request["trigger_iterations"], [2, 3, 4])
        self.assertEqual(request["status"], "completed")
        bundle = json.loads((Path(request["directory"]) / "feedback.json").read_text(encoding="utf-8"))
        self.assertFalse(bundle["human_response_received"])
        self.assertFalse(any(s.get("source") == "human" for s in load_history(str(self.work))["suggest_next"]))
        self.assertEqual(len(load_history(str(self.work))["rounds"]), 4)

    def test_consultation_keeps_full_evidence_but_not_final_experience_task(self):
        self.routing.config["workflow"]["human_review"]["consultation_enabled"] = True
        self.routing.config["workflow"]["semantic_exit"]["underperforming_window"] = 1
        self.routing.perfs = {1: (0.8,), 2: (0.8,), 3: (0.7,), 4: (0.8,)}
        self.run_flow(4)
        prompt = self.question_prompts[0]
        self.assertIn("FUSION_HANDOFF[stage9]", prompt)
        self.assertIn("fusion_scheme_rationale.md", prompt)
        self.assertIn("Performance change computed by the program (consultation reference only)", prompt)
        self.assertNotIn("Stage9 output file:", prompt)
        self.assertNotIn("proven_pattern", prompt)
        self.assertTrue((self.work / "knowledge/proven_patterns.md").is_file())
        self.assertTrue((self.work / "knowledge/regression_patterns.md").is_file())

    def test_opinion_during_development_is_handled_with_next_iteration(self):
        sent = []
        def arriving(stage, iteration, _prompt):
            if stage == "stage3" and iteration == 1 and not sent:
                sent.append(self.submit("adjust the tiling first next round"))
        self.on_stage = arriving
        self.run_flow(2)
        self.assertEqual(self.decisions[0][1], set())
        self.assertEqual(self.decisions[1][1], {sent[0]["id"]})
        self.assertEqual(self.human.all_messages()[0]["executed_iteration"], 2)

    def test_consultation_wait_then_answer_and_restore_original_deadline(self):
        self.routing.config["workflow"]["human_review"]["consultation_enabled"] = True
        self.routing.config["workflow"]["semantic_exit"]["underperforming_window"] = 1
        self.routing.perfs = {i: (0.8,) for i in range(1, 5)}
        def interrupt():
            self.submit("请等待")
            self.sleep_hook = None
            raise RuntimeError("test interruption while waiting")
        self.sleep_hook = interrupt
        with self.assertRaisesRegex(RuntimeError, "test interruption"):
            self.run_flow(4)
        original = self.human.active_consultation()
        self.assertEqual(original["original_deadline"], 1120)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter4_stage9")
        self.now = 1140
        self.submit("choose A; keep optimizing the slow cases")
        self.run_flow()
        request = self.human.get_consultation(original["request_id"])
        self.assertEqual(request["deadline"], 1720)
        self.assertEqual(len(self.questions), 1)
        self.assertEqual(self.fixture.read_state()["iteration"], 4)
        self.assertEqual(len(self.human.state()["stagnation"]["seen_iterations"]), 4)
        self.assertEqual([m["status"] for m in self.human.all_messages() if m["kind"] == "direction"], ["executed"])

    def test_semantic_exit_handles_human_but_does_not_start_extra_development(self):
        self.routing.config["workflow"]["semantic_exit"]["passed_window"] = 1
        sent = []
        def arriving(_output, _payload, iteration, _prompt):
            if iteration == 2 and not sent:
                sent.append(self.submit("switch to another fusion method"))
        self.on_decision = arriving
        self.run_flow(8)
        self.assertEqual(self.fixture.read_state()["stopped_by"], "semantic_stagnation")
        self.assertEqual(self.human.all_messages()[0]["status"], "not_executed")
        self.assertEqual(self.human.all_messages()[0]["reason"], "semantic_stagnation")
        self.assertEqual([iteration for iteration, stage, _ in self.routing.seen if stage == "stage3"], [1])
        self.assertIn("Human feedback processing and execution status", self.fixture.prompts["stage10"])

    def test_missing_execution_receipt_stops_and_recovers_same_last_round(self):
        self.submit("adjust the slow cases")
        self.omit_receipt = True
        with self.assertRaisesRegex(RuntimeError, "receipt"):
            self.run_flow(1)
        self.assertEqual(self.fixture.read_state()["stage9_context"]["phase"], "developing")
        self.omit_receipt = False
        self.run_flow()
        self.assertEqual(self.fixture.read_state()["iteration"], 1)
        self.assertEqual(self.human.all_messages()[0]["status"], "executed")
        self.assertEqual(self.fixture.events.count("stage6"), 1)

    def test_compile_failure_resume_keeps_scene_and_does_not_request_performance(self):
        self.routing.build_failures = {1}
        self.routing.interrupt_stage9_iteration = 1
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.run_flow(1)
        saved = self.fixture.read_state()
        self.assertEqual(saved["stage9_context"]["fail_reason"], "build_fail")
        # Simulate the pre-change checkpoint: original scene is recoverable from request.json.
        saved.pop("stage9_context")
        self.fixture.write_json(".state.json", saved)
        self.routing.interrupt_stage9_iteration = None
        self.run_flow()
        self.assertEqual(self.decisions[-1][2]["scene"], "compile")
        self.assertNotIn("This round's formal performance", self.decisions[-1][3])
        self.assertNotIn("## Conditional Task: Record This Round's Performance Experience in Detail", self.role_texts[-1])
        self.assertIn("build_fail", self.fixture.prompts["stage3"])
        self.assertNotIn("stage6", self.fixture.events)

    def test_precision_failure_resume_does_not_run_profiling(self):
        self.routing.precision_failures = {1}
        self.routing.interrupt_stage9_iteration = 1
        with self.assertRaises(RuntimeError):
            self.run_flow(1)
        self.routing.interrupt_stage9_iteration = None
        self.run_flow()
        self.assertEqual(self.decisions[-1][2]["scene"], "precision")
        self.assertNotIn("stage7", self.fixture.events)
        self.assertIn("precision_fail", self.fixture.prompts["stage3"])

    def test_final_report_arrival_is_reviewed_and_report_regenerated(self):
        sent = []
        def arriving(stage, _iteration, _prompt):
            if stage == "stage10" and not sent:
                sent.append(self.submit("consider partial fusion next round"))
        self.on_stage = arriving
        self.run_flow(1)
        self.assertEqual(self.fixture.events.count("stage10"), 2)
        self.assertEqual(self.human.all_messages()[0]["status"], "not_executed")
        self.assertIn("max_iterations", self.human.all_messages()[0]["reason"])
        self.assertTrue(self.human.state()["workflow_closed"])


class HumanProtocolTests(unittest.TestCase):
    def test_model_cannot_relabel_direction_as_question_or_lower_priority(self):
        message = {"id": "m1", "kind": "direction", "text": "keep the current scheme"}
        for kind, priority in (("question", "P0"), ("direction", "P1")):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_human_responses({
                    "human_responses": [{"message_id": "m1", "kind": kind, "answer": "handled"}],
                    "suggest_next": [{"priority": priority, "source": "human", "human_message_id": "m1"}],
                }, [message])

    def test_timeout_cannot_invent_human_p0(self):
        with self.assertRaises(ValueError):
            validate_human_responses({"suggest_next": [{"priority": "P0", "source": "human", "human_message_id": "fake"}]}, [])

    def test_conflict_requires_alternative_and_still_p0(self):
        message = {"id": "m1", "kind": "direction"}
        output = {"human_responses": [{"message_id": "m1", "kind": "conflict", "answer": "hardware not supported"}],
                  "suggest_next": [{"priority": "P0", "source": "human", "human_message_id": "m1"}]}
        with self.assertRaises(ValueError):
            validate_human_responses(output, [message])
        output["human_responses"][0]["alternative"] = "use a tiled loop to satisfy the memory limit"
        validate_human_responses(output, [message])
