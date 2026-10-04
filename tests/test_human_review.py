"""Offline human inbox, exact deadline/resume, evidence and trigger tests."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lib.human_review import HumanReview, is_wait_message, submit_message
from tools.human_review import main


class FakeClock:
    def __init__(self, value=1000.0):
        self.value = value
        self.on_sleep = None

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds
        if self.on_sleep:
            self.on_sleep()


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.clock = FakeClock()
        self.runtime = HumanReview(str(self.work), clock=self.clock.now, sleeper=self.clock.sleep)
        self.runtime.configure()

    def send(self, text, **kwargs):
        return submit_message(str(self.work), text, now=self.clock.now(), **kwargs)

    def request(self, start=True, evidence=None):
        request = self.runtime.create_consultation(6, "stagnation", "perf_optimize",
                                                   evidence or [], {"avg_speedup": 2.3})
        Path(request["question_path"]).write_text(
            "# Question\nslow case3 is not at target yet.\nA adjust tiling; B partial fusion.\nRecommended A: smaller change.", encoding="utf-8")
        if start:
            request = self.runtime.start_wait(request["request_id"])
        return request

    def test_unique_messages_preserve_raw_text_without_rewriting_state(self):
        state_path = self.work / "human_review" / "state.json"
        before = state_path.read_bytes()
        first = self.send("  keep the scheme\noptimize case3.  ")
        second = self.send("can the memory layout be explained first?", kind="question")
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(state_path.read_bytes(), before)
        messages = {item["id"]: item for item in self.runtime.pending_messages()}
        self.assertEqual(messages[first["id"]]["text"], "  keep the scheme\noptimize case3.  ")
        self.assertEqual(messages[second["id"]]["kind"], "question")

    def test_processing_and_execution_are_distinct_and_idempotent(self):
        message = self.send("adjust the tiling")
        self.runtime.mark_processed([message["id"]], "decision.json")
        self.assertEqual(self.runtime.pending_messages(), [])
        self.assertEqual(self.runtime.all_messages()[0]["status"], "processed")
        self.runtime.mark_executed([message["id"]], 7, "fusion_scheme_rationale.md")
        self.runtime.mark_processed([message["id"]], "another-decision.json")
        record = self.runtime.all_messages()[0]
        self.assertEqual(record["status"], "executed")
        self.assertEqual(record["executed_iteration"], 7)
        self.assertEqual(record["decision_path"], "decision.json")

    def test_unexecuted_opinion_is_preserved_with_reason(self):
        message = self.send("run one more experiment")
        self.runtime.mark_unexecuted([message["id"]], "iteration limit reached")
        self.assertEqual(self.runtime.pending_messages(), [])
        self.assertEqual(self.runtime.all_messages()[0]["reason"], "iteration limit reached")
        with self.assertRaises(ValueError):
            self.runtime.mark_processed(["missing"], "decision.json")

    def test_wait_detection_does_not_swallow_mixed_directions(self):
        for text in ("请等待", " 等待一会儿。", "please wait!", "稍等"):
            self.assertTrue(is_wait_message(text))
        self.assertFalse(is_wait_message("请等待，并且保持当前方案"))
        self.assertFalse(is_wait_message("A"))

    def test_third_distinct_valid_stagnation_triggers(self):
        for iteration, expected in ((4, False), (5, False), (6, True)):
            result = self.runtime.record_evaluation(iteration, valid=True, all_passed=False,
                                                    review_fusion=True, window_gain=0.01)
            self.assertEqual(result["consultation_due"], expected)
        repeat = self.runtime.record_evaluation(6, valid=True, all_passed=False, review_fusion=True)
        self.assertEqual(repeat["count"], 3)
        self.assertEqual(repeat["trigger_iterations"], [4, 5, 6])
        restored = HumanReview(str(self.work))
        self.assertEqual(restored.state()["stagnation"]["count"], 3)

    def test_failed_eval_does_not_count_and_retry_can_count(self):
        self.runtime.record_evaluation(4, valid=False, all_passed=False, review_fusion=True)
        result = self.runtime.record_evaluation(4, valid=True, all_passed=False, review_fusion=True)
        self.assertEqual(result["count"], 1)

    def test_recovery_and_all_passed_reset_counter(self):
        self.runtime.record_evaluation(4, valid=True, all_passed=False, review_fusion=True)
        result = self.runtime.record_evaluation(5, valid=True, all_passed=False,
                                                review_fusion=False, window_gain=0.05)
        self.assertEqual(result["count"], 0)
        self.runtime.record_evaluation(6, valid=True, all_passed=False, review_fusion=True)
        result = self.runtime.record_evaluation(7, valid=True, all_passed=True, review_fusion=False)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["seen_iterations"], [4, 5, 6, 7])

    def test_warming_window_does_not_count_or_falsely_reset(self):
        self.runtime.record_evaluation(4, valid=True, all_passed=False, review_fusion=True)
        result = self.runtime.record_evaluation(5, valid=True, all_passed=False, review_fusion=False)
        self.assertEqual(result["count"], 1)

    def test_evidence_is_copied_missing_explicit_and_scene_restored(self):
        source = self.work / "impl"
        source.mkdir()
        (source / "kernel.py").write_text("original", encoding="utf-8")
        request = self.request(start=False, evidence=[
            {"path": "impl", "purpose": "operator implementation", "read_hint": "check the slow-shape branch"},
            {"path": "missing.json", "purpose": "not produced yet", "read_hint": "do not infer scores when missing"}])
        manifest = json.loads(Path(request["evidence_manifest_path"]).read_text(encoding="utf-8"))
        snapshot = Path(manifest["files"][0]["snapshot_path"])
        (source / "kernel.py").write_text("new version", encoding="utf-8")
        self.assertEqual((snapshot / "kernel.py").read_text(), "original")
        self.assertEqual(manifest["files"][1]["status"], "missing")
        restored = HumanReview(str(self.work))
        self.assertEqual(restored.active_consultation()["scene"], "stagnation")
        self.assertEqual(restored.active_consultation()["fail_reason"], "perf_optimize")
        reused = restored.create_consultation(6, "wrong scene", "wrong reason", [], {})
        self.assertEqual(reused["request_id"], request["request_id"])

    def test_recursive_evidence_root_rejected(self):
        with self.assertRaises(ValueError):
            self.runtime.create_consultation(6, "stagnation", "perf_optimize",
                                             [{"path": str(self.work)}], {})

    def test_proactive_bundle_preserves_context_and_code_without_waiting(self):
        code = self.work / "kernel.py"
        code.write_text("version 1", encoding="utf-8")
        message = self.send("only change case3")
        bundle_path = self.runtime.create_proactive_bundle(
            6, "compile_failed", "compiler error", [{"path": str(code), "purpose": "this round's implementation"}],
            {"failure_log": "compile.txt"}, [message])
        code.write_text("version 2", encoding="utf-8")
        bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
        self.assertEqual(bundle["request"]["scene"], "compile_failed")
        self.assertEqual(bundle["messages"][0]["text"], "only change case3")
        self.assertEqual(Path(bundle["evidence"][0]["snapshot_path"]).read_text(), "version 1")
        self.assertIsNone(self.runtime.active_consultation())

    def test_wait_begins_only_after_notification_and_resume_keeps_deadline(self):
        request = self.request(start=False)
        self.clock.sleep(20)
        request = self.runtime.start_wait(request["request_id"])
        self.assertEqual(request["deadline"], 1140)
        self.clock.sleep(70)
        restored = HumanReview(str(self.work), clock=self.clock.now, sleeper=self.clock.sleep)
        self.assertEqual(restored.start_wait(request["request_id"])["deadline"], 1140)
        result = restored.wait_consultation(request["request_id"], poll_interval=30)
        self.assertEqual(self.clock.now(), 1140)
        self.assertEqual(result["status"], "timed_out")

    def test_direct_reply_ends_wait_and_is_pending_for_stage9(self):
        request = self.request()
        self.clock.sleep(13)
        message = self.send("choose A; keep the current fusion scheme")
        self.assertEqual(self.runtime.pending_messages(), [])
        result = self.runtime.poll_consultation(request["request_id"])
        self.assertEqual(result["status"], "replied")
        self.assertEqual(result["response_message_ids"], [message["id"]])
        self.assertEqual(self.runtime.pending_messages()[0]["id"], message["id"])

    def test_wait_extends_once_from_original_deadline_and_no_human_consent(self):
        request = self.request()
        self.clock.sleep(119)
        self.send("请等待")
        result = self.runtime.poll_consultation(request["request_id"])
        self.assertEqual(result["deadline"], 1720)
        self.clock.sleep(100)
        self.send("等待一会儿")
        self.assertEqual(self.runtime.poll_consultation(request["request_id"])["deadline"], 1720)
        result = self.runtime.wait_consultation(request["request_id"], poll_interval=60)
        self.assertEqual(self.clock.now(), 1720)
        self.assertEqual(result["status"], "timed_out")
        bundle = json.loads(Path(self.runtime.build_feedback_bundle(request["request_id"])).read_text(encoding="utf-8"))
        self.assertFalse(bundle["human_response_received"])
        self.assertEqual(len(bundle["messages"]), 2)
        self.assertTrue(all(message["kind"] == "wait" for message in bundle["messages"]))

    def test_reply_during_extension_exits_early_and_all_messages_retained(self):
        request = self.request()
        wait = self.send("请等待")
        self.runtime.poll_consultation(request["request_id"])
        self.clock.sleep(200)
        reply = self.send("B")
        result = self.runtime.wait_consultation(request["request_id"])
        self.assertEqual(result["status"], "replied")
        self.assertEqual(self.clock.now(), 1200)
        bundle = json.loads(Path(self.runtime.build_feedback_bundle(request["request_id"])).read_text(encoding="utf-8"))
        self.assertEqual({item["id"] for item in bundle["messages"]}, {wait["id"], reply["id"]})
        self.assertIn("Recommended A", bundle["question_text"])
        self.assertEqual(bundle["context"]["avg_speedup"], 2.3)

    def test_delayed_poll_honors_on_time_wait_and_reply(self):
        request = self.request()
        self.clock.sleep(110)
        self.send("请等待")
        self.clock.sleep(400)
        reply = self.send("A")
        self.clock.sleep(300)
        result = self.runtime.poll_consultation(request["request_id"])
        self.assertEqual(result["status"], "replied")
        self.assertEqual(result["response_message_ids"], [reply["id"]])

    def test_late_wait_does_not_extend_and_late_reply_is_proactive(self):
        request = self.request()
        self.clock.sleep(121)
        wait = self.send("请等待")
        reply = self.send("switch to B")
        result = self.runtime.poll_consultation(request["request_id"])
        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(result["deadline"], 1120)
        self.assertEqual(result["message_ids"], [])
        pending = self.runtime.pending_messages()
        self.assertEqual({item["id"] for item in pending}, {wait["id"], reply["id"]})
        self.assertTrue(all(item["late_reply"] for item in pending))

    def test_complete_resets_counter_without_recounting_same_eval(self):
        for iteration in (4, 5, 6):
            self.runtime.record_evaluation(iteration, valid=True, all_passed=False, review_fusion=True)
        request = self.request()
        message = self.send("A")
        self.runtime.poll_consultation(request["request_id"])
        self.runtime.complete_consultation(request["request_id"], "decision.json")
        self.assertIsNone(self.runtime.active_consultation())
        self.assertEqual(self.runtime.pending_messages(), [])
        self.assertEqual(self.runtime.all_messages()[0]["id"], message["id"])
        self.assertEqual(self.runtime.all_messages()[0]["status"], "processed")
        result = self.runtime.record_evaluation(6, valid=True, all_passed=False, review_fusion=True)
        self.assertEqual(result["count"], 0)
        self.runtime.record_evaluation(7, valid=True, all_passed=False, review_fusion=True)
        self.runtime.complete_consultation(request["request_id"], "decision.json")
        self.assertEqual(self.runtime.state()["stagnation"]["count"], 1)

    def test_recover_interruption_between_request_complete_and_state_update(self):
        request = self.request()
        self.send("A")
        request = self.runtime.poll_consultation(request["request_id"])
        request["status"] = "completed"
        request["decision_path"] = "decision.json"
        self.runtime._save_request(request)
        self.runtime.complete_consultation(request["request_id"], "decision.json")
        self.assertIsNone(self.runtime.active_consultation())
        self.assertEqual(self.runtime.pending_messages(), [])

    def test_fake_clock_receives_reply_while_waiting(self):
        request = self.request()
        sent = []
        def send_later():
            if self.clock.now() >= 1005 and not sent:
                sent.append(self.send("A"))
        self.clock.on_sleep = send_later
        result = self.runtime.wait_consultation(request["request_id"])
        self.assertEqual(self.clock.now(), 1005)
        self.assertEqual(result["status"], "replied")

    def test_disabled_proactive_allows_active_consultation_reply_but_closed_rejects(self):
        self.runtime.configure(active_enabled=False)
        with self.assertRaises(ValueError):
            self.send("proactive feedback")
        self.request()
        self.assertEqual(self.send("A")["kind"], "direction")
        self.runtime.close_workflow("done")
        with self.assertRaises(ValueError):
            self.send("A")

    def test_invalid_input_is_not_written(self):
        for kwargs in ({"text": " "}, {"text": "A", "request_id": "not-found"},
                       {"text": "A", "kind": "unknown"}):
            with self.assertRaises(ValueError):
                submit_message(str(self.work), **kwargs)
        self.assertEqual(self.runtime.all_messages(), [])

    def test_submit_racing_final_close_does_not_report_accepted(self):
        from lib.human_review import _write
        def close_after_publish(path, value):
            _write(path, value)
            if path.parent.name == "inbox":
                self.runtime.close_workflow("completed during submit")
        with patch("lib.human_review._write", side_effect=close_after_publish):
            with self.assertRaisesRegex(ValueError, "archived but will not be executed"):
                self.send("the last piece of feedback")
        self.assertEqual(len(self.runtime.all_messages()), 1)

    def test_cli_text_file_question_status_and_closed_error(self):
        message_file = self.work / "reply.txt"
        message_file.write_text("explain first why case3 is slow?", encoding="utf-8")
        with redirect_stdout(io.StringIO()) as stdout:
            result = main(["--work-dir", str(self.work), "--file", str(message_file), "--kind", "question"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["kind"], "question")
        with redirect_stdout(io.StringIO()) as stdout:
            result = main(["--work-dir", str(self.work), "--status"])
        self.assertEqual(result, 0)
        self.assertEqual(len(json.loads(stdout.getvalue())["pending"]), 1)
        with redirect_stdout(io.StringIO()) as stdout:
            result = main(["--work-dir", str(self.work), "--message", "keep the scheme"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["text"], "keep the scheme")
        self.runtime.close_workflow("done")
        with redirect_stderr(io.StringIO()) as stderr:
            result = main(["--work-dir", str(self.work), "--text", "A"])
        self.assertEqual(result, 2)
        self.assertIn("has finished", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
