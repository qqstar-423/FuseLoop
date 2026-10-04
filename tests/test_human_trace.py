"""Human lifecycle tracing through the real workflow FileHandler, offline."""
from contextlib import redirect_stdout
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest

from lib.human_review import HumanReview, submit_message
from lib.logger import setup_logger
from tools.human_review import main


class HumanTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        self.clock = 1000.0
        self.log = logging.getLogger("triton-ascend-workflow")
        self.previous = (list(self.log.handlers), self.log.level, self.log.propagate, self.log.disabled)
        for handler in self.log.handlers[:]:
            self.log.removeHandler(handler)
        self.log.disabled = False
        self.log.propagate = False
        setup_logger(str(self.work), console_level="CRITICAL")
        self.addCleanup(self.restore_logger)
        self.runtime = HumanReview(str(self.work), clock=lambda: self.clock)
        self.runtime.configure()

    def restore_logger(self):
        for handler in self.log.handlers[:]:
            self.log.removeHandler(handler)
            handler.close()
        handlers, self.log.level, self.log.propagate, self.log.disabled = self.previous
        for handler in handlers:
            self.log.addHandler(handler)

    def contents(self):
        for handler in self.log.handlers:
            handler.flush()
        return (self.work / "log" / "workflow.log").read_text(encoding="utf-8")

    def send(self, text, **kwargs):
        return submit_message(str(self.work), text, now=self.clock, **kwargs)

    def question(self):
        request = self.runtime.create_consultation(6, "stagnation", "perf_optimize", [], {})
        Path(request["question_path"]).write_text("case3 is slow; A tiling, B partial fusion; recommended A.", encoding="utf-8")
        self.runtime.start_wait(request["request_id"])
        return request

    def test_submission_and_cli_status_do_not_claim_workflow_received(self):
        message = self.send("keep the current scheme")
        self.assertNotIn("Human message read", self.contents())
        self.assertTrue(Path(message["path"]).is_file())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--work-dir", str(self.work), "--status"]), 0)
        self.assertFalse((self.runtime.root / "receipts").exists())
        self.runtime.pending_messages()
        self.assertIn(message["id"], self.contents())
        self.assertTrue((self.runtime.root / "receipts" / f"{message['id']}.json").is_file())

    def test_observation_is_once_across_reads_and_restore_and_preserves_raw(self):
        raw = "  keep the current scheme\r\n\toptimize case3\x1b[31m " + "detailed feedback " * 100
        message = self.send(raw)
        self.runtime.pending_messages()
        self.runtime.all_messages()
        restored = HumanReview(str(self.work), clock=lambda: self.clock)
        restored.pending_messages()
        lines = [line for line in self.contents().splitlines() if "Human message read" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("kind=direction", lines[0])
        self.assertIn("request_id=proactive", lines[0])
        self.assertIn(message["path"], lines[0])
        self.assertNotIn("\x1b", lines[0])
        self.assertNotIn("detailed feedback " * 100, lines[0])
        self.assertEqual(json.loads(Path(message["path"]).read_text(encoding="utf-8"))["text"], raw)
        receipt = json.loads((self.runtime.root / "receipts" / f"{message['id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["message_id"], message["id"])
        self.assertEqual(receipt["observed_at"], 1000.0)
        self.assertEqual(receipt["original_path"], message["path"])

    def test_reader_without_workflow_file_handler_cannot_consume_receipt(self):
        message = self.send("optimize the tail block first")
        handler = next(h for h in self.log.handlers if isinstance(h, logging.FileHandler))
        self.log.removeHandler(handler)
        try:
            self.runtime.pending_messages()
            self.assertFalse((self.runtime.root / "receipts").exists())
        finally:
            self.log.addHandler(handler)
        self.runtime.pending_messages()
        self.assertEqual(self.contents().count("Human message read"), 1)
        self.assertIn(message["id"], self.contents())

    def test_question_extension_timeout_transitions_once_without_poll_noise(self):
        request = self.question()
        request_id = request["request_id"]
        self.runtime.create_consultation(6, "stagnation", "perf_optimize", [], {})
        self.runtime.start_wait(request_id)
        before = self.contents()
        for _ in range(4):
            self.runtime.poll_consultation(request_id)
        self.assertEqual(self.contents(), before)
        first = self.send("请等待")
        self.runtime.poll_consultation(request_id)
        self.clock += 1
        second = self.send("等待一会儿")
        self.runtime.poll_consultation(request_id)
        self.clock = 1720.0
        self.assertEqual(self.runtime.poll_consultation(request_id)["status"], "timed_out")
        self.runtime.poll_consultation(request_id)
        bundle = self.runtime.build_feedback_bundle(request_id)
        self.runtime.build_feedback_bundle(request_id)
        self.runtime.complete_consultation(request_id, "decision.json")
        self.runtime.complete_consultation(request_id, "decision.json")
        text = self.contents()
        for marker in ("Human consultation created", "Human question ready", "Human consultation extended", "Human consultation wait finished",
                       "Human consultation feedback packaged", "Human consultation produced the final decision"):
            self.assertEqual(text.count(marker), 1, marker)
        self.assertIn("status=timed_out", text)
        self.assertIn("human_response_received=False", text)
        self.assertIn(request["question_path"], text)
        self.assertIn(request["evidence_manifest_path"], text)
        self.assertIn(bundle, text)
        self.assertIn(first["id"], text)
        self.assertIn(second["id"], text)
        self.assertNotIn("handling=human P0 suggestion", text)
        self.assertEqual(self.runtime.state()["messages"][first["id"]]["status"], "processed")

    def test_reply_final_decision_and_execution_are_linked(self):
        request = self.question()
        message = self.send("choose A; prioritize small shapes")
        self.runtime.poll_consultation(request["request_id"])
        bundle = self.runtime.build_feedback_bundle(request["request_id"])
        decision = str(self.work / "knowledge" / "stage9" / "decision.json")
        evidence = str(self.work / "iter7" / "fusion_scheme_rationale.md")
        self.runtime.complete_consultation(request["request_id"], decision)
        self.runtime.mark_executed([message["id"]], 7, evidence)
        self.clock += 20
        self.runtime.mark_executed([message["id"]], 7, evidence)
        text = self.contents()
        self.assertIn("status=replied", text)
        self.assertIn("human_response_received=True", text)
        self.assertIn("status=processed handling=human P0 suggestion", text)
        self.assertEqual(text.count("status=executed"), 1)
        self.assertIn(decision, text)
        self.assertIn(evidence, text)
        self.assertIn(bundle, text)
        self.assertEqual(self.runtime.state()["messages"][message["id"]]["decision_path"], decision)

    def test_question_and_unexecuted_direction_have_distinct_audit_entries(self):
        question = self.send("why change the scheme?", kind="question")
        direction = self.send("test once more")
        self.runtime.mark_processed([question["id"]], "answer-decision.json")
        self.runtime.mark_unexecuted([direction["id"]], "iteration limit reached\nkeep the original message")
        self.clock += 10
        self.runtime.mark_unexecuted([direction["id"]], "iteration limit reached\nkeep the original message")
        text = self.contents()
        self.assertIn("status=processed handling=question answer", text)
        self.assertEqual(text.count("status=not_executed"), 1)
        self.assertIn("reason=iteration limit reached keep the original message", text)
        self.assertIn(str(self.runtime.root / "state.json"), text)
        self.assertEqual(self.runtime.state()["messages"][direction["id"]]["reason"], "iteration limit reached\nkeep the original message")

    def test_proactive_bundle_links_messages_and_snapshot(self):
        message = self.send("fix the build only")
        messages = self.runtime.pending_messages()
        path = self.runtime.create_proactive_bundle(3, "compile_failed", "compile failure", [], {}, messages)
        text = self.contents()
        self.assertIn("Proactive human feedback packaged", text)
        self.assertIn("iter=3 scene=compile_failed", text)
        self.assertIn(message["id"], text)
        self.assertIn(path, text)
        self.assertIn("evidence_manifest.json", text)
        self.assertEqual(self.runtime.pending_messages()[0]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
