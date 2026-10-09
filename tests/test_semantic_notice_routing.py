"""Semantic trigger notices agree with real routing, disk logs and Stage9 inputs."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_semantic_routing as semantic_routing_tests


class SemanticNoticeRoutingTests(unittest.TestCase):
    def setUp(self):
        self.routing = semantic_routing_tests.SemanticRoutingTests(methodName="runTest")
        self.routing.setUp()
        self.addCleanup(self.routing.doCleanups)
        self.work = self.routing.work

    def events(self):
        return json.loads((self.work / "selection/semantic_events.json").read_text(encoding="utf-8"))

    def requests(self, iteration):
        return [json.loads(path.read_text(encoding="utf-8")) for path in
                (self.work / "knowledge/stage9" / f"iter{iteration}").glob("*/request.json")]

    def assert_event_delivered(self, event):
        requests = self.requests(event["iteration"])
        self.assertTrue(requests)
        for request in requests:
            injected = request["semantic_event"]
            # Stage6 records the event before Stage9 loads it, so Stage9 sees a retry.
            self.assertTrue(injected["reused"])
            self.assertEqual({key: value for key, value in injected.items() if key != "reused"}, event)
            prompt = Path(request["prompt_path"]).read_text(encoding="utf-8")
            self.assertIn(f"entry {event['entry_count']} into the {event['scene']} stagnation scene", prompt)
            self.assertIn("selection/semantic_events.json", prompt)
            self.assertIn(event["state_path"], prompt)

    def assert_banner_count(self, scene_number, entry_count, expected=1):
        banner = f"===== Scenario {scene_number} stagnation trigger: entry {entry_count} into this scene ====="
        for name in ("workflow.log", "state_transitions.log"):
            text = (self.work / "log" / name).read_text(encoding="utf-8")
            self.assertEqual(text.count(banner), expected, name)

    def test_passed_window_has_separate_count_and_reaches_stage9_before_exit(self):
        self.routing.config["workflow"]["semantic_exit"] = {
            "passed_window": 1, "underperforming_window": 4,
        }
        self.routing.perfs = {1: (1.5, 1.0), 2: (1.5, 1.0)}
        self.routing.run_workflow(max_iterations=6, file_logs=True)
        data = self.events()
        self.assertEqual(data["counts"], {"passed": 1, "underperforming": 0})
        self.assertEqual(len(data["events"]), 1)
        event = data["events"][0]
        self.assertEqual(event["iteration"], 2)
        self.assertEqual(event["window"]["required_improvements"], 1)
        self.assertEqual(self.routing.fixture.read_state()["stopped_by"], "semantic_stagnation")
        self.assert_banner_count(1, 1)
        for name in ("workflow.log", "state_transitions.log"):
            text = (self.work / "log" / name).read_text(encoding="utf-8")
            self.assertIn("workflow.semantic_exit.passed_window=1", text)
            self.assertIn("scenario 1: every case's speedup is >= 1", text)
            self.assertIn("x=1 consecutive valid performance iterations", text)
            self.assertIn("1 baseline sample plus 1 valid performance iterations within the same scene (2 samples in total); invalid rounds do not count", text)
            self.assertIn("checkpoint resume does not double count", text)
            self.assertIn("iter1→iter2", text)
            self.assertIn("0.00% < 5.00%", text)
        self.assert_event_delivered(event)

    def test_three_new_underperforming_evaluations_count_once_each(self):
        self.routing.config["workflow"]["semantic_exit"] = {
            "passed_window": 4, "underperforming_window": 1,
        }
        self.routing.perfs = {1: (.8, 2.0), 2: (.81, 2.0), 3: (.82, 2.0), 4: (.83, 2.0)}
        with patch("builtins.input", side_effect=AssertionError("Interactive input is forbidden")), \
                patch("time.sleep", side_effect=AssertionError("Workflow must not wait for input")):
            self.routing.run_workflow(max_iterations=4, file_logs=True)
        data = self.events()
        self.assertEqual(data["counts"], {"passed": 0, "underperforming": 3})
        self.assertEqual([event["iteration"] for event in data["events"]], [2, 3, 4])
        self.assertEqual([event["entry_count"] for event in data["events"]], [1, 2, 3])
        for event in data["events"]:
            self.assertEqual(event["window"]["required_improvements"], 1)
            self.assert_banner_count(2, event["entry_count"])
            self.assert_event_delivered(event)
        for name in ("workflow.log", "state_transitions.log"):
            text = (self.work / "log" / name).read_text(encoding="utf-8")
            self.assertIn("workflow.semantic_exit.underperforming_window=1", text)
            self.assertIn("scenario 2: not every case's speedup is >= 1 (at least one case < 1)", text)
            self.assertIn("y=1 consecutive valid performance iterations", text)
            self.assertIn("1 baseline sample plus 1 valid performance iterations within the same scene (2 samples in total); invalid rounds do not count", text)
            self.assertIn("checkpoint resume does not double count", text)
        self.assertFalse((self.work / "human_review").exists())
        for iteration in range(1, 5):
            for request in self.requests(iteration):
                self.assertEqual(request["phase"], "decision")
                self.assertFalse(any(key.startswith("human_") for key in request))
        self.assertEqual(self.routing.fixture.events.count("stage3"), 4)
        self.assertEqual(self.routing.fixture.events.count("stage10"), 1)
        self.assertNotIn("Human feedback", self.routing.fixture.prompts["stage10"])
        self.assertEqual(self.routing.fixture.read_state()["stopped_by"], "max_iterations")

    def test_stage9_interruption_resume_keeps_event_and_does_not_repeat_banner(self):
        self.routing.perfs = {1: (.8, 2.0), 2: (.81, 2.0), 3: (.82, 2.0)}
        self.routing.interrupt_stage9_iteration = 3
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.routing.run_workflow(max_iterations=3, file_logs=True)
        before = (self.work / "selection/semantic_events.json").read_bytes()
        self.assertEqual(self.routing.fixture.read_state()["current_stage"], "iter3_stage9")
        self.routing.interrupt_stage9_iteration = None
        self.routing.run_workflow(file_logs=True)
        self.assertEqual((self.work / "selection/semantic_events.json").read_bytes(), before)
        self.assertEqual(self.routing.fixture.events.count("stage6"), 3)
        self.assertEqual(len(self.requests(3)), 2)
        self.assert_banner_count(2, 1)
        self.assert_event_delivered(self.events()["events"][0])

    def test_compile_failure_does_not_reuse_previous_stagnation_or_increment_count(self):
        self.routing.perfs = {1: (.8, 2.0), 2: (.81, 2.0), 3: (.82, 2.0), 4: (.83, 2.0)}
        self.routing.build_failures = {4}
        self.routing.run_workflow(max_iterations=4, file_logs=True)
        data = self.events()
        self.assertEqual(data["counts"], {"passed": 0, "underperforming": 1})
        self.assertEqual([event["iteration"] for event in data["events"]], [3])
        self.assert_banner_count(2, 1)
        self.assert_banner_count(2, 2, expected=0)
        self.assert_event_delivered(data["events"][0])
        requests = self.requests(4)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["scene"], "compile")
        self.assertIsNone(requests[0]["semantic_event"])
        self.assertNotIn(" stagnation scene", Path(requests[0]["prompt_path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
