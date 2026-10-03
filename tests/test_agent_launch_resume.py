"""An agent launch error must leave the original Stage3 checkpoint recoverable."""

import errno
from pathlib import Path
import unittest
from unittest.mock import patch

from lib import agent_runner
from lib.fusion_evidence import implementation_hash, load_evidence
from lib.history_manager import load_history
import test_human_routing as human_routing


class AgentLaunchResumeTests(unittest.TestCase):
    def setUp(self):
        # Reuse synthetic transports, not routing or storage mocks. The real
        # main(), Stage9 validation, evidence binding and human receipts run.
        self.flow = human_routing.HumanRoutingTests(methodName="runTest")
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)
        self.work = self.flow.work

    def test_iter3_popen_e2big_resumes_same_stage_without_repeating_measurements_or_human_review(self):
        self.flow.routing.perfs = {1: (1.6,), 2: (1.8,), 3: (2.0,)}
        messages, launches, source_hashes = [], [], []
        role = Path(__file__).resolve().parents[1] / "roles/n1_stage3_fix_and_optimize.md"

        def popen_failure(*args, **kwargs):
            # The mock sits at Popen, after the real runner constructs argv.
            # It has not spawned an agent or changed operator code. Even with
            # fixed prompt transport, OS launch failures must stay recoverable.
            launches.append((args, kwargs))
            raise OSError(errno.E2BIG, "Argument list too long")

        def arriving(stage, iteration, prompt):
            if stage == "stage4" and iteration == 3 and not messages:
                messages.append(self.flow.submit("保持融合方向，优先调整慢 case 的切块"))
            if stage == "stage3" and iteration == 3:
                source_hashes.append(implementation_hash(self.work / "impl"))
                with patch.object(agent_runner, "_agent_env", return_value={}), \
                        patch.object(agent_runner.subprocess, "Popen", side_effect=popen_failure):
                    agent_runner.run_cannbot(str(role), str(self.work), prompt,
                                            config={"cli": "/offline/cannbot", "cwd": str(self.work)})

        self.flow.on_stage = arriving
        with self.assertRaises(OSError) as error:
            self.flow.run_flow(3)
        self.assertEqual(error.exception.errno, errno.E2BIG)
        self.assertEqual(len(launches), 1)
        self.assertEqual(len(messages), 1)

        checkpoint = self.flow.fixture.read_state()
        self.assertEqual(checkpoint["iteration"], 3)
        self.assertEqual(checkpoint["current_stage"], "iter3_stage3")
        self.assertEqual(checkpoint["stage9_context"]["iteration"], 3)
        self.assertEqual(checkpoint["stage9_context"]["phase"], "developing")
        self.assertEqual(checkpoint["stage9_context"]["development_reason"], "perf_pass_optimize")
        self.assertEqual(implementation_hash(self.work / "impl"), source_hashes[0])
        self.assertFalse(load_evidence(self.work)["eligible"])
        self.assertIn("Development is pending", load_evidence(self.work)["reason"])

        person = self.flow.human.all_messages()[0]
        self.assertEqual(person["id"], messages[0]["id"])
        self.assertEqual(person["status"], "processed")
        self.assertNotIn("executed_iteration", person)
        receipt = self.work / "develop/iter3/human_feedback.json"
        self.assertFalse(receipt.exists())
        self.assertTrue(any(item.get("priority") == "P0" and item.get("source") == "human"
                            and item.get("human_message_id") == person["id"]
                            for item in load_history(str(self.work))["suggest_next"]))

        # Capture real knowledge output, including the two measured gains, so
        # a duplicate Stage9/knowledge write cannot hide behind a count check.
        history_path = self.work / "knowledge/history.json"
        patterns_path = self.work / "knowledge/proven_patterns.md"
        history_before = history_path.read_bytes()
        patterns_before = patterns_path.read_bytes()
        self.assertTrue(patterns_before)
        decisions_before = len(self.flow.decisions)
        seen_before = len(self.flow.routing.seen)
        evaluations_before = self.flow.fixture.events.count("stage6")
        self.assertEqual(evaluations_before, 3)

        # The ordinary fixture patches sys.argv to --work-dir on this same
        # directory. Do not reset its checkpoint or create a new work.
        self.flow.on_stage = None
        self.flow.run_flow()

        resumed = [(iteration, stage) for iteration, stage, _
                   in self.flow.routing.seen[seen_before:]]
        self.assertEqual(resumed, [(3, "stage3"), (3, "stage10")])
        self.assertEqual(self.flow.fixture.read_state()["iteration"], 3)
        self.assertEqual(self.flow.fixture.events.count("stage6"), evaluations_before)
        self.assertEqual(len(self.flow.decisions), decisions_before)
        self.assertEqual(history_path.read_bytes(), history_before)
        self.assertEqual(patterns_path.read_bytes(), patterns_before)
        self.assertEqual(self.flow.human.all_messages()[0]["status"], "executed")
        self.assertEqual(self.flow.human.all_messages()[0]["executed_iteration"], 3)
        self.assertTrue(receipt.is_file())
        self.assertTrue(load_evidence(self.work)["eligible"])


if __name__ == "__main__":
    unittest.main()
