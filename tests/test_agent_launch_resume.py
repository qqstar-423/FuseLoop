"""An agent launch error must leave the original Stage3 checkpoint recoverable."""

import errno
from pathlib import Path
import unittest
from unittest.mock import patch

from lib import agent_runner
from lib.fusion_evidence import implementation_hash, load_evidence
import test_semantic_routing as semantic_routing


class AgentLaunchResumeTests(unittest.TestCase):
    def setUp(self):
        # Reuse synthetic transports, not routing or storage mocks. The real
        # main(), Stage9 validation and evidence binding run.
        self.flow = semantic_routing.SemanticRoutingTests(methodName="runTest")
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)
        self.work = self.flow.work

    def test_iter3_popen_e2big_resumes_same_stage_without_repeating_measurements_or_review(self):
        self.flow.perfs = {1: (1.6,), 2: (1.8,), 3: (2.0,)}
        launches, source_hashes = [], []
        original_agent = self.flow.original_agent
        role = Path(__file__).resolve().parents[1] / "roles/n1_stage3_fix_and_optimize.md"

        def popen_failure(*args, **kwargs):
            # The mock sits at Popen, after the real runner constructs argv.
            # It has not spawned an agent or changed operator code. Even with
            # fixed prompt transport, OS launch failures must stay recoverable.
            launches.append((args, kwargs))
            raise OSError(errno.E2BIG, "Argument list too long")

        def interrupted(agent, role_path, work, prompt, **kwargs):
            iteration = self.flow.fixture.read_state()["iteration"]
            if Path(role_path).name == role.name and iteration == 3:
                source_hashes.append(implementation_hash(self.work / "impl"))
                with patch.object(agent_runner, "_agent_env", return_value={}), \
                        patch.object(agent_runner.subprocess, "Popen", side_effect=popen_failure):
                    agent_runner.run_cannbot(str(role), str(self.work), prompt,
                                            config={"cli": "/offline/cannbot", "cwd": str(self.work)})
            return original_agent(agent, role_path, work, prompt, **kwargs)

        self.flow.original_agent = interrupted
        with self.assertRaises(OSError) as error:
            self.flow.run_workflow(max_iterations=3)
        self.assertEqual(error.exception.errno, errno.E2BIG)
        self.assertEqual(len(launches), 1)

        checkpoint = self.flow.fixture.read_state()
        self.assertEqual(checkpoint["iteration"], 3)
        self.assertEqual(checkpoint["current_stage"], "iter3_stage3")
        self.assertEqual(checkpoint["stage9_context"]["iteration"], 3)
        self.assertEqual(checkpoint["stage9_context"]["phase"], "developing")
        self.assertEqual(checkpoint["stage9_context"]["development_reason"], "perf_pass_optimize")
        self.assertEqual(implementation_hash(self.work / "impl"), source_hashes[0])
        self.assertFalse(load_evidence(self.work)["eligible"])
        self.assertIn("Development is pending", load_evidence(self.work)["reason"])

        self.assertFalse((self.work / "human_review").exists())

        # Capture real knowledge output, including the two measured gains, so
        # a duplicate Stage9/knowledge write cannot hide behind a count check.
        history_path = self.work / "knowledge/history.json"
        patterns_path = self.work / "knowledge/proven_patterns.md"
        history_before = history_path.read_bytes()
        patterns_before = patterns_path.read_bytes()
        self.assertTrue(patterns_before)
        decisions_before = self.flow.fixture.events.count("stage9")
        seen_before = len(self.flow.seen)
        evaluations_before = self.flow.fixture.events.count("stage6")
        self.assertEqual(evaluations_before, 3)

        # The ordinary fixture patches sys.argv to --work-dir on this same
        # directory. Do not reset its checkpoint or create a new work.
        self.flow.original_agent = original_agent
        with patch("builtins.input", side_effect=AssertionError("Interactive input is forbidden")), \
                patch("time.sleep", side_effect=AssertionError("Workflow must not wait for input")):
            self.flow.run_workflow()

        resumed = [(iteration, stage) for iteration, stage, _
                   in self.flow.seen[seen_before:]]
        self.assertEqual(resumed, [(3, "stage3"), (3, "stage10")])
        self.assertEqual(self.flow.fixture.read_state()["iteration"], 3)
        self.assertEqual(self.flow.fixture.events.count("stage6"), evaluations_before)
        self.assertEqual(self.flow.fixture.events.count("stage9"), decisions_before)
        self.assertEqual(history_path.read_bytes(), history_before)
        self.assertEqual(patterns_path.read_bytes(), patterns_before)
        self.assertFalse((self.work / "human_review").exists())
        self.assertNotIn("human_feedback", self.flow.fixture.prompts["stage3"])
        self.assertNotIn("Human feedback", self.flow.fixture.prompts["stage10"])
        self.assertTrue(load_evidence(self.work)["eligible"])


if __name__ == "__main__":
    unittest.main()
