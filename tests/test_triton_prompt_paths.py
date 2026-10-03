"""Check generated development prompts/logs against the submission package layout."""

from pathlib import Path
import unittest
from unittest.mock import patch

import orchestrator
import test_semantic_routing as semantic


class TritonPromptPathTests(unittest.TestCase):
    def new_flow(self):
        flow = semantic.SemanticRoutingTests(methodName="runTest")
        flow.setUp()
        self.addCleanup(flow.doCleanups)
        return flow

    def test_first_implementation_is_a_package_not_a_fixed_root_module(self):
        flow = self.new_flow()
        with patch.object(orchestrator, "log_io", wraps=orchestrator.log_io) as events:
            flow.run_workflow(max_iterations=0)
        prompt = flow.fixture.prompts["stage2"]
        self.assertIn("impl/cann_bench/", prompt.replace("\\", "/"))
        self.assertIn("setup.py", prompt)
        self.assertNotIn("synthetic_op_impl.py", prompt)
        stage2 = next(call for call in events.call_args_list if "阶段2" in call.args[1])
        self.assertTrue(any(Path(item).name == "impl" for item in stage2.args[3]))
        self.assertFalse(any(item.endswith("_impl.py") for item in stage2.args[3]))

    def test_stage3_logs_actual_project_scope_in_every_repair_scene(self):
        for reason in ("compile", "precision", "zero_score", "underperforming", "all_passed"):
            with self.subTest(reason=reason):
                flow = self.new_flow()
                flow.perfs = {1: (0.8,)}
                if reason == "compile":
                    flow.build_failures = {1}
                elif reason == "precision":
                    flow.precision_failures = {1}
                elif reason == "zero_score":
                    flow.perfs = {1: (0.0,)}
                elif reason == "all_passed":
                    flow.perfs = {1: (1.6,)}
                with patch.object(orchestrator, "log_io", wraps=orchestrator.log_io) as events:
                    flow.run_workflow(max_iterations=1)
                repairs = [call for call in events.call_args_list if "阶段3" in call.args[1]]
                self.assertTrue(repairs)
                for call in repairs:
                    for paths in (call.args[2], call.args[3]):
                        self.assertTrue(any(Path(item).name == "impl" for item in paths))
                        self.assertFalse(any(item.endswith("_impl.py") for item in paths))
                if reason == "zero_score":
                    prompt = flow.fixture.prompts["stage3"]
                    self.assertNotIn("pip install -e", prompt)
                    self.assertIn("python3 -m pip install . --force-reinstall --no-deps", prompt)


if __name__ == "__main__":
    unittest.main()
