"""Versioned Stage9 plans are checked before history publication and Stage3 handoff."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import orchestrator
from lib.history_manager import load_history
import test_fusion_routing as fusion
import test_semantic_routing as semantic


class Stage9PlanHandoffTests(unittest.TestCase):
    def setUp(self):
        self.routing = semantic.SemanticRoutingTests(methodName="runTest")
        self.routing.setUp()
        self.addCleanup(self.routing.doCleanups)
        self.fixture = self.routing.fixture
        self.work = self.routing.work
        self.routing.perfs = {1: (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8)}
        self.outputs = []
        self.requests = []

    def read_json(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def write(self, output, payload):
        self.outputs.append(output)
        self.requests.append(self.read_json(output.parent / "request.json"))
        self.fixture.write_json(output, payload)

    def assert_rejected(self, mutate, token=None):
        before = {}

        def agent(output, payload, _iteration, _prompt):
            if not self.outputs:
                before.update(deepcopy(load_history(str(self.work))))
            mutate(payload)
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        with self.assertRaises(RuntimeError):
            self.routing.run_workflow(max_iterations=1)
        self.assertEqual(len(self.outputs), 3)
        self.assertEqual(load_history(str(self.work)), before)
        self.assertNotIn("stage3", self.fixture.events)
        self.assertEqual(self.fixture.events.count("stage6"), 1)
        for output in self.outputs:
            error = self.read_json(output.parent / "validation_error.json")
            if token:
                self.assertIn(token, error["error"])
            self.assertFalse((output.parent / "commit.json").exists())

    def test_schema_template_and_all_cases_are_inputs_and_details_reach_stage3(self):
        def agent(output, payload, _iteration, prompt):
            request = self.read_json(output.parent / "request.json")
            self.assertEqual(request["plan_version"], 2)
            for field, filename in (("schema_path", "decision_schema.json"),
                                    ("template_path", "decision_template.json"),
                                    ("case_catalog_path", "case_catalog.json")):
                path = Path(request[field])
                self.assertEqual(path, output.parent / filename)
                self.assertTrue(path.is_file())
                self.assertIn(str(path), prompt)
                relative = path.relative_to(self.work).as_posix()
                self.assertIn(relative, prompt)
            template = self.read_json(request["template_path"])
            self.assertEqual(template["request_id"], request["request_id"])
            self.assertEqual(template["plan_version"], 2)
            self.assertNotIn("modify_files", template["ledger_entry"])
            schema = self.read_json(request["schema_path"])
            self.assertEqual(schema["type"], "object")
            self.assertIn("suggest_next", schema["properties"])
            catalog_text = Path(request["case_catalog_path"]).read_text(encoding="utf-8")
            self.assertIn("synthetic_op_8", catalog_text)
            # The fastest case is outside worst_6 but remains a legitimate task target.
            self.assertIn("synthetic_op_8", request["known_case_ids"])
            self.assertNotIn("synthetic_op_8", request["performance_case_ids"])
            payload["suggest_next"] = [fusion.stage9_plan_task(["synthetic_op_8"])]
            task = payload["suggest_next"][0]
            task["changes"][0].update(location="fast_case_tile_loop", method="KEEP_FAST_CASE_TILE_METHOD")
            task["acceptance_checks"] = ["VERIFY_FAST_CASE_PRECISION_AND_LATENCY"]
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=1)
        self.assertEqual(len(self.outputs), 1)
        history = load_history(str(self.work))
        ledger = history["ledger"][-1]
        self.assertEqual(history["plan_version"], 2)
        self.assertEqual(ledger["action_plan"]["version"], 2)
        tasks_without_derived_scope = [{key: value for key, value in task.items()
                                        if key not in {"inspect_files", "modify_files"}}
                                       for task in history["suggest_next"]]
        self.assertEqual(ledger["action_plan"]["tasks"], tasks_without_derived_scope)
        self.assertEqual(ledger["modify_files"], ["impl/synthetic_op.py"])
        self.assertEqual(history["suggest_next"][0]["target_cases"], ["synthetic_op_8"])
        prompt = self.fixture.prompts["stage3"]
        for text in ("synthetic_op_8", "fast_case_tile_loop", "KEEP_FAST_CASE_TILE_METHOD",
                     "VERIFY_FAST_CASE_PRECISION_AND_LATENCY", "synthetic operator entry", "T1"):
            self.assertIn(text, prompt)

    def test_unknown_case_is_rejected_without_publishing_an_invented_route(self):
        self.assert_rejected(
            lambda payload: payload.update(suggest_next=[fusion.stage9_plan_task(["unknown_case_7"])]),
            "unknown_case_7")

    def test_declared_case_binding_c2_cannot_authorize_modification_of_c3(self):
        c2, c3 = "impl/c2.py", "impl/c3.py"
        for file in (c2, c3):
            (self.work / file).parent.mkdir(parents=True, exist_ok=True)
            (self.work / file).write_text("# routing fixture\n", encoding="utf-8")

        def mismatch(payload):
            task = fusion.stage9_plan_task(["synthetic_op_7"], file=c2)
            task["changes"][0]["file"] = c3
            payload["suggest_next"] = [task]

        self.assert_rejected(mismatch, c3)

    def test_legacy_ledger_modify_list_cannot_override_derived_changes(self):
        self.assert_rejected(lambda payload: payload["ledger_entry"].update(modify_files=[]),
                             "modify_files")

    def test_legacy_suggestion_scope_is_rejected_even_if_it_agrees_with_changes(self):
        self.assert_rejected(lambda payload: payload["suggest_next"][0].update(
            modify_files=["impl/synthetic_op.py"]), "modify_files")

    def test_modification_of_missing_implementation_is_rejected(self):
        missing = "impl/does_not_exist.py"

        def replace(payload):
            payload["suggest_next"] = [fusion.stage9_plan_task(["synthetic_op_1"], file=missing)]

        self.assert_rejected(replace, missing)

    def test_missing_route_evidence_file_is_rejected(self):
        missing = "impl/imaginary_dispatcher.py"

        def replace(payload):
            payload["suggest_next"][0]["case_bindings"][0]["route_evidence"][0]["file"] = missing

        self.assert_rejected(replace, missing)

    def test_create_cannot_silently_overwrite_an_existing_implementation(self):
        self.assert_rejected(lambda payload: payload["suggest_next"][0]["changes"][0].update(
            operation="create"), "impl/synthetic_op.py")

    def test_inspect_task_cannot_hide_an_explicit_modification(self):
        self.assert_rejected(lambda payload: payload["suggest_next"][0].update(task_type="inspect"))

    def test_stage3_can_resume_after_creating_a_declared_new_file(self):
        new_file = "impl/new_variant.py"

        def agent(output, payload, _iteration, _prompt):
            task = fusion.stage9_plan_task(["synthetic_op_1"], file=new_file, operation="create")
            task["case_bindings"][0]["route_evidence"] = [{
                "file": "impl/synthetic_op.py", "location": "existing entry",
                "explanation": "The existing implementation supplies the case semantics for the new variant",
            }]
            payload["suggest_next"] = [task]
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        original_agent = self.fixture.agent

        def interrupted(agent, role, work, prompt, **kwargs):
            if Path(role).name == "n1_stage3_fix_and_optimize.md":
                (self.work / new_file).write_text("# created before interrupted Stage3 finished\n", encoding="utf-8")
                raise RuntimeError("synthetic interruption after file creation")
            return original_agent(agent, role, work, prompt, **kwargs)

        self.fixture.agent = interrupted
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption after file creation"):
            self.routing.run_workflow(max_iterations=1)
        self.assertEqual(len(self.outputs), 1)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter1_stage3")
        self.assertTrue((self.work / new_file).is_file())
        original_history = deepcopy(load_history(str(self.work)))
        self.fixture.agent = original_agent
        seen_before = len(self.routing.seen)
        self.routing.run_workflow()
        resumed = [(iteration, stage) for iteration, stage, _ in self.routing.seen[seen_before:]]
        self.assertEqual(resumed, [(1, "stage3"), (1, "stage10")])
        self.assertEqual(len(self.outputs), 1, "A valid partially executed plan must not request Stage9 again")
        self.assertEqual(self.fixture.events.count("stage6"), 1)
        self.assertEqual(load_history(str(self.work)), original_history)

    def test_regression_plan_uses_best_code_base_before_restore(self):
        best_file = "impl/best_variant.py"
        self.routing.perfs = {1: (2.0,), 2: (1.8,)}
        self.routing.config["workflow"]["semantic_exit"]["passed_window"] = 8
        original_agent = self.fixture.agent
        restored_before_stage3 = []

        def transport(agent, role, work, prompt, **kwargs):
            stage = fusion.ROLES[Path(role).name]
            iteration = self.fixture.read_state()["iteration"]
            if stage == "stage3" and iteration == 2:
                self.assertTrue((self.work / best_file).is_file())
                restored_before_stage3.append((self.work / best_file).read_text(encoding="utf-8"))
            result = original_agent(agent, role, work, prompt, **kwargs)
            if stage == "stage2":
                (self.work / best_file).write_text("# validated best implementation\n", encoding="utf-8")
            elif stage == "stage3" and iteration == 1:
                # Model a previous refactor that removed a file present in the best snapshot.
                (self.work / best_file).unlink()
            return result

        def decision(output, payload, iteration, prompt):
            if iteration == 2:
                self.assertFalse((self.work / best_file).exists())
                request = self.read_json(output.parent / "request.json")
                best_base = Path(request["implementation_base"])
                self.assertNotEqual(best_base, self.work / "impl")
                self.assertTrue((best_base / "best_variant.py").is_file())
                self.assertIn(str(best_base), prompt)
                payload["suggest_next"] = [fusion.stage9_plan_task(["synthetic_op_1"], file=best_file)]
            self.write(output, payload)
            return True

        self.fixture.agent = transport
        self.routing.stage9_callback = decision
        self.routing.run_workflow(max_iterations=2)
        self.assertEqual(len(self.outputs), 2, "A file in the selected best base must not cause a correction retry")
        self.assertEqual(restored_before_stage3, ["# validated best implementation\n"])
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        history = load_history(str(self.work))
        self.assertEqual(history["ledger"][-1]["modify_files"], [best_file])
        self.assertEqual([round["iter"] for round in history["rounds"]], [1, 2])
        restore = self.read_json(self.work / "selection/rollbacks/iter2/record.json")
        self.assertEqual(restore["status"], "restored")
        self.assertTrue((self.work / "knowledge/regression_patterns.md").is_file())

    def test_partial_error_report_does_not_claim_an_authoritative_case_catalog(self):
        self.fixture.write_json("eval/iter1/perf_result.json", {
            "total_cases": 3, "cases": [{"case_id": "synthetic_op_1"}],
        })
        catalog = orchestrator._stage9_case_catalog(str(self.work), 1, "evaluation_error")
        self.assertFalse(catalog["complete"])
        self.assertIsNone(catalog["case_ids"])
        self.assertEqual(catalog["observed_case_ids"], ["synthetic_op_1"])
        self.assertEqual(catalog["source"], "eval/iter1/perf_result.json")

    def test_repeated_case_ids_do_not_make_a_catalog_complete(self):
        self.fixture.write_json("eval/iter1/perf_result.json", {
            "total_cases": 3,
            "cases": [{"case_id": case} for case in ("synthetic_op_1", "synthetic_op_1", "synthetic_op_2")],
        })
        catalog = orchestrator._stage9_case_catalog(str(self.work), 1, "all_passed")
        self.assertFalse(catalog["complete"])
        self.assertIsNone(catalog["case_ids"])
        self.assertEqual(catalog["observed_case_ids"], ["synthetic_op_1", "synthetic_op_2"])

    def test_invalid_case_totals_never_authorize_the_observed_subset(self):
        for total in (None, 0, -1, 1.0, True, "1"):
            with self.subTest(total=total):
                self.fixture.write_json("eval/iter1/perf_result.json", {
                    "total_cases": total, "cases": [{"case_id": "synthetic_op_1"}],
                })
                catalog = orchestrator._stage9_case_catalog(str(self.work), 1, "evaluation_error")
                self.assertFalse(catalog["complete"])
                self.assertIsNone(catalog["case_ids"])
                self.assertEqual(catalog["observed_case_ids"], ["synthetic_op_1"])

    def test_complete_unique_case_catalog_can_authoritatively_validate_targets(self):
        expected = ["synthetic_op_1", "synthetic_op_2", "synthetic_op_3"]
        self.fixture.write_json("eval/iter1/perf_result.json", {
            "total_cases": 3, "cases": [{"case_id": case} for case in expected],
        })
        catalog = orchestrator._stage9_case_catalog(str(self.work), 1, "all_passed")
        self.assertTrue(catalog["complete"])
        self.assertEqual(catalog["case_ids"], expected)
        self.assertEqual(catalog["observed_case_ids"], expected)


if __name__ == "__main__":
    unittest.main()
