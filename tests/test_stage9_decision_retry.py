"""Real Stage9 validation/correction flow with synthetic agent transports only."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import orchestrator
from lib.history_manager import load_history, save_history
import test_semantic_routing as semantic_routing
import test_human_routing as human_routing


C2 = "impl/c2/test_operator.py"
C3 = "impl/c3/test_operator.py"


class Stage9DecisionRetryTests(unittest.TestCase):
    def setUp(self):
        self.routing = semantic_routing.SemanticRoutingTests(methodName="runTest")
        self.routing.setUp()
        self.addCleanup(self.routing.doCleanups)
        self.fixture = self.routing.fixture
        self.work = self.routing.work
        self.routing.perfs = {1: (1.6,), 2: (1.7,)}
        self.routing.config["workflow"]["semantic_exit"]["passed_window"] = 8
        for relative in (C2, C3):
            path = self.work / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic implementation\n", encoding="utf-8")

    def write(self, path, payload):
        self.routing.write_json(path, payload)

    def conflicting_plan(self, payload):
        payload["ledger_entry"]["readonly_files"] = [C3]
        task = deepcopy(payload["suggest_next"][0])
        task.update(action="Change the tile traversal in the bf16 implementation",
                    reason="The measured slow case routes to c3")
        task["changes"] = [{"file": C3, "operation": "modify", "location": "tile traversal",
                            "method": "Reorder the tile traversal to balance core workloads"}]
        for binding in task["case_bindings"]:
            binding["implementation_files"] = [C3]
            binding["route_evidence"] = [{"file": C3, "location": "bf16 entry",
                                          "explanation": "This measured case uses the bf16 implementation"}]
        payload["suggest_next"] = [task]

    def corrected_plan(self, payload):
        self.conflicting_plan(payload)
        payload["ledger_entry"]["readonly_files"] = [C2]

    def attempts_for(self, iteration):
        return sorted((self.work / f"knowledge/stage9/iter{iteration}").rglob("request.json"))

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def assert_attempt_chain(self, outputs):
        requests = [self.read_json(output.parent / "request.json") for output in outputs]
        self.assertEqual(len({request["request_id"] for request in requests}), len(outputs))
        for index, (output, request) in enumerate(zip(outputs, requests)):
            expected_dir = outputs[0].parent if index == 0 else outputs[0].parent / f"retry{index}"
            self.assertEqual(output.parent, expected_dir)
            self.assertEqual(request["correction_attempt"], index)
            self.assertEqual(request["attempt_number"], index + 1)
            self.assertEqual(request["max_attempts"], 3)
            self.assertEqual(request["max_corrections"], 2)
            if index:
                self.assertEqual(request["retry_of_request_id"], requests[index - 1]["request_id"])
                self.assertEqual(Path(request["rejected_decision_path"]), outputs[index - 1])
            template = self.read_json(Path(request["template_path"]))
            schema = self.read_json(Path(request["schema_path"]))
            self.assertEqual(template["request_id"], request["request_id"])
            self.assertEqual(schema["properties"]["request_id"]["const"], request["request_id"])
        return requests

    def assert_pattern_contract(self, output, field):
        request = self.read_json(output.parent / "request.json")
        schema = self.read_json(Path(request["schema_path"]))
        template = self.read_json(Path(request["template_path"]))
        explanation = "why_it_worked" if field == "proven_pattern" else "why_it_failed"
        required = {"what_changed", explanation, "fusion_related", "case_analysis",
                    "evidence", "applicability", "next_action"}
        self.assertIn(field, schema["required"])
        pattern_schema = schema["properties"][field]
        self.assertTrue(required.issubset(pattern_schema["required"]))
        self.assertEqual(pattern_schema["properties"]["applicability"]["type"], "string")
        analysis_schema = pattern_schema["properties"]["case_analysis"]["items"]
        self.assertTrue({"case_id", "observation", "explanation"}.issubset(analysis_schema["required"]))
        self.assertEqual(analysis_schema["properties"]["explanation"]["type"], "string")
        self.assertTrue(required.issubset(template[field]))
        self.assertIsInstance(template[field]["applicability"], str)
        self.assertTrue(template[field]["case_analysis"])
        for analysis in template[field]["case_analysis"]:
            self.assertIn("explanation", analysis)
        self.assertEqual(template["request_id"], request["request_id"])
        self.assertEqual(template["iteration"], request["iteration"])
        other = "regression_pattern" if field == "proven_pattern" else "proven_pattern"
        self.assertNotIn(other, schema["required"])

    def assert_no_required_patterns(self, output):
        request = self.read_json(output.parent / "request.json")
        schema = self.read_json(Path(request["schema_path"]))
        self.assertNotIn("proven_pattern", schema["required"])
        self.assertNotIn("regression_pattern", schema["required"])

    def check_precision_cross_scene_fields(self, *, correct_on_retry):
        self.routing.perfs = {1: (1.6,) * 24, 2: (1.8,) * 24}
        self.routing.precision_failures = {2}
        case_ids = [f"synthetic_op_{index}" for index in range(16, 25)]
        summary = ("PRECISION_ADDRESS_AUDIT: case 16 accesses an invalid address; cases 17-24 "
                   "fail or are skipped after the device error. Verify bounds and every case.")
        requests = []
        before = {}

        def agent(output, payload, iteration, prompt):
            if iteration == 1:
                self.write(output, payload)
                return True
            requests.append((output, prompt))
            payload["ledger_entry"]["evaluation_summary"] = summary
            if len(requests) == 1:
                before.update(deepcopy(load_history(str(self.work))))
            else:
                self.assertEqual(load_history(str(self.work)), before)
            if len(requests) == 1 or not correct_on_retry:
                payload["ledger_entry"]["case_analysis"] = [
                    {"case_id": case_id, "observation": "Runtime failure or cascade skip",
                     "explanation": "An invalid memory access may corrupt the device state",
                     "evidence": "eval/iter2/precision_result.json",
                     "next_action": "Verify address bounds and rerun correctness checks"}
                    for case_id in case_ids]
                payload["regression_pattern"] = {
                    "id": "RP2_masked_load_oob", "title": "Masked load accesses invalid memory",
                    "trigger": "An out-of-bounds address reaches a masked load",
                    "consequence": "The device fails and later cases cannot execute",
                    "root_cause": "Address bounds need verification independently of the load mask",
                    "fix": "Check valid addresses before loading",
                    "evidence": "eval/iter2/precision_result.json", "scope": "Padding loads"}
                payload["pitfall"] = {
                    "id": "PF2_self_test_memory_layout", "title": "Self-test differs from evaluation",
                    "description": "A different allocation layout can expose an invalid address",
                    "evidence": "eval/iter2/precision_result.json",
                    "mitigation": "Audit address bounds and continuous calls"}
            else:
                payload["ledger_entry"].pop("case_analysis", None)
                payload.pop("regression_pattern", None)
                payload.pop("pitfall", None)
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        if correct_on_retry:
            self.routing.run_workflow(max_iterations=2, file_logs=True)
        else:
            with self.assertRaises(RuntimeError):
                self.routing.run_workflow(max_iterations=2, file_logs=True)
        self.assertEqual(len(requests), 2 if correct_on_retry else 3)
        self.assert_attempt_chain([output for output, _ in requests])
        first_path, _ = requests[0]
        retry_path, retry_prompt = requests[-1]
        first_error = self.read_json(first_path.parent / "validation_error.json")
        self.assertGreaterEqual(len(first_error["errors"]), 3)
        for field in ("ledger_entry.case_analysis", "regression_pattern", "pitfall"):
            self.assertTrue(any(field in error for error in first_error["errors"]))
            self.assertIn(field, retry_prompt)
        # A precision repair should omit unrelated knowledge, not invent missing performance claims.
        self.assertNotIn("regression_pattern.what_changed", first_error["error"])
        self.assertNotIn("pitfall.verdict", first_error["error"])
        self.assertEqual(len(self.read_json(first_path)["ledger_entry"]["case_analysis"]), 9)
        for output, _ in requests:
            request = self.read_json(output.parent / "request.json")
            self.assertEqual(request["scene"], "precision")
            self.assertEqual(request["perf_diff"], {})
            self.assertEqual(request["performance_case_ids"], [])
            self.assertIsNone(request["question_path"])
            schema = self.read_json(Path(request["schema_path"]))
            template = self.read_json(Path(request["template_path"]))
            for field in ("proven_pattern", "regression_pattern", "pitfall"):
                self.assertNotIn(field, schema["properties"])
                self.assertNotIn(field, schema["required"])
                self.assertNotIn(field, template)
            self.assertIs(schema["additionalProperties"], False)
            case_schema = schema["properties"]["ledger_entry"]["properties"]["case_analysis"]
            self.assertEqual(case_schema["maxItems"], 0)
            self.assertEqual(template["ledger_entry"].get("case_analysis", []), [])
            self.assertEqual(self.read_json(output.parent / "history_before.json"), before)
        self.assertEqual(self.fixture.events.count("stage6"), 1)
        self.assertEqual(self.fixture.read_state()["iteration"], 2)
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage9"],
                         [1] + [2] * len(requests))
        self.assertFalse((first_path.parent / "commit.json").exists())
        for filename in ("proven_patterns.md", "regression_patterns.md", "tech_lead_pitfalls.md"):
            self.assertFalse((self.work / "knowledge" / filename).exists())
        history = load_history(str(self.work))
        if correct_on_retry:
            self.assertTrue((retry_path.parent / "commit.json").is_file())
            self.assertFalse((retry_path.parent / "validation_error.json").exists())
            self.assertEqual(history["rounds"], before["rounds"])
            self.assertEqual(history["ledger"][0], before["ledger"][0])
            current = history["ledger"][-1]
            self.assertEqual(current["evaluation_summary"], summary)
            self.assertEqual(current.get("case_analysis", []), [])
            self.assertEqual(Path(current["stage9_decision_path"]), retry_path)
            self.assertIn(summary, self.fixture.prompts["stage3"])
            self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1, 2])
        else:
            self.assertEqual(history, before)
            self.assertFalse((retry_path.parent / "commit.json").exists())
            self.assertEqual(self.fixture.read_state()["current_stage"], "iter2_stage9")
            retry_error = self.read_json(retry_path.parent / "validation_error.json")
            self.assertIs(retry_error["will_retry"], False)
            for field in ("ledger_entry.case_analysis", "regression_pattern", "pitfall"):
                self.assertTrue(any(field in error for error in retry_error["errors"]))
            self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1])
            self.assertNotIn("stage10", self.fixture.events)

    def test_precision_reports_all_cross_scene_fields_then_hands_corrected_summary_to_stage3(self):
        self.check_precision_cross_scene_fields(correct_on_retry=True)

    def test_precision_repeated_cross_scene_fields_stop_without_publishing_or_development(self):
        self.check_precision_cross_scene_fields(correct_on_retry=False)

    def check_profiler_inspection_correction(self, *, correct_directory):
        profiler_dir = self.work / "eval/iter2/prof_data"
        profiler_file = profiler_dir / "synthetic_profiler.json"
        # The real archive step copies fresh profiler data beside the synthetic report.
        profiler_source = self.work / "prof_data/synthetic_profiler.json"
        profiler_source.parent.mkdir(parents=True, exist_ok=True)
        profiler_source.write_text('{"case_id": "synthetic_op_1", "kernel_time_us": 12}', encoding="utf-8")
        requests = []
        before = {}

        def agent(output, payload, iteration, prompt):
            if iteration == 1:
                self.write(output, payload)
                return True
            requests.append((output, prompt))
            self.corrected_plan(payload)
            task = payload["suggest_next"][0]
            task.update(task_id="T3", task_type="inspect",
                        action="Inspect the measured case profiler and its implementation")
            inspection = {
                "file": "eval/iter2/prof_data", "operation": "inspect",
                "location": "Profiler evidence for synthetic_op_1",
                "method": "Read kernel timing and relate it to the bound implementation"}
            task["changes"] = [inspection]
            if len(requests) == 1:
                before.update(deepcopy(load_history(str(self.work))))
            else:
                self.assertEqual(load_history(str(self.work)), before)
                task["changes"].insert(0, {
                    "file": C3, "operation": "inspect", "location": "bf16 entry",
                    "method": "Read the case route and kernel body alongside profiler evidence"})
                if correct_directory:
                    inspection["file"] = profiler_file.relative_to(self.work).as_posix()
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        if correct_directory:
            self.routing.run_workflow(max_iterations=2, file_logs=True)
        else:
            with self.assertRaises(RuntimeError):
                self.routing.run_workflow(max_iterations=2, file_logs=True)
        self.assertEqual(len(requests), 2 if correct_directory else 3)
        self.assert_attempt_chain([output for output, _ in requests])
        first_path, _ = requests[0]
        retry_path, _ = requests[-1]
        retry_prompt = requests[1][1]
        first_error = self.read_json(first_path.parent / "validation_error.json")
        self.assertGreaterEqual(len(first_error["errors"]), 2)
        self.assertTrue(any("no inspect operation" in error for error in first_error["errors"]))
        self.assertTrue(any("eval/iter2/prof_data" in error and "具体文件" in error
                            for error in first_error["errors"]))
        for token in ("T3", "no inspect operation", "eval/iter2/prof_data", "具体文件"):
            self.assertIn(token, retry_prompt)
        for output, _ in requests:
            self.assertEqual(self.read_json(output.parent / "request.json")["scene"], "all_passed")
            self.assertEqual(self.read_json(output.parent / "history_before.json"), before)
        self.assertFalse((first_path.parent / "commit.json").exists())
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        self.assertEqual(self.fixture.read_state()["iteration"], 2)
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage9"],
                         [1] + [2] * len(requests))
        history = load_history(str(self.work))
        if correct_directory:
            self.assertTrue((retry_path.parent / "commit.json").is_file())
            self.assertFalse((retry_path.parent / "validation_error.json").exists())
            self.assertEqual(len(list(first_path.parent.parent.rglob("commit.json"))), 1)
            self.assertEqual(history["rounds"], before["rounds"])
            self.assertEqual(history["ledger"][0], before["ledger"][0])
            self.assertEqual([entry["iter"] for entry in history["ledger"]], [1, 2])
            self.assertEqual(Path(history["ledger"][-1]["stage9_decision_path"]), retry_path)
            self.assertEqual(history["suggest_next"][0]["inspect_files"],
                             [C3, "eval/iter2/prof_data/synthetic_profiler.json"])
            self.assertEqual(history["suggest_next"][0]["modify_files"], [])
            self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1, 2])
            knowledge = (self.work / "knowledge/proven_patterns.md").read_text(encoding="utf-8")
            self.assertEqual(knowledge.count("## iter2:"), 1)
            self.assertIn(str(retry_path), knowledge)
        else:
            retry_error = self.read_json(retry_path.parent / "validation_error.json")
            self.assertIs(retry_error["will_retry"], False)
            self.assertIn("eval/iter2/prof_data", retry_error["error"])
            self.assertIn("具体文件", retry_error["error"])
            self.assertNotIn("no inspect operation", retry_error["error"])
            self.assertFalse((retry_path.parent / "commit.json").exists())
            self.assertFalse((self.work / "knowledge/proven_patterns.md").exists())
            self.assertEqual(history, before)
            self.assertEqual(self.fixture.read_state()["current_stage"], "iter2_stage9")
            self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1])
            self.assertNotIn("stage10", self.fixture.events)

    def test_inspect_binding_and_profiler_directory_errors_are_fixed_in_one_correction(self):
        self.check_profiler_inspection_correction(correct_directory=True)

    def test_inspect_binding_fix_cannot_deliver_a_remaining_profiler_directory(self):
        self.check_profiler_inspection_correction(correct_directory=False)

    def test_independent_task_and_file_errors_reach_two_corrections_before_one_commit(self):
        source = self.work / "prof_data"
        source.mkdir(parents=True, exist_ok=True)
        good_reports = [f"eval/iter2/prof_data/synthetic_{index}.json" for index in range(3)]
        for index in range(3):
            (source / f"synthetic_{index}.json").write_text('{"kernel_time_us": 12}', encoding="utf-8")
        bad_reports = ["eval/iter2/prof_data", "eval/iter2/missing_alpha.json",
                       "eval/iter2/missing_beta.json"]
        outputs, prompts = [], []
        before = {}

        def agent(output, payload, iteration, prompt):
            if iteration == 1:
                self.write(output, payload)
                return True
            outputs.append(output)
            prompts.append(prompt)
            attempt = len(outputs)
            if attempt == 1:
                before.update(deepcopy(load_history(str(self.work))))
            else:
                self.assertEqual(load_history(str(self.work)), before)
            self.corrected_plan(payload)
            base = payload["suggest_next"][0]
            tasks = []
            for task_id, implementation in (("T_CONFLICT_C2", C2), ("T_CONFLICT_C3", C3)):
                task = deepcopy(base)
                task["task_id"] = task_id
                task["changes"][0]["file"] = implementation
                for binding in task["case_bindings"]:
                    binding["implementation_files"] = [implementation]
                    binding["route_evidence"][0]["file"] = implementation
                tasks.append(task)
            file_task = deepcopy(base)
            file_task.update(task_id="T_FILES", task_type="inspect", action="Inspect each concrete profiler report")
            file_task["changes"][0]["operation"] = "inspect"
            for index, valid in enumerate(good_reports):
                path = bad_reports[index] if attempt == 1 or (attempt == 2 and index == 2) else valid
                file_task["changes"].append({
                    "file": path, "operation": "inspect", "location": "Measured kernel timing",
                    "method": "Compare this timing with the bound implementation"})
            tasks.append(file_task)
            syntax_task = deepcopy(base)
            syntax_task.update(task_id="T_SYNTAX", task_type="inspect",
                               action="" if attempt == 1 else "Inspect the mapped case route")
            syntax_task["changes"][0]["operation"] = "inspect"
            tasks.append(syntax_task)
            payload["suggest_next"] = tasks
            payload["ledger_entry"]["readonly_files"] = [C2, C3] if attempt == 1 else []
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=2, file_logs=True)
        self.assertEqual(len(outputs), 3)
        requests = self.assert_attempt_chain(outputs)
        first_error = self.read_json(outputs[0].parent / "validation_error.json")
        second_error = self.read_json(outputs[1].parent / "validation_error.json")
        self.assertIs(first_error["will_retry"], True)
        self.assertIs(second_error["will_retry"], True)
        self.assertGreaterEqual(len(first_error["errors"]), 6)
        for task_id, file in (("T_CONFLICT_C2", C2), ("T_CONFLICT_C3", C3)):
            self.assertTrue(any(task_id in error and file in error and "readonly_files" in error
                                for error in first_error["errors"]))
            self.assertIn(task_id, prompts[1])
            self.assertIn(file, prompts[1])
        self.assertTrue(any("suggest_next[3].action" in error for error in first_error["errors"]))
        for path in bad_reports:
            self.assertTrue(any(path in error for error in first_error["errors"]))
            self.assertIn(path, prompts[1])
        self.assertIn("suggest_next[3].action", prompts[1])
        self.assertTrue(any(bad_reports[0] in error and "具体文件" in error for error in first_error["errors"]))
        self.assertIn(bad_reports[2], second_error["error"])
        self.assertNotIn("readonly_files", second_error["error"])
        self.assertNotIn("suggest_next[3].action", second_error["error"])
        self.assertNotIn(bad_reports[1], second_error["error"])
        self.assertIn(bad_reports[2], prompts[2])
        self.assertIn(str(outputs[1].parent / "validation_error.json"), prompts[2])
        self.assertEqual([line for line in prompts[2].splitlines() if line.startswith("Stage9 输出文件：")],
                         [f"Stage9 输出文件：{outputs[2]}"])
        for output, request in zip(outputs, requests):
            self.assert_pattern_contract(output, "proven_pattern")
            self.assertEqual(self.read_json(output.parent / "history_before.json"), before)
            for field in ("iteration", "scene", "performance_case_ids", "known_case_ids", "human_message_ids"):
                self.assertEqual(request[field], requests[0][field])
        self.assertTrue(all(not (output.parent / "commit.json").exists() for output in outputs[:2]))
        self.assertTrue((outputs[2].parent / "commit.json").is_file())
        self.assertEqual(len(list(outputs[0].parent.parent.rglob("commit.json"))), 1)
        history = load_history(str(self.work))
        self.assertEqual(history["rounds"], before["rounds"])
        self.assertEqual(history["ledger"][0], before["ledger"][0])
        self.assertEqual([entry["iter"] for entry in history["ledger"]], [1, 2])
        self.assertEqual(Path(history["ledger"][-1]["stage9_decision_path"]), outputs[2])
        self.assertEqual([task["task_id"] for task in history["suggest_next"]],
                         ["T_CONFLICT_C2", "T_CONFLICT_C3", "T_FILES", "T_SYNTAX"])
        self.assertEqual(self.fixture.read_state()["iteration"], 2)
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage9"], [1, 2, 2, 2])
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1, 2])
        knowledge = (self.work / "knowledge/proven_patterns.md").read_text(encoding="utf-8")
        self.assertEqual(knowledge.count("## iter2:"), 1)
        self.assertIn(str(outputs[2]), knowledge)
        for name in ("workflow.log", "state_transitions.log"):
            logs = (self.work / "log" / name).read_text(encoding="utf-8")
            for index, request in enumerate(requests):
                self.assertTrue(any(request["request_id"] in line and f"尝试次数={index + 1}/3" in line
                                    and f"修正次数={index}/2" in line for line in logs.splitlines()))
            self.assertIn("下一步=同轮修正1/2次", logs)
            self.assertIn("下一步=同轮修正2/2次", logs)
            self.assertNotIn("===== Stage9 修正失败，停止交付 =====", logs)

    def test_all_pattern_errors_reach_one_correction_before_single_experience_commit(self):
        self.routing.perfs = {1: (1.6, 1.8), 2: (1.8, 2.0)}
        case_ids = ("synthetic_op_1", "synthetic_op_2")
        requests = []
        first_round = []
        before = {}

        def agent(output, payload, iteration, prompt):
            if iteration == 1:
                first_round.append(output)
                self.write(output, payload)
                return True
            requests.append((output, prompt))
            pattern = payload["proven_pattern"]
            pattern["case_analysis"] = [
                {"case_id": case_id, "observation": "Measured latency fell after the tile change"}
                for case_id in case_ids]
            if len(requests) == 1:
                before.update(deepcopy(load_history(str(self.work))))
                pattern["applicability"] = {"shapes": "The two measured same-shape cases"}
            else:
                self.assertEqual(load_history(str(self.work)), before)
                for analysis in pattern["case_analysis"]:
                    analysis["explanation"] = "The measured tile change removes redundant launches"
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=2, file_logs=True)
        self.assertEqual(len(requests), 2)
        first_path, _ = requests[0]
        retry_path, retry_prompt = requests[1]
        self.assertEqual(retry_path.parent, first_path.parent / "retry1")
        error = self.read_json(first_path.parent / "validation_error.json")
        paths = ["proven_pattern.applicability"] + [
            f"proven_pattern.case_analysis[{index}].explanation" for index in range(len(case_ids))]
        for token in paths + list(case_ids):
            self.assertIn(token, error["error"])
            self.assertIn(token, retry_prompt)
        for output, _ in requests:
            self.assertEqual(self.read_json(output.parent / "request.json")["scene"], "all_passed")
            self.assert_pattern_contract(output, "proven_pattern")
            self.assertEqual(self.read_json(output.parent / "history_before.json"), before)
        self.assert_no_required_patterns(first_round[0])
        self.assertFalse((first_path.parent / "commit.json").exists())
        self.assertTrue((retry_path.parent / "commit.json").is_file())
        self.assertEqual(len(list(first_path.parent.parent.rglob("commit.json"))), 1)
        self.assertFalse((retry_path.parent / "validation_error.json").exists())
        history = load_history(str(self.work))
        self.assertEqual(history["rounds"], before["rounds"])
        self.assertEqual(history["ledger"][0], before["ledger"][0])
        self.assertEqual([entry["iter"] for entry in history["ledger"]], [1, 2])
        self.assertEqual(Path(history["ledger"][-1]["stage9_decision_path"]), retry_path)
        self.assertEqual(self.fixture.read_state()["iteration"], 2)
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage9"], [1, 2, 2])
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1, 2])
        knowledge = (self.work / "knowledge/proven_patterns.md").read_text(encoding="utf-8")
        self.assertEqual(knowledge.count("## iter2:"), 1)
        self.assertIn(str(retry_path), knowledge)
        for case_id in case_ids:
            self.assertIn(case_id, knowledge)

    def test_uncorrected_pattern_explanations_stop_with_specific_errors_and_no_commit(self):
        self.routing.perfs = {1: (1.6, 1.8), 2: (1.8, 2.0)}
        case_ids = ("synthetic_op_1", "synthetic_op_2")
        requests = []
        before = {}

        def agent(output, payload, iteration, prompt):
            if iteration == 1:
                self.write(output, payload)
                return True
            requests.append((output, prompt))
            pattern = payload["proven_pattern"]
            pattern["case_analysis"] = [
                {"case_id": case_id, "observation": "Measured latency fell after the tile change"}
                for case_id in case_ids]
            if len(requests) == 1:
                before.update(deepcopy(load_history(str(self.work))))
                pattern["applicability"] = {"shapes": "The two measured same-shape cases"}
            else:
                self.assertEqual(load_history(str(self.work)), before)
                # Fix the top-level type, but still omit both case explanations.
                self.assertIsInstance(pattern["applicability"], str)
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        with self.assertRaises(RuntimeError) as raised:
            self.routing.run_workflow(max_iterations=2, file_logs=True)
        self.assertEqual(len(requests), 3)
        self.assert_attempt_chain([output for output, _ in requests])
        first_path, _ = requests[0]
        retry_path, _ = requests[-1]
        retry_prompt = requests[1][1]
        first_error = self.read_json(first_path.parent / "validation_error.json")
        retry_error = self.read_json(retry_path.parent / "validation_error.json")
        self.assertIn("proven_pattern.applicability", first_error["error"])
        self.assertIn("proven_pattern.applicability", retry_prompt)
        self.assertNotIn("proven_pattern.applicability", retry_error["error"])
        self.assertIs(first_error["will_retry"], True)
        self.assertIs(retry_error["will_retry"], False)
        for filename in ("workflow.log", "state_transitions.log"):
            logs = (self.work / "log" / filename).read_text(encoding="utf-8")
            self.assertIn(str(retry_path.parent / "validation_error.json"), logs)
            self.assertIn("Stage9 修正失败", logs)
            self.assertIn("不进入 Stage3", logs)
            for index, case_id in enumerate(case_ids):
                for token in (f"proven_pattern.case_analysis[{index}].explanation", case_id):
                    self.assertIn(token, first_error["error"])
                    self.assertIn(token, retry_prompt)
                    self.assertIn(token, retry_error["error"])
                    self.assertIn(token, str(raised.exception))
                    self.assertIn(token, logs)
        for output, _ in requests:
            self.assert_pattern_contract(output, "proven_pattern")
            self.assertEqual(self.read_json(output.parent / "history_before.json"), before)
            self.assertFalse((output.parent / "commit.json").exists())
        self.assertFalse((self.work / "knowledge/proven_patterns.md").exists())
        self.assertEqual(load_history(str(self.work)), before)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter2_stage9")
        self.assertEqual(self.fixture.read_state()["iteration"], 2)
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage9"], [1, 2, 2, 2])
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1])
        self.assertNotIn("stage10", self.fixture.events)

    def test_conflict_is_corrected_before_stage3_with_one_history_and_experience_commit(self):
        requests = []
        before = {}

        def agent(output, payload, iteration, prompt):
            if iteration == 1:
                self.write(output, payload)
                return True
            requests.append((output, payload["request_id"], prompt))
            if len(requests) == 1:
                before.update(deepcopy(load_history(str(self.work))))
                self.conflicting_plan(payload)
                tampered = deepcopy(before)
                tampered["rounds"] = []
                tampered["ledger"][0]["direction"] = "UNAUTHORIZED_DIRECT_EDIT"
                save_history(str(self.work), tampered)
            else:
                self.assertEqual(load_history(str(self.work)), before)
                self.corrected_plan(payload)
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=2, file_logs=True)
        self.assertEqual(len(requests), 2)
        first_path, first_id, _ = requests[0]
        retry_path, retry_id, retry_prompt = requests[1]
        self.assertEqual(retry_path.parent, first_path.parent / "retry1")
        self.assertNotEqual(first_id, retry_id)
        error_path = first_path.parent / "validation_error.json"
        error = self.read_json(error_path)
        self.assertEqual(error["iteration"], 2)
        self.assertEqual(error["request_id"], first_id)
        self.assertEqual(Path(error["decision_path"]), first_path)
        self.assertIn(C3, error["error"])
        self.assertIn(str(error_path), retry_prompt)
        self.assertIn(str(first_path), retry_prompt)
        self.assertIn("FUSION_HANDOFF[stage9]", retry_prompt)
        output_hint = next(line for line in retry_prompt.splitlines()
                           if "用途：本次 Stage9 决策输出（待你生成）" in line)
        retry_relative = retry_path.relative_to(self.work).as_posix()
        self.assertIn(f"文件：`{retry_path}`", output_hint)
        self.assertIn(f"相对工作目录：`{retry_relative}`", output_hint)
        self.assertIn(f"迭代模板：`{retry_relative.replace('/iter2/', '/<iter>/')}`", output_hint)
        output_contract = [line for line in retry_prompt.splitlines()
                           if line.startswith("Stage9 输出文件：")]
        self.assertEqual(output_contract, [f"Stage9 输出文件：{retry_path}"])
        previous_relative = first_path.relative_to(self.work).as_posix()
        previous_template = previous_relative.replace("/iter2/", "/<iter>/")
        old_output_lines = [line for line in retry_prompt.splitlines()
                            if any(path in line for path in
                                   (str(first_path), previous_relative, previous_template))]
        self.assertEqual(len(old_output_lines), 1)
        self.assertIn("用途：被拒绝的原始决策", old_output_lines[0])
        first_request = self.read_json(first_path.parent / "request.json")
        retry_request = self.read_json(retry_path.parent / "request.json")
        for key in ("iteration", "scene", "phase", "fail_reason", "performance_case_ids",
                    "human_message_ids", "human_bundle_path", "known_case_ids", "plan_version"):
            self.assertEqual(retry_request[key], first_request[key])
        for request, request_path in ((first_request, first_path), (retry_request, retry_path)):
            for key, filename in (("schema_path", "decision_schema.json"),
                                  ("template_path", "decision_template.json"),
                                  ("case_catalog_path", "case_catalog.json")):
                artifact = Path(request[key])
                self.assertEqual(artifact, request_path.parent / filename)
                self.assertTrue(artifact.is_file())
            template = self.read_json(Path(request["template_path"]))
            self.assertEqual(template["request_id"], request["request_id"])
            self.assertEqual(template["iteration"], 2)
            self.assertEqual(template["plan_version"], 2)
        self.assertIn(retry_request["template_path"], retry_prompt)
        self.assertNotIn(first_request["template_path"], retry_prompt)
        self.assertEqual(self.read_json(retry_path.parent / "history_before.json"), before)
        retry_role = Path(retry_request["role_path"])
        self.assertEqual(retry_role.parent, retry_path.parent)
        self.assertEqual(retry_role.read_text(encoding="utf-8"),
                         Path(first_request["role_path"]).read_text(encoding="utf-8"))
        self.assertFalse((first_path.parent / "commit.json").exists())
        self.assertTrue((retry_path.parent / "commit.json").is_file())
        history = load_history(str(self.work))
        self.assertEqual(history["rounds"], before["rounds"])
        self.assertEqual(history["ledger"][0], before["ledger"][0])
        self.assertEqual([entry["iter"] for entry in history["ledger"]], [1, 2])
        self.assertEqual(Path(history["ledger"][-1]["stage9_decision_path"]), retry_path)
        self.assertEqual(history["ledger"][-1]["modify_files"], [C3])
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1, 2])
        self.assertIn(C3, self.fixture.prompts["stage3"])
        knowledge = (self.work / "knowledge/proven_patterns.md").read_text(encoding="utf-8")
        self.assertEqual(knowledge.count("## iter2:"), 1)
        self.assertIn(str(retry_path), knowledge)
        logs = (self.work / "log/workflow.log").read_text(encoding="utf-8")
        rejected = next(line for line in logs.splitlines()
                        if "===== Stage9 决策校验失败" in line)
        correction = next(line for line in logs.splitlines()
                          if "===== Stage9 同轮修正 =====" in line)
        accepted = next(line for line in logs.splitlines()
                        if "[Stage9 决策校验通过]" in line and retry_id in line)
        for token in ("iter=2", f"request={first_id}", f"scene={first_request['scene']}", str(first_path),
                      str(error_path), "下一步=同轮修正1/2次"):
            self.assertIn(token, rejected)
        for token in ("iter=2", first_id, retry_id, str(error_path), str(first_path),
                      str(retry_path.parent / "prompt.md"), str(retry_path), "不增加性能迭代次数"):
            self.assertIn(token, correction)
        for token in ("iter=2", f"request={retry_id}", "尝试次数=2/3", "修正次数=1/2", str(retry_path)):
            self.assertIn(token, accepted)
        self.assertLess(logs.index(rejected), logs.index(correction))
        self.assertLess(logs.index(correction), logs.index(accepted))
        self.assertNotIn("===== Stage9 修正失败，停止交付 =====", logs)
        state_logs = (self.work / "log/state_transitions.log").read_text(encoding="utf-8")
        self.assertIn(correction, state_logs)
        self.assertTrue(any("[Stage9 决策校验通过]" in line and f"request={retry_id}" in line
                            and "iter=2" in line and "修正次数=1/2" in line
                            for line in state_logs.splitlines()))

    def test_third_conflict_stops_without_publishing_advice_or_experience(self):
        attempts = []
        before = {}

        def agent(output, payload, iteration, _prompt):
            if iteration == 1:
                self.write(output, payload)
                return True
            if not attempts:
                before.update(deepcopy(load_history(str(self.work))))
            else:
                self.assertEqual(load_history(str(self.work)), before)
            attempts.append(output)
            self.conflicting_plan(payload)
            save_history(str(self.work), {"rounds": [], "ledger": [], "suggest_next": []})
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        with self.assertRaises(RuntimeError):
            self.routing.run_workflow(max_iterations=2, file_logs=True)
        self.assertEqual(len(attempts), 3)
        requests = self.assert_attempt_chain(attempts)
        self.assertEqual(load_history(str(self.work)), before)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter2_stage9")
        self.assertEqual(self.fixture.read_state()["iteration"], 2)
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage9"], [1, 2, 2, 2])
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1])
        self.assertNotIn("stage10", self.fixture.events)
        self.assertFalse((self.work / "knowledge/proven_patterns.md").exists())
        for index, output in enumerate(attempts):
            error = self.read_json(output.parent / "validation_error.json")
            self.assertEqual(error["will_retry"], index < 2)
            self.assertFalse((output.parent / "commit.json").exists())
        retry_path = attempts[-1]
        retry_id = requests[-1]["request_id"]
        error_path = retry_path.parent / "validation_error.json"
        error = self.read_json(error_path)
        logs = (self.work / "log/workflow.log").read_text(encoding="utf-8")
        rejected = [line for line in logs.splitlines()
                    if "===== Stage9 决策校验失败" in line]
        self.assertEqual(len(rejected), 3)
        corrections = [line for line in logs.splitlines() if "===== Stage9 同轮修正 =====" in line]
        self.assertEqual(len(corrections), 2)
        for index, (output, request, line) in enumerate(zip(attempts, requests, rejected)):
            next_step = f"下一步=同轮修正{index + 1}/2次" if index < 2 else "停止于 Stage9"
            for token in ("iter=2", f"request={request['request_id']}", f"scene={request['scene']}",
                          str(output.parent / "validation_error.json"), f"尝试次数={index + 1}/3",
                          f"修正次数={index}/2", next_step):
                self.assertIn(token, line)
            if index == 2:
                continue
            following = attempts[index + 1]
            correction = corrections[index]
            for token in ("iter=2", request["request_id"], requests[index + 1]["request_id"], str(output),
                          str(output.parent / "validation_error.json"), str(following.parent / "prompt.md"),
                          str(following), "不增加性能迭代次数"):
                self.assertIn(token, correction)
            self.assertLess(logs.index(line), logs.index(correction))
            self.assertLess(logs.index(correction), logs.index(rejected[index + 1]))
        stopped = next(line for line in logs.splitlines()
                       if "===== Stage9 修正失败，停止交付 =====" in line)
        for token in ("iter=2", f"request={retry_id}", str(retry_path), str(error_path),
                      error["error"], "不进入 Stage3"):
            self.assertIn(token, stopped)
        self.assertLess(logs.index(rejected[-1]), logs.index(stopped))
        self.assertFalse(any("[Stage9 决策校验通过]" in line and retry_id in line
                             for line in logs.splitlines()))
        state_logs = (self.work / "log/state_transitions.log").read_text(encoding="utf-8")
        for correction in corrections:
            self.assertIn(correction, state_logs)
        self.assertIn(stopped, state_logs)

    def test_corrected_regression_review_writes_one_detailed_regression_record(self):
        self.routing.perfs = {1: (2.0,), 2: (1.9,)}
        outputs = []

        def agent(output, payload, iteration, _prompt):
            if iteration == 2:
                outputs.append(output)
                if len(outputs) == 1:
                    self.conflicting_plan(payload)
                else:
                    self.corrected_plan(payload)
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=2)
        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assert_pattern_contract(output, "regression_pattern")
        report = (self.work / "knowledge/regression_patterns.md").read_text(encoding="utf-8")
        self.assertEqual(report.count("## iter2:"), 1)
        self.assertIn("More padding overhead", report)
        self.assertIn(str(outputs[-1]), report)
        history = load_history(str(self.work))
        self.assertEqual([item["iter"] for item in history["rounds"]], [1, 2])
        self.assertEqual([item["iter"] for item in history["ledger"]], [1, 2])
        self.assertEqual(self.fixture.events.count("stage6"), 2)

    def check_invalid_first_output_gets_fresh_retry(self, invalid_first):
        outputs = []

        def agent(output, payload, _iteration, _prompt):
            outputs.append(output)
            if len(outputs) == 1:
                invalid_first(output, payload)
            else:
                self.corrected_plan(payload)
                self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=1)
        self.assertEqual(len(outputs), 2)
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertTrue((outputs[0].parent / "validation_error.json").is_file())
        self.assertEqual(Path(load_history(str(self.work))["ledger"][-1]["stage9_decision_path"]), outputs[1])
        self.assertEqual(self.fixture.events.count("stage6"), 1)
        self.assertEqual(self.fixture.events.count("stage3"), 1)

    def test_missing_output_retries_without_reusing_older_decision(self):
        self.check_invalid_first_output_gets_fresh_retry(lambda _output, _payload: None)

    def test_malformed_json_retries_before_publishing_history(self):
        self.check_invalid_first_output_gets_fresh_retry(
            lambda output, _payload: output.write_text('{"iteration": 1,', encoding="utf-8"))

    def test_stale_request_binding_retries_with_a_new_request_id(self):
        def stale(output, payload):
            payload["request_id"] = "00000000-0000-0000-0000-000000000000"
            self.write(output, payload)
        self.check_invalid_first_output_gets_fresh_retry(stale)

    def check_transport_failure_is_not_retried(self, failure):
        outputs = []
        before = {}

        def agent(output, payload, _iteration, _prompt):
            outputs.append(output)
            before.update(deepcopy(load_history(str(self.work))))
            self.write(output, payload)
            save_history(str(self.work), {"ledger": [], "rounds": []})
            if isinstance(failure, Exception):
                raise failure
            return failure

        self.routing.stage9_callback = agent
        with self.assertRaises(RuntimeError):
            self.routing.run_workflow(max_iterations=1)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(self.attempts_for(1)), 1)
        self.assertEqual(load_history(str(self.work)), before)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter1_stage9")
        self.assertNotIn("stage3", self.fixture.events)
        self.assertFalse((outputs[0].parent / "commit.json").exists())

    def test_agent_false_is_a_transport_failure_even_when_output_is_valid(self):
        self.check_transport_failure_is_not_retried(False)

    def test_agent_value_error_is_not_mistaken_for_output_validation_failure(self):
        self.check_transport_failure_is_not_retried(ValueError("synthetic transport failed"))

    def check_retry2_resume(self, *, interrupt_transport):
        outputs = []
        before = {}

        def interrupted(output, payload, iteration, _prompt):
            if iteration == 1:
                self.write(output, payload)
                return True
            outputs.append(output)
            if len(outputs) == 1:
                before.update(deepcopy(load_history(str(self.work))))
            else:
                self.assertEqual(load_history(str(self.work)), before)
            # A complete but invalid retry2 file must not be accepted on process restart.
            self.conflicting_plan(payload)
            self.write(output, payload)
            if len(outputs) == 3 and interrupt_transport:
                raise RuntimeError("synthetic interruption during correction")
            return True

        self.routing.stage9_callback = interrupted
        with self.assertRaises(RuntimeError) as raised:
            self.routing.run_workflow(max_iterations=2)
        if interrupt_transport:
            self.assertIn("synthetic interruption during correction", str(raised.exception))
        else:
            self.assertIn("readonly_files", str(raised.exception))
        self.assertEqual(len(outputs), 3)
        failed_requests = self.assert_attempt_chain(outputs)
        self.assertEqual(load_history(str(self.work)), before)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter2_stage9")
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1])
        self.assertFalse((self.work / "knowledge/proven_patterns.md").exists())
        if not interrupt_transport:
            error = self.read_json(outputs[-1].parent / "validation_error.json")
            self.assertIs(error["will_retry"], False)
        previous = list(outputs)

        def resumed(output, payload, iteration, _prompt):
            self.assertEqual(iteration, 2)
            self.assertNotIn(output, previous)
            outputs.append(output)
            self.corrected_plan(payload)
            self.write(output, payload)
            return True

        self.routing.stage9_callback = resumed
        self.routing.run_workflow()
        self.assertEqual(len(outputs), 4)
        fresh_request = self.read_json(outputs[-1].parent / "request.json")
        self.assertEqual(fresh_request["correction_attempt"], 0)
        self.assertNotIn(fresh_request["request_id"], [request["request_id"] for request in failed_requests])
        self.assertEqual(outputs[-1].parent.parent, outputs[0].parent.parent)
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        self.assertEqual(self.fixture.read_state()["iteration"], 2)
        history = load_history(str(self.work))
        self.assertEqual(history["rounds"], before["rounds"])
        self.assertEqual(history["ledger"][0], before["ledger"][0])
        self.assertEqual([entry["iter"] for entry in history["ledger"]], [1, 2])
        self.assertEqual(Path(history["ledger"][-1]["stage9_decision_path"]), outputs[-1])
        self.assertEqual([i for i, stage, _ in self.routing.seen if stage == "stage3"], [1, 2])
        self.assertTrue(all(not (output.parent / "commit.json").exists() for output in previous))
        self.assertEqual(len(list(outputs[0].parent.parent.rglob("commit.json"))), 1)
        knowledge = (self.work / "knowledge/proven_patterns.md").read_text(encoding="utf-8")
        self.assertEqual(knowledge.count("## iter2:"), 1)
        self.assertIn(str(outputs[-1]), knowledge)

    def test_interrupted_retry2_resumes_fresh_without_repeating_evaluation_or_experience(self):
        self.check_retry2_resume(interrupt_transport=True)

    def test_exhausted_retry2_resumes_fresh_without_repeating_evaluation_or_experience(self):
        self.check_retry2_resume(interrupt_transport=False)

    def test_stage3_resume_rechecks_legacy_advice_through_same_iteration_stage9(self):
        original_agent = self.fixture.agent

        def interrupted(agent, role, work, prompt, **kwargs):
            if Path(role).name == "n1_stage3_fix_and_optimize.md":
                raise RuntimeError("synthetic Stage3 interruption")
            return original_agent(agent, role, work, prompt, **kwargs)

        self.fixture.agent = interrupted
        with self.assertRaisesRegex(RuntimeError, "synthetic Stage3 interruption"):
            self.routing.run_workflow(max_iterations=1)
        state = self.fixture.read_state()
        self.assertEqual(state["current_stage"], "iter1_stage3")
        self.assertEqual(state["stage9_context"]["phase"], "developing")
        history = load_history(str(self.work))
        original_decision = history["ledger"][-1]["stage9_decision_path"]
        rounds = deepcopy(history["rounds"])
        history["ledger"][-1].pop("plan_version", None)
        history["ledger"][-1].pop("action_plan", None)
        save_history(str(self.work), history)
        seen_before = len(self.routing.seen)
        self.fixture.agent = original_agent
        self.routing.run_workflow(file_logs=True)
        resumed = [(iteration, stage) for iteration, stage, _ in self.routing.seen[seen_before:]]
        self.assertEqual(resumed, [(1, "stage9"), (1, "stage3"), (1, "stage10")])
        self.assertEqual(self.fixture.events.count("stage6"), 1)
        self.assertEqual(self.fixture.read_state()["iteration"], 1)
        after = load_history(str(self.work))
        self.assertEqual(after["rounds"], rounds)
        self.assertEqual(len(after["ledger"]), 1)
        self.assertNotEqual(after["ledger"][0]["stage9_decision_path"], original_decision)
        self.assertTrue(all("inspect_files" in item and "modify_files" in item
                            for item in after["suggest_next"]))
        logs = (self.work / "log/workflow.log").read_text(encoding="utf-8")
        self.assertIn("[Stage3 交付拦截]", logs)

    def local_scope(self, modify_files, readonly_files=None, inspect_files=None):
        return {
            "ledger_entry": {"modify_files": modify_files, "readonly_files": readonly_files or []},
            "suggest_next": [{"modify_files": modify_files, "inspect_files": inspect_files or []}],
        }

    def test_real_directory_cannot_replace_a_concrete_file_target(self):
        decision = self.local_scope(["impl/c3"])
        with self.assertRaisesRegex(ValueError, "目录"):
            orchestrator._validate_stage9_file_targets(str(self.work), decision)

    def test_new_file_inside_work_is_allowed_without_creating_it(self):
        new_file = "impl/new_variant/kernel.py"
        decision = self.local_scope([new_file], readonly_files=[C2])
        orchestrator._validate_stage9_file_targets(str(self.work), decision)
        self.assertFalse((self.work / new_file).exists())

    def test_resolved_alias_cannot_make_a_readonly_file_writable(self):
        root = self.work.resolve()
        alias = root / "impl/alias_of_c3.py"
        target = root / C3
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            actual = original_resolve(path, *args, **kwargs)
            return target if actual == alias else actual

        decision = self.local_scope([C3], readonly_files=["impl/alias_of_c3.py"])
        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "同一文件"):
                orchestrator._validate_stage9_file_targets(str(self.work), decision)
        self.assertFalse(alias.exists(), "No real symlink or external file is needed")

    def test_resolved_modification_alias_cannot_escape_work_directory(self):
        root = self.work.resolve()
        alias = root / "impl/external_alias.py"
        outside = root.parent / "outside_target.py"
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            actual = original_resolve(path, *args, **kwargs)
            return outside if actual == alias else actual

        decision = self.local_scope(["impl/external_alias.py"])
        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "工作目录内"):
                orchestrator._validate_stage9_file_targets(str(self.work), decision)
        self.assertFalse(alias.exists(), "No real symlink or external file is needed")

    def test_readonly_linked_task_inputs_are_accepted_without_retry_in_full_workflow(self):
        root = self.work.resolve()
        task_root = root / "task"
        external_source = root.parent / f"synthetic_external_task_{root.name}"
        original_resolve = Path.resolve
        original_is_dir = Path.is_dir
        golden = task_root / "golden.py"
        golden.write_text("# read-only reference\n", encoding="utf-8")
        (task_root / "cases.yaml").write_text("cases: []\n", encoding="utf-8")

        def resolve(path, *args, **kwargs):
            actual = original_resolve(path, *args, **kwargs)
            if actual.is_relative_to(task_root):
                return external_source / actual.relative_to(task_root)
            return actual

        def is_dir(path):
            # Model an existing link target as well as its resolved path; startup
            # now verifies the task directory before dispatching any agents.
            return True if path == external_source else original_is_dir(path)

        outputs = []

        def agent(output, payload, _iteration, _prompt):
            outputs.append(output)
            payload["ledger_entry"]["readonly_files"] = ["task/golden.py"]
            payload["suggest_next"][0]["changes"].extend([
                {"file": path, "operation": "inspect", "location": "test input definition",
                 "method": "Read the immutable test contract before changing the implementation"}
                for path in ("task/cases.yaml", "task/golden.py")])
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        with patch.object(Path, "resolve", resolve), patch.object(Path, "is_dir", is_dir):
            # Check both the low-level boundary and the normal Stage9/Stage3 route.
            orchestrator._validate_stage9_file_targets(
                str(self.work), self.local_scope([C3], readonly_files=["task/golden.py"],
                                                 inspect_files=["task/cases.yaml"]))
            self.routing.run_workflow(max_iterations=1)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(self.fixture.events.count("stage3"), 1)
        self.assertEqual(self.fixture.events.count("stage6"), 1)
        self.assertFalse((outputs[0].parent / "validation_error.json").exists())
        self.assertEqual(golden.read_text(encoding="utf-8"), "# read-only reference\n")
        self.assertFalse(external_source.exists(), "The external task link is mocked, never created")

    def test_readonly_file_can_be_inspected_without_an_unnecessary_retry(self):
        outputs = []

        def agent(output, payload, _iteration, _prompt):
            outputs.append(output)
            self.conflicting_plan(payload)
            suggestion = payload["suggest_next"][0]
            suggestion.update(action="Inspect the traversal and report whether load imbalance is plausible",
                              task_type="inspect")
            suggestion["changes"] = [{"file": C3, "operation": "inspect", "location": "tile traversal",
                                       "method": "Read the traversal and report evidence without changing code"}]
            self.write(output, payload)
            return True

        self.routing.stage9_callback = agent
        self.routing.run_workflow(max_iterations=1)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(self.fixture.events.count("stage3"), 1)
        self.assertFalse((outputs[0].parent / "validation_error.json").exists())

    def test_correction_keeps_human_direction_and_scene_context(self):
        flow = human_routing.HumanRoutingTests(methodName="runTest")
        flow.setUp()
        self.addCleanup(flow.doCleanups)
        message = flow.submit("Keep the fusion direction and tune the existing implementation")
        seen = []

        def amend(output, payload, _iteration, prompt):
            request = json.loads((output.parent / "request.json").read_text(encoding="utf-8"))
            seen.append((output, request, prompt))
            if len(seen) == 1:
                payload["ledger_entry"]["readonly_files"] = ["impl/synthetic_op.py"]

        flow.on_decision = amend
        flow.run_flow(1)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][1]["human_message_ids"], [message["id"]])
        self.assertEqual(seen[1][1]["human_message_ids"], [message["id"]])
        for _, request, prompt in seen:
            self.assertEqual(request["scene"], "all_passed")
            self.assertEqual(request["phase"], "feedback")
            self.assertIn(message["id"], prompt)
            self.assertIn(message["text"], prompt)
        history = load_history(str(flow.work))
        p0 = [item for item in history["suggest_next"] if item.get("source") == "human"]
        self.assertEqual(len(p0), 1)
        self.assertEqual(p0[0]["priority"], "P0")
        self.assertEqual(p0[0]["human_message_id"], message["id"])
        self.assertEqual(flow.human.all_messages()[0]["status"], "executed")
        self.assertIn(message["id"], flow.fixture.prompts["stage3"])
        self.assertEqual(len(history["rounds"]), 1)


if __name__ == "__main__":
    unittest.main()
