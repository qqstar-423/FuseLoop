"""Program-owned Stage9 task schema and resumable Stage3 handoff tests."""

from copy import deepcopy
import unittest

from lib.stage9_plan import (build_decision_schema, build_decision_template,
                             normalize_decision_plan, validate_current_plan, _validate_schema)


def decision():
    return {
        "plan_version": 2, "iteration": 3, "request_id": "request-3",
        "ledger_entry": {
            "evaluation_summary": "case7 remains slow; the report is the evidence.",
            "direction": "Adjust the tile traversal of the bf16 implementation.",
            "readonly_files": ["impl/c1.py", "impl/dispatcher.py"],
        },
        "suggest_next": [{
            "task_id": "T1", "priority": "P1", "task_type": "modify",
            "action": "Improve case7 tile traversal", "reason": "Uneven block occupancy.",
            "case_scope": "cases", "target_cases": ["case7"], "operator_reason": "",
            "case_bindings": [{
                "case_id": "case7", "implementation_files": ["impl/c2.py"],
                "route_evidence": [{"file": "impl/dispatcher.py", "location": "dispatch, bf16 branch",
                                    "explanation": "case7 dtype is bf16; this branch imports c2."}],
            }],
            "changes": [
                {"file": "impl/dispatcher.py", "operation": "inspect", "location": "dispatch",
                 "method": "Confirm case7 uses c2 before adjusting the implementation."},
                {"file": "impl/c2.py", "operation": "modify", "location": "kernel tile loop",
                 "method": "Traverse adjacent output tiles on neighboring cores."},
            ],
            "acceptance_checks": ["Run case7 correctness and compare measured speedup to iter3."],
        }],
    }


class Stage9PlanTests(unittest.TestCase):
    def check_rejected(self, alter, text=None):
        value = decision()
        alter(value)
        if text is None:
            with self.assertRaises(ValueError):
                normalize_decision_plan(value, known_case_ids=["case7", "case8"], required=True)
        else:
            with self.assertRaisesRegex(ValueError, text):
                normalize_decision_plan(value, known_case_ids=["case7", "case8"], required=True)

    def test_permissions_derived_from_one_task_list_without_mutating_input(self):
        raw = decision()
        original = deepcopy(raw)
        result = normalize_decision_plan(raw, known_case_ids=["case7"], required=True)
        self.assertEqual(raw, original)
        self.assertEqual(result["ledger_entry"]["modify_files"], ["impl/c2.py"])
        self.assertEqual(result["suggest_next"][0]["modify_files"], ["impl/c2.py"])
        self.assertEqual(result["suggest_next"][0]["inspect_files"], ["impl/dispatcher.py"])
        self.assertEqual(result["ledger_entry"]["action_plan"],
                         {"version": 2, "tasks": raw["suggest_next"]})
        result["suggest_next"][0]["changes"][1]["method"] = "changed"
        self.assertNotEqual(result["ledger_entry"]["action_plan"]["tasks"][0]["changes"][1]["method"],
                            "changed")

    def test_raw_model_cannot_author_derived_or_duplicate_plan_fields(self):
        for field in ("modify_files", "fix_plan", "action_plan"):
            with self.subTest(field=field):
                self.check_rejected(lambda item: item["ledger_entry"].update({field: []}), "unsupported")
        for field in ("inspect_files", "modify_files", "readonly_files"):
            with self.subTest(field=field):
                self.check_rejected(lambda item: item["suggest_next"][0].update({field: []}), "unsupported")

    def test_unknown_fields_rejected_at_every_action_plan_level(self):
        locations = [lambda item: item, lambda item: item["ledger_entry"],
                     lambda item: item["suggest_next"][0],
                     lambda item: item["suggest_next"][0]["changes"][0],
                     lambda item: item["suggest_next"][0]["case_bindings"][0],
                     lambda item: item["suggest_next"][0]["case_bindings"][0]["route_evidence"][0]]
        for locate in locations:
            self.check_rejected(lambda item: locate(item).update({"surprise": True}), "unsupported")

    def test_task_required_fields_cannot_be_omitted(self):
        for field in build_decision_schema()["properties"]["suggest_next"]["items"]["required"]:
            with self.subTest(field=field):
                self.check_rejected(lambda item: item["suggest_next"][0].pop(field), "missing")

    def test_invalid_nested_types_and_empty_details_rejected(self):
        for field in ("action", "reason", "task_id"):
            for value in (None, 42, "", "  \n"):
                with self.subTest(field=field, value=value):
                    self.check_rejected(lambda item: item["suggest_next"][0].update({field: value}))
        for field in ("location", "method"):
            self.check_rejected(lambda item: item["suggest_next"][0]["changes"][1].update({field: ""}))
        self.check_rejected(lambda item: item["suggest_next"][0].update(acceptance_checks=[]))
        self.check_rejected(lambda item: item["suggest_next"][0].update(acceptance_checks=[" "]))
        self.check_rejected(lambda item: item["suggest_next"][0]["case_bindings"][0].update(route_evidence=[]))

    def test_plan_and_iteration_reject_booleans(self):
        for field in ("plan_version", "iteration"):
            self.check_rejected(lambda item: item.update({field: True}), "integer")

    def test_unsupported_version_rejected_even_in_legacy_mode(self):
        value = decision()
        value["plan_version"] = 1
        with self.assertRaisesRegex(ValueError, "must be 2"):
            normalize_decision_plan(value)

    def test_legacy_passthrough_is_copy_but_required_contract_rejects(self):
        legacy = {"ledger_entry": {"modify_files": ["impl/c2.py"]}, "suggest_next": []}
        result = normalize_decision_plan(legacy)
        result["ledger_entry"]["modify_files"].append("impl/new.py")
        self.assertEqual(legacy["ledger_entry"]["modify_files"], ["impl/c2.py"])
        with self.assertRaisesRegex(ValueError, "plan_version=2"):
            normalize_decision_plan(legacy, required=True)

    def test_duplicate_task_ids_rejected(self):
        self.check_rejected(lambda item: item["suggest_next"].append(deepcopy(item["suggest_next"][0])),
                            "task_id must be unique")

    def test_unknown_case_rejected_against_complete_authoritative_set(self):
        with self.assertRaisesRegex(ValueError, "unknown case IDs"):
            normalize_decision_plan(decision(), known_case_ids=["case8"])
        normalize_decision_plan(decision(), known_case_ids=None)

    def test_case_binding_must_cover_exact_case_set(self):
        self.check_rejected(lambda item: item["suggest_next"][0].update(case_bindings=[]), "exactly once")
        self.check_rejected(lambda item: item["suggest_next"][0]["case_bindings"][0].update(case_id="case8"),
                            "exactly once")
        self.check_rejected(lambda item: item["suggest_next"][0]["case_bindings"].append(
            deepcopy(item["suggest_next"][0]["case_bindings"][0])), "exactly once")
        self.check_rejected(lambda item: item["suggest_next"][0]["target_cases"].append("case7"), "duplicate")

    def test_case_implementation_cannot_be_missing_or_empty(self):
        self.check_rejected(lambda item: item["suggest_next"][0]["case_bindings"][0].update(
            implementation_files=[]), "at least 1")

    def test_case_cannot_be_optimized_by_writing_only_an_unbound_file(self):
        self.check_rejected(lambda item: item["suggest_next"][0]["changes"][1].update(file="impl/c3.py"),
                            "outside its case_bindings")

    def test_each_case_in_multi_case_task_must_receive_an_actual_operation(self):
        value = decision()
        task = value["suggest_next"][0]
        task["target_cases"].append("case8")
        second = deepcopy(task["case_bindings"][0])
        second.update(case_id="case8", implementation_files=["impl/c3.py"])
        task["case_bindings"].append(second)
        with self.assertRaisesRegex(ValueError, "case 'case8' has no modify operation"):
            normalize_decision_plan(value)
        # Shared implementations may legitimately cover multiple cases.
        second["implementation_files"] = ["impl/c2.py"]
        normalize_decision_plan(value)

    def test_inspection_task_cannot_hide_a_write(self):
        self.check_rejected(lambda item: item["suggest_next"][0].update(task_type="inspect"),
                            "inspect task cannot contain")

    def test_modify_task_must_have_a_write(self):
        self.check_rejected(lambda item: item["suggest_next"][0]["changes"][1].update(operation="inspect"),
                            "modify task requires")

    def test_inspection_of_protected_implementation_is_allowed(self):
        value = decision()
        task = value["suggest_next"][0]
        task["task_type"] = "inspect"
        task["changes"][1]["operation"] = "inspect"
        value["ledger_entry"]["readonly_files"].append("impl/c2.py")
        result = normalize_decision_plan(value)
        self.assertEqual(result["ledger_entry"]["modify_files"], [])
        self.assertEqual(result["suggest_next"][0]["inspect_files"],
                         ["impl/dispatcher.py", "impl/c2.py"])

    def test_inspection_must_touch_bound_implementation_not_just_dispatcher(self):
        def alter(item):
            item["suggest_next"][0]["task_type"] = "inspect"
            item["suggest_next"][0]["changes"].pop()
        self.check_rejected(alter, "no inspect operation")

    def test_readonly_blocks_modification_even_with_correct_binding(self):
        self.check_rejected(lambda item: item["ledger_entry"]["readonly_files"].append("impl/c2.py"),
                            "conflicts with.*readonly_files")

    def test_readonly_is_global_across_all_tasks(self):
        value = decision()
        next_task = deepcopy(value["suggest_next"][0])
        next_task["task_id"] = "T2"
        next_task["case_bindings"][0]["implementation_files"] = ["impl/c1.py"]
        next_task["changes"][1]["file"] = "impl/c1.py"
        value["suggest_next"].append(next_task)
        with self.assertRaisesRegex(ValueError, "T2.*readonly_files"):
            normalize_decision_plan(value)

    def test_readonly_prefix_cannot_be_bypassed(self):
        self.check_rejected(lambda item: item["ledger_entry"].update(readonly_files=["impl"]),
                            "conflicts with.*readonly_files")

    def test_operator_task_requires_global_reason_and_no_case_claim(self):
        value = decision()
        task = value["suggest_next"][0]
        task.update(case_scope="operator", target_cases=[], case_bindings=[],
                    operator_reason="Fix a shared compilation error before any case can run.")
        normalize_decision_plan(value)
        for alteration in ({"operator_reason": " "}, {"target_cases": ["case7"]},
                           {"case_bindings": decision()["suggest_next"][0]["case_bindings"]}):
            invalid = deepcopy(value)
            invalid["suggest_next"][0].update(alteration)
            with self.assertRaisesRegex(ValueError, "operator scope"):
                normalize_decision_plan(invalid)

    def test_case_scope_does_not_accept_global_task_escape_hatch(self):
        self.check_rejected(lambda item: item["suggest_next"][0].update(operator_reason="global"), "cases scope")
        self.check_rejected(lambda item: item["suggest_next"][0].update(target_cases=[]), "cases scope")

    def test_create_operation_derives_writable_target_and_needs_binding(self):
        value = decision()
        task = value["suggest_next"][0]
        task["case_bindings"][0]["implementation_files"].append("impl/new_tile.py")
        task["changes"].append({"file": "impl/new_tile.py", "operation": "create", "location": "new helper",
                                "method": "Introduce the tile traversal helper imported by c2."})
        result = normalize_decision_plan(value)
        self.assertEqual(result["ledger_entry"]["modify_files"], ["impl/c2.py", "impl/new_tile.py"])
        task["case_bindings"][0]["implementation_files"].remove("impl/new_tile.py")
        with self.assertRaisesRegex(ValueError, "outside its case_bindings"):
            normalize_decision_plan(value)

    def test_path_aliases_normalized_everywhere(self):
        value = decision()
        task = value["suggest_next"][0]
        task["changes"][1]["file"] = r".\impl\c2.py"
        task["case_bindings"][0]["implementation_files"] = ["./impl/c2.py"]
        task["case_bindings"][0]["route_evidence"][0]["file"] = r"impl\dispatcher.py"
        value["ledger_entry"]["readonly_files"] = ["./impl/c1.py"]
        result = normalize_decision_plan(value)
        self.assertEqual(result["ledger_entry"]["modify_files"], ["impl/c2.py"])
        self.assertEqual(result["ledger_entry"]["readonly_files"], ["impl/c1.py"])
        self.assertEqual(result["suggest_next"][0]["case_bindings"][0]["route_evidence"][0]["file"],
                         "impl/dispatcher.py")

    def test_unsafe_paths_rejected_in_all_file_positions(self):
        setters = [lambda item, path: item["ledger_entry"].update(readonly_files=[path]),
                   lambda item, path: item["suggest_next"][0]["changes"][1].update(file=path),
                   lambda item, path: item["suggest_next"][0]["case_bindings"][0].update(implementation_files=[path]),
                   lambda item, path: item["suggest_next"][0]["case_bindings"][0]["route_evidence"][0].update(file=path)]
        for path in ("../secret.py", "C:/other.py", "/tmp/file.py", "impl/", "impl/*.py",
                     "impl/a.py:stream", "impl/../a.py", " impl/a.py", ".", "impl/a\nb.py"):
            for setter in setters:
                with self.subTest(path=path):
                    self.check_rejected(lambda item: setter(item, path), "file|paths")

    def test_duplicate_normalized_file_aliases_rejected(self):
        self.check_rejected(lambda item: item["ledger_entry"].update(readonly_files=["impl/c1.py", "./impl/c1.py"]),
                            "duplicate file aliases")

    def test_invalid_enums_rejected(self):
        for field, value in (("task_type", "optimize"), ("priority", "P3"), ("case_scope", "unknown")):
            self.check_rejected(lambda item: item["suggest_next"][0].update({field: value}), "must be one of")
        self.check_rejected(lambda item: item["suggest_next"][0]["changes"][0].update(operation="write"),
                            "must be one of")

    def test_legacy_conclusion_and_experience_payloads_preserved_for_existing_validator(self):
        value = decision()
        value.update(insights=["Measured tile reuse improved."], bottleneck_now="Memory traffic",
                     worst_cases_tracker={"case7": {"custom": "legacy detail"}},
                     fusion_kernel_strategy=[{"iter": 3, "direction": "Current fusion"}],
                     proven_pattern={"what_changed": "tiling", "evidence": {"custom": "legacy detail"}},
                     regression_pattern={"what_changed": "other"}, pitfall={"topic": "memory"})
        value["suggest_next"][0].update(priority="P0")
        result = normalize_decision_plan(value)
        for field in value:
            if field not in {"ledger_entry", "suggest_next"}:
                self.assertEqual(result[field], value[field])
        self.assertEqual(result["suggest_next"][0]["priority"], "P0")

    def test_valid_current_handoff_is_verified_and_copied(self):
        normalized = normalize_decision_plan(decision())
        ledger = normalized["ledger_entry"]
        ledger.update(iter=3, verdict="improved", speedup_after=2.0)
        checked_ledger, checked_tasks = validate_current_plan(ledger, normalized["suggest_next"],
                                                              known_case_ids=["case7"])
        self.assertEqual(checked_ledger, ledger)
        self.assertEqual(checked_tasks, normalized["suggest_next"])
        checked_ledger["modify_files"].append("impl/new.py")
        self.assertNotIn("impl/new.py", ledger["modify_files"])

    def test_resume_rejects_mismatched_writable_list_or_suggestions(self):
        for field in ("ledger_scope", "suggestion_scope", "suggestion_text", "raw_plan"):
            with self.subTest(field=field):
                value = normalize_decision_plan(decision())
                ledger = value["ledger_entry"]
                tasks = value["suggest_next"]
                if field == "ledger_scope":
                    ledger["modify_files"] = ["impl/c3.py"]
                elif field == "suggestion_scope":
                    tasks[0]["modify_files"] = ["impl/c3.py"]
                elif field == "suggestion_text":
                    tasks[0]["action"] = "Unreviewed new direction"
                else:
                    ledger["action_plan"]["tasks"][0]["changes"][1]["method"] = "Unreviewed new method"
                with self.assertRaisesRegex(ValueError, "differs"):
                    validate_current_plan(ledger, tasks)

    def test_resume_rejects_old_plan_and_new_protection_conflict(self):
        with self.assertRaisesRegex(ValueError, "action_plan"):
            validate_current_plan({"modify_files": []}, [])
        value = normalize_decision_plan(decision())
        value["ledger_entry"]["readonly_files"].append("impl/c2.py")
        with self.assertRaisesRegex(ValueError, "readonly_files"):
            validate_current_plan(value["ledger_entry"], value["suggest_next"])

    def test_schema_and_template_are_independent_and_not_fake_completed_decisions(self):
        schema = build_decision_schema()
        schema["properties"]["suggest_next"]["items"]["required"].clear()
        self.assertIn("task_id", build_decision_schema()["properties"]["suggest_next"]["items"]["required"])
        template = build_decision_template(5, "id-5", ["case7", "case8"])
        self.assertEqual(template["iteration"], 5)
        self.assertEqual([item["case_id"] for item in template["ledger_entry"]["case_analysis"]], ["case7", "case8"])
        self.assertNotIn("modify_files", template["ledger_entry"])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            normalize_decision_plan(template)
        self.assertEqual(build_decision_template(5, "id-5", [], True)["suggest_next"], [])
        self.assertEqual(build_decision_template(5, "id-5", [])["suggest_next"][0]["case_scope"], "operator")

    def test_performance_contract_only_requires_current_triggered_experience(self):
        for flag, name, reason in (("has_improvement", "proven_pattern", "why_it_worked"),
                                   ("has_regression", "regression_pattern", "why_it_failed")):
            with self.subTest(name=name):
                schema = build_decision_schema(perf_diff={flag: True})
                template = build_decision_template(3, "request-3", ["case19"],
                                                   perf_diff={flag: True})
                pattern = schema["properties"][name]
                self.assertIn(name, schema["required"])
                self.assertIn(reason, pattern["required"])
                self.assertEqual(pattern["properties"]["applicability"]["type"], "string")
                case = pattern["properties"]["case_analysis"]["items"]
                self.assertIn("explanation", case["required"])
                self.assertEqual(set(template[name]), set(pattern["required"]))
                self.assertEqual(template[name]["case_analysis"][0]["explanation"], "")
                self.assertIsNone(template[name]["fusion_related"])
                # Mutating one saved request must not corrupt the next request.
                pattern["required"].clear()
                fresh = build_decision_schema(perf_diff={flag: True})
                self.assertIn(reason, fresh["properties"][name]["required"])
        for diff in (None, {}, {"has_improvement": False, "has_regression": False}):
            for name in ("proven_pattern", "regression_pattern"):
                self.assertNotIn(name, build_decision_schema(perf_diff=diff)["required"])
                self.assertNotIn(name, build_decision_template(3, "request-3", [], perf_diff=diff))

    def test_schema_has_no_shared_mutable_nodes_within_one_request(self):
        def check_node(value, path, seen):
            if not isinstance(value, (dict, list)):
                return
            self.assertNotIn(id(value), seen, f"Shared schema container at {path}; also at {seen.get(id(value))}")
            seen[id(value)] = path
            children = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in children:
                check_node(child, f"{path}.{key}", seen)

        for diff in (None, {"has_improvement": True}, {"has_regression": True}):
            with self.subTest(perf_diff=diff):
                check_node(build_decision_schema(perf_diff=diff), "schema", {})

    def test_binding_request_identity_does_not_constrain_unrelated_fields(self):
        def constants(value, path=()):
            result = {}
            if isinstance(value, dict):
                if "const" in value:
                    result[path] = value["const"]
                children = value.items()
            elif isinstance(value, list):
                children = enumerate(value)
            else:
                return result
            for key, child in children:
                result.update(constants(child, (*path, key)))
            return result

        for diff in (None, {"has_improvement": True}, {"has_regression": True}):
            with self.subTest(perf_diff=diff):
                schema = build_decision_schema(perf_diff=diff)
                original_constants = constants(schema)
                schema["properties"]["iteration"]["const"] = 3
                schema["properties"]["request_id"]["const"] = "request-3"
                self.assertEqual(constants(schema), {
                    **original_constants, ("properties", "iteration"): 3,
                    ("properties", "request_id"): "request-3",
                })
                self.assertEqual(constants(build_decision_schema(perf_diff=diff)), original_constants)
                if diff is None:
                    # A normal plan contains many strings distinct from its request ID.
                    _validate_schema(decision(), schema, "decision")

    def test_nested_rule_mutation_is_local_to_its_field(self):
        schema = build_decision_schema()
        case_fields = schema["properties"]["ledger_entry"]["properties"]["case_analysis"]["items"]["properties"]
        case_fields["explanation"]["minLength"] = 7
        self.assertEqual(case_fields["observation"]["minLength"], 1)
        self.assertEqual(schema["properties"]["request_id"]["minLength"], 1)
        ledger = schema["properties"]["ledger_entry"]["properties"]
        ledger["readonly_files"]["items"]["minLength"] = 9
        task = schema["properties"]["suggest_next"]["items"]["properties"]
        self.assertEqual(task["changes"]["items"]["properties"]["file"]["minLength"], 1)
        self.assertEqual(build_decision_schema()["properties"]["request_id"]["minLength"], 1)


if __name__ == "__main__":
    unittest.main()
