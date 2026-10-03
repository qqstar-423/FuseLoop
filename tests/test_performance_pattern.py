"""Offline checks for the shared Stage9 experience schema and diagnostics."""

from copy import deepcopy
import unittest

from lib.performance_pattern import (
    PerformancePatternValidationError,
    build_performance_pattern_schema,
    build_performance_pattern_template,
    validate_performance_pattern_details,
)


def complete_pattern(name="proven_pattern"):
    result = {
        "applicability": "Current Ascend chip; only the measured shapes are verified",
        "next_action": "Compare the neighboring tile size under the same timing protocol",
        "evidence": ["eval/iter3/perf_result.json"],
        "case_analysis": [{"case_id": "operator_1", "observation": "Kernel latency decreased",
                           "explanation": "Hypothesis: less padding; confirm with a controlled tile change"}],
    }
    if name in {"proven_pattern", "regression_pattern"}:
        why = "why_it_worked" if name == "proven_pattern" else "why_it_failed"
        result.update(what_changed="Changed the reduction tile", fusion_related=False)
        result[why] = "The measured kernel timings changed; the cause still needs a controlled comparison"
    return result


class PerformancePatternTests(unittest.TestCase):
    def test_reports_bad_applicability_and_all_eight_missing_explanations_together(self):
        value = complete_pattern()
        value["applicability"] = {"hardware": "Ascend"}
        value["case_analysis"] = [{"case_id": f"operator_{index}", "observation": "Measured latency changed"}
                                  for index in range(8)]
        with self.assertRaises(PerformancePatternValidationError) as caught:
            validate_performance_pattern_details(value, "proven_pattern")
        errors = caught.exception.errors
        self.assertEqual(len(errors), 9)
        self.assertIn("proven_pattern.applicability", errors[0])
        for index in range(8):
            self.assertIn(f"proven_pattern.case_analysis[{index}].explanation", errors[index + 1])
            self.assertIn(f'case_id="operator_{index}"', errors[index + 1])
        self.assertEqual(str(caught.exception), "; ".join(errors))

    def test_concrete_names_require_their_own_basic_fields(self):
        for name, why in (("proven_pattern", "why_it_worked"), ("regression_pattern", "why_it_failed")):
            with self.subTest(name=name):
                value = complete_pattern(name)
                validate_performance_pattern_details(value, name)
                value.pop(why)
                value["what_changed"] = " "
                value["fusion_related"] = None
                with self.assertRaises(PerformancePatternValidationError) as caught:
                    validate_performance_pattern_details(value, name)
                self.assertEqual(len(caught.exception.errors), 3)
                for field in ("what_changed", why, "fusion_related"):
                    self.assertIn(f"{name}.{field}", str(caught.exception))

    def test_generic_name_keeps_details_only_compatibility(self):
        value = complete_pattern("performance_pattern")
        validate_performance_pattern_details(value)
        with self.assertRaises(PerformancePatternValidationError) as caught:
            validate_performance_pattern_details(value, "regression_pattern")
        self.assertEqual(len(caught.exception.errors), 3)

    def test_nonempty_text_rules_apply_to_every_text_field(self):
        for name in ("proven_pattern", "regression_pattern"):
            template = build_performance_pattern_template(name)
            text_fields = [key for key, value in template.items() if isinstance(value, str) and key != "evidence"]
            for field in text_fields:
                for invalid in (None, "", " \n\t", {}, [], 1, True):
                    with self.subTest(name=name, field=field, invalid=invalid):
                        value = complete_pattern(name)
                        value[field] = invalid
                        with self.assertRaises(PerformancePatternValidationError) as caught:
                            validate_performance_pattern_details(value, name)
                        self.assertEqual(len(caught.exception.errors), 1)
                        self.assertIn(f"{name}.{field}", str(caught.exception))

    def test_each_case_field_rejects_missing_null_blank_and_wrong_types(self):
        for name in ("proven_pattern", "regression_pattern"):
            for field in ("case_id", "observation", "explanation"):
                for invalid in (None, "", " \n", {}, [], 1, False):
                    with self.subTest(name=name, field=field, invalid=invalid):
                        value = complete_pattern(name)
                        value["case_analysis"][0][field] = invalid
                        with self.assertRaises(PerformancePatternValidationError) as caught:
                            validate_performance_pattern_details(value, name)
                        self.assertEqual(len(caught.exception.errors), 1)
                        self.assertIn(f"{name}.case_analysis[0].{field}", str(caught.exception))
                value = complete_pattern(name)
                value["case_analysis"][0].pop(field)
                with self.subTest(name=name, field=field, missing=True):
                    with self.assertRaises(PerformancePatternValidationError):
                        validate_performance_pattern_details(value, name)

    def test_container_and_boolean_types_are_checked(self):
        for invalid in (None, [], "", True, 1):
            with self.subTest(root=invalid), self.assertRaises(PerformancePatternValidationError):
                validate_performance_pattern_details(invalid)
        for field, invalid_values in (
            ("case_analysis", (None, [], {}, "analysis", True)),
            ("fusion_related", (None, "false", 0, 1, [], {})),
        ):
            for invalid in invalid_values:
                value = complete_pattern()
                value[field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(PerformancePatternValidationError):
                    validate_performance_pattern_details(value, "proven_pattern")
        value = complete_pattern()
        value["case_analysis"] = [None, "case2", {}]
        with self.assertRaises(PerformancePatternValidationError) as caught:
            validate_performance_pattern_details(value, "proven_pattern")
        self.assertEqual(len(caught.exception.errors), 5)
        self.assertIn("case_analysis[0] must be an object", caught.exception.errors[0])
        self.assertIn("case_analysis[1] must be an object", caught.exception.errors[1])

    def test_evidence_retains_shallow_nonempty_contract(self):
        for evidence in ("eval/iter3/perf_result.json", [""], [None], {"path": None}):
            value = complete_pattern()
            value["evidence"] = evidence
            with self.subTest(valid=evidence):
                validate_performance_pattern_details(value, "proven_pattern")
        for evidence in (None, "", " \n", {}, [], True, 1):
            value = complete_pattern()
            value["evidence"] = evidence
            with self.subTest(invalid=evidence), self.assertRaises(PerformancePatternValidationError):
                validate_performance_pattern_details(value, "proven_pattern")

    def test_schema_and_template_describe_the_same_fields(self):
        for name in ("proven_pattern", "regression_pattern", "performance_pattern"):
            with self.subTest(name=name):
                schema = build_performance_pattern_schema(name)
                template = build_performance_pattern_template(name)
                self.assertEqual(set(template), set(schema["required"]))
                self.assertEqual(set(template), set(schema["properties"]))
                case_schema = schema["properties"]["case_analysis"]["items"]
                self.assertEqual(set(template["case_analysis"][0]), set(case_schema["required"]))
                self.assertNotIn("additionalProperties", schema)
                self.assertNotIn("additionalProperties", case_schema)
                self.assertEqual(schema["properties"]["applicability"],
                                 {"type": "string", "minLength": 1, "pattern": r"\S"})
                self.assertEqual(schema["properties"]["evidence"]["anyOf"][1:],
                                 [{"type": "object", "minProperties": 1}, {"type": "array", "minItems": 1}])
                if name != "performance_pattern":
                    self.assertIsNone(template["fusion_related"])
                    self.assertEqual(schema["properties"]["fusion_related"], {"type": "boolean"})
                with self.assertRaises(PerformancePatternValidationError) as caught:
                    validate_performance_pattern_details(template, name)
                self.assertEqual(len(caught.exception.errors), len(template) + 2)

    def test_no_input_mutation_or_shared_schema_template_state(self):
        value = complete_pattern()
        value["additional_detail"] = {"source": "model"}
        value["case_analysis"][0]["extra"] = True
        before = deepcopy(value)
        validate_performance_pattern_details(value, "proven_pattern")
        self.assertEqual(value, before)
        schema = build_performance_pattern_schema("proven_pattern")
        schema["properties"]["case_analysis"]["items"]["required"].clear()
        self.assertEqual(len(build_performance_pattern_schema("proven_pattern")["properties"]["case_analysis"]["items"]["required"]), 3)
        template = build_performance_pattern_template("proven_pattern")
        template["case_analysis"][0]["case_id"] = "modified"
        self.assertEqual(build_performance_pattern_template("proven_pattern")["case_analysis"][0]["case_id"], "")

    def test_diagnostics_escape_case_id_newlines(self):
        value = complete_pattern()
        value["case_analysis"][0] = {"case_id": "中文\ncase\rname", "observation": "Changed"}
        with self.assertRaises(PerformancePatternValidationError) as caught:
            validate_performance_pattern_details(value, "proven_pattern")
        self.assertNotIn("\n", str(caught.exception))
        self.assertNotIn("\r", str(caught.exception))
        self.assertIn('case_id="中文\\ncase\\rname"', str(caught.exception))


if __name__ == "__main__":
    unittest.main()
