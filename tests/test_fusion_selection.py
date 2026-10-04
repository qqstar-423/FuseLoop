"""Offline Stage1.5 scoring, artifact integrity and downstream prompt checks."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lib.fusion_selection import (
    ROOT, _build_request, _rank_response, _validate_requirements, format_fusion_library_for_prompt,
    fusion_library_path, fusion_requirements_byte_budget, run_fusion_selection,
)
from lib.jev_client import JevSettings, check_budget


class FusionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.methods_path = self.work / "methods.md"
        self.methods_path.write_text("# 融合方法\nUse complete source, never a path only.", encoding="utf-8")
        self.options_path = self.work / "options.json"
        self.catalog = {"schema_version": 1, "caveats": ["Do not equate L2 and DSM."], "methods": [
            {"id": "F1", "name": "Fission", "requirements": ["Allow HBM intermediate"]},
            {"id": "F2", "name": "Vertical fusion", "requirements": ["On-chip dependent tiles"]},
            {"id": "F3", "name": "Horizontal fusion", "requirements": ["Independent branches"]},
        ]}
        self.write_json(self.options_path, self.catalog)
        (self.work / "ANALYSIS.md").write_text("Full stage1 analysis source.\n", encoding="utf-8")
        self.requirements = {
            "language": "en", "operator_summary": "Conv2d followed by sigmoid.",
            "semantics": {"bias": "optional", "output": "sigmoid(conv(x, weight) + bias)"},
            "case_groups": [{"dtype": "float16", "shape": [1, 32, 64, 64]}],
            "implementation_constraints": ["Use the current Triton Ascend runtime and hardware."],
            "optimization_hint": "Preserve every case and compare legal fusion choices.",
        }
        self.write_json(self.work / "fusion_requirements.en.json", self.requirements)
        self.hardware = {"chip_model": "synthetic Ascend", "ub_size_kb": 248, "device_id": 0}
        self.write_json(self.work / "device_info.json", self.hardware)
        self.settings = JevSettings(api_key="synthetic-key")
        self.translations = {
            "融合方法": "Fusion methods",
            "卷积后接 sigmoid。": "Convolution followed by sigmoid.",
            "保持精度": "Preserve accuracy",
            "设备探测成功": "Device detection succeeded",
            "不要把 L2 等同于 DSM。": "Do not equate L2 and DSM.",
            "垂直融合": "Vertical fusion",
            "允许 HBM 中间结果": "Allow HBM intermediates",
            "行优先": "Row major",
        }
        translation = patch("lib.kerminal_rpc.translate_fields", side_effect=self.translate_fields)
        self.translate = translation.start()
        self.addCleanup(translation.stop)
        settings_patch = patch("lib.jev_client.load_settings", return_value=self.settings)
        self.load_settings = settings_patch.start()
        self.addCleanup(settings_patch.stop)
        transport = patch("lib.jev_client.evaluate", return_value=self.response())
        self.evaluate = transport.start()
        self.addCleanup(transport.stop)

    def write_json(self, path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def response(self, values=(0.4, 0.9, 0.6)):
        return {"model": "jev-1.13.0", "answers": {
            method["id"]: {"type": "noul", "noul": value}
            for method, value in zip(self.catalog["methods"], values)
        }, "usage": {"input_tokens": 321, "output_tokens": 3}}

    def translate_fields(self, cli, text_fields, run_dir, timeout=240):
        self.assertEqual(cli, "synthetic-kerminal")
        translated = {}
        for key, value in text_fields.items():
            for original, english in self.translations.items():
                value = value.replace(original, english)
            translated[key] = value
        return translated

    def run_selection(self, top_n=2, **kwargs):
        return run_fusion_selection(
            self.work, top_n=top_n, methods_path=self.methods_path,
            options_path=self.options_path, translation_cli="synthetic-kerminal", **kwargs,
        )

    def read_artifact(self, name):
        return json.loads((self.work / "fusion" / name).read_text(encoding="utf-8"))

    def test_request_uses_full_source_structured_requirements_hardware_and_each_complete_method(self):
        library = self.run_selection()
        request = self.evaluate.call_args.args[0]
        self.assertEqual(request["state"]["fusion_methods_source"],
                         self.methods_path.read_bytes().decode("utf-8").replace("融合方法", "Fusion methods"))
        self.assertEqual(request["state"]["operator_requirements"], self.requirements)
        self.assertEqual(request["state"]["hardware"], self.hardware)
        self.assertEqual(request["state"]["language"], "en")
        self.translate.assert_called_once()
        self.assertEqual(request["state"]["catalog_context"]["caveats"], self.catalog["caveats"])
        self.assertEqual(set(request["questions"]), {"F1", "F2", "F3"})
        for method in self.catalog["methods"]:
            question = request["questions"][method["id"]]
            self.assertEqual(question["type"], "noul")
            self.assertEqual(json.loads(question["instructions"].split("Method JSON: ")[1]), method)
        self.assertIn("stage1_analysis", library["sources"])
        self.assertEqual(len(library["sources"]["stage1_analysis"]["sha256"]), 64)
        self.assertEqual(self.read_artifact("jev_request.json"), request)
        self.assertEqual(self.read_artifact("jev_response.json"), self.response())

    def test_ranking_is_descending_top_n_and_preserves_details_without_normalization(self):
        library = self.run_selection()
        self.assertEqual([c["method"]["id"] for c in library["candidates"]], ["F2", "F3"])
        self.assertEqual([c["probability"] for c in library["candidates"]], [0.9, 0.6])
        self.assertEqual(library["candidates"][0]["method"], self.catalog["methods"][1])
        self.assertEqual(len(self.read_artifact("ranking.json")["candidates"]), 3)
        self.assertEqual(self.read_artifact("fusion_library.json"), library)
        self.assertFalse(library["cache_reused"])

    def test_nested_chinese_values_are_translated_but_library_keeps_original_methods(self):
        self.catalog["caveats"] = ["不要把 L2 等同于 DSM。"]
        self.catalog["methods"][1].update({
            "name": "垂直融合",
            "requirements": ["允许 HBM 中间结果"],
            "variants": [{"id": "F2_v1", "layout": {"description": "行优先", "tile": [16, 32]}}],
        })
        self.requirements["operator_summary"] = "卷积后接 sigmoid。"
        self.requirements["case_groups"][0]["notes"] = ["保持精度"]
        self.hardware["probe"] = {"notes": ["设备探测成功"], "supported": True}
        self.write_json(self.options_path, self.catalog)
        self.write_json(self.work / "fusion_requirements.en.json", self.requirements)
        self.write_json(self.work / "device_info.json", self.hardware)

        library = self.run_selection()
        request = self.evaluate.call_args.args[0]
        self.assertNotRegex(json.dumps(request, ensure_ascii=False), r"[\u3400-\u9fff]")
        self.assertEqual(request["state"]["language"], "en")
        self.assertEqual(request["state"]["operator_requirements"]["operator_summary"],
                         "Convolution followed by sigmoid.")
        self.assertEqual(request["state"]["operator_requirements"]["case_groups"][0]["shape"], [1, 32, 64, 64])
        self.assertEqual(request["state"]["hardware"]["probe"],
                         {"notes": ["Device detection succeeded"], "supported": True})
        self.assertEqual(request["state"]["catalog_context"]["caveats"], ["Do not equate L2 and DSM."])
        method = json.loads(request["questions"]["F2"]["instructions"].split("Method JSON: ")[1])
        self.assertEqual(method["id"], "F2")
        self.assertEqual(method["name"], "Vertical fusion")
        self.assertEqual(method["variants"], [{"id": "F2_v1", "layout": {"description": "Row major", "tile": [16, 32]}}])
        self.assertEqual(library["candidates"][0]["method"], self.catalog["methods"][1])
        self.assertEqual(self.read_artifact("jev_request.json"), request)
        self.assertIn("融合方法", self.methods_path.read_text(encoding="utf-8"))

    def test_translation_failure_never_calls_jev_or_publishes_a_library(self):
        self.translate.side_effect = RuntimeError("synthetic translation failure")
        with self.assertRaisesRegex(RuntimeError, "translation failure"):
            self.run_selection()
        self.evaluate.assert_not_called()
        self.assertFalse(fusion_library_path(self.work).exists())
        self.assertFalse((self.work / "fusion/ranking.json").exists())

    def test_remaining_chinese_after_translation_never_reaches_jev(self):
        self.translate.side_effect = lambda cli, text_fields, run_dir, timeout=240: dict(text_fields)
        with self.assertRaises(ValueError):
            self.run_selection()
        self.evaluate.assert_not_called()
        self.assertFalse(fusion_library_path(self.work).exists())

    def test_expanded_translation_is_checked_against_final_request_budget(self):
        def expanded(cli, text_fields, run_dir, timeout=240):
            translated = self.translate_fields(cli, text_fields, run_dir, timeout)
            return {key: value + " English explanation" * 2000 for key, value in translated.items()}

        self.translate.side_effect = expanded
        with self.assertRaisesRegex(ValueError, "byte budget"):
            self.run_selection()
        self.evaluate.assert_not_called()
        self.assertFalse(fusion_library_path(self.work).exists())

    def test_translation_failure_after_a_source_change_removes_previous_library(self):
        self.run_selection()
        self.methods_path.write_text("# 融合方法\nNew requirements.", encoding="utf-8")
        self.translate.side_effect = RuntimeError("synthetic translation failure")
        with self.assertRaisesRegex(RuntimeError, "translation failure"):
            self.run_selection()
        self.evaluate.assert_called_once()
        self.assertFalse(fusion_library_path(self.work).exists())
        self.assertFalse((self.work / "fusion/ranking.json").exists())

    def test_equal_probabilities_keep_catalog_order(self):
        self.evaluate.return_value = self.response((0.8, 0.8, 0.8))
        result = self.run_selection()
        self.assertEqual([c["method"]["id"] for c in result["candidates"]], ["F1", "F2"])

    def test_top_n_change_reuses_scores_without_another_jev_call(self):
        original = self.run_selection(1)
        result = self.run_selection(3)
        self.evaluate.assert_called_once()
        self.translate.assert_called_once()
        self.assertTrue(result["cache_reused"])
        self.assertEqual(result["fingerprint_sha256"], original["fingerprint_sha256"])
        self.assertEqual(len(result["candidates"]), 3)

    def test_all_source_inputs_invalidate_cache_including_full_analysis(self):
        self.run_selection()
        for path in (self.methods_path, self.options_path, self.work / "ANALYSIS.md",
                     self.work / "fusion_requirements.en.json", self.work / "device_info.json"):
            with self.subTest(path=path.name):
                old_calls = self.evaluate.call_count
                # Whitespace changes preserve semantics but deliberately change source provenance.
                with path.open("a", encoding="utf-8") as stream:
                    stream.write("\n")
                result = self.run_selection()
                self.assertFalse(result["cache_reused"])
                self.assertEqual(self.evaluate.call_count, old_calls + 1)

    def test_model_and_budget_config_invalidate_cache_but_api_key_does_not_persist(self):
        original = self.run_selection()
        self.load_settings.return_value = replace(self.settings, model="jev-other")
        result = self.run_selection()
        self.assertNotEqual(original["fingerprint_sha256"], result["fingerprint_sha256"])
        self.assertEqual(self.evaluate.call_count, 2)
        self.load_settings.return_value = replace(self.settings, model="jev-other", max_request_bytes=55000)
        self.run_selection()
        self.assertEqual(self.evaluate.call_count, 3)
        self.load_settings.return_value = replace(self.load_settings.return_value, api_key="another-synthetic-key")
        self.assertTrue(self.run_selection()["cache_reused"])
        self.assertEqual(self.evaluate.call_count, 3)
        for path in (self.work / "fusion").iterdir():
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("synthetic-key", content)
                self.assertNotIn("api_key", content)

    def test_invalid_top_n_removes_old_library_without_request(self):
        for top_n in (0, -1, 4, 1.5, True, "2"):
            with self.subTest(top_n=top_n):
                self.run_selection()
                calls = self.evaluate.call_count
                with self.assertRaisesRegex(ValueError, "top_n"):
                    self.run_selection(top_n)
                self.assertEqual(self.evaluate.call_count, calls)
                self.assertFalse(fusion_library_path(self.work).exists())

    def test_invalid_probabilities_types_and_answer_sets_rejected(self):
        bad_responses = []
        for value in (True, False, "0.7", None, -0.1, 1.1, 10 ** 400, float("nan"), float("inf"), float("-inf")):
            bad_responses.append(self.response((value, 0.6, 0.8)))
        missing = self.response()
        missing["answers"].pop("F1")
        extra = self.response()
        extra["answers"]["other"] = {"type": "noul", "noul": 0.5}
        wrong_type = self.response()
        wrong_type["answers"]["F1"]["type"] = "score"
        bad_responses.extend([missing, extra, wrong_type, [], {}, {"answers": []}])
        for response in bad_responses:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    _rank_response(response, self.catalog["methods"])

    def test_invalid_provider_response_never_publishes_ranking_or_library(self):
        self.evaluate.return_value = self.response((1.5, 0.3, 0.8))
        with self.assertRaisesRegex(ValueError, "probability"):
            self.run_selection()
        self.assertTrue((self.work / "fusion/jev_response.json").exists())
        self.assertFalse((self.work / "fusion/ranking.json").exists())
        self.assertFalse(fusion_library_path(self.work).exists())

    def test_failed_fresh_evaluation_cannot_reuse_previous_success(self):
        self.run_selection()
        self.write_json(self.work / "device_info.json", {**self.hardware, "device_id": 1})
        self.evaluate.side_effect = RuntimeError("synthetic transport failure")
        with self.assertRaisesRegex(RuntimeError, "transport"):
            self.run_selection()
        self.assertFalse(fusion_library_path(self.work).exists())
        self.assertFalse((self.work / "fusion/ranking.json").exists())
        self.assertFalse((self.work / "fusion/jev_response.json").exists())
        with self.assertRaises(FileNotFoundError):
            format_fusion_library_for_prompt(self.work, 2)

    def test_configuration_or_input_failure_cannot_leave_a_previous_library(self):
        self.run_selection()
        self.load_settings.side_effect = ValueError("synthetic settings failure")
        with self.assertRaisesRegex(ValueError, "settings"):
            self.run_selection()
        self.assertFalse(fusion_library_path(self.work).exists())
        self.assertTrue((self.work / "fusion/jev_request.json").exists())
        self.assertTrue((self.work / "fusion/jev_response.json").exists())
        self.load_settings.side_effect = None
        self.run_selection()
        (self.work / "fusion_requirements.en.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_selection()
        self.assertFalse(fusion_library_path(self.work).exists())

    def test_corrupt_cached_response_triggers_new_evaluation(self):
        self.run_selection()
        self.write_json(self.work / "fusion/jev_response.json", self.response((0.3, 0.2, 0.1)))
        result = self.run_selection()
        self.assertFalse(result["cache_reused"])
        self.assertEqual(self.evaluate.call_count, 2)
        self.assertEqual(result["candidates"][0]["method"]["id"], "F2")

    def test_byte_budget_failure_never_calls_transport_or_keeps_old_scores(self):
        self.run_selection()
        self.load_settings.return_value = replace(self.settings, max_state_question_bytes=1)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            self.run_selection()
        self.evaluate.assert_called_once()
        self.assertFalse(fusion_library_path(self.work).exists())
        self.assertTrue((self.work / "fusion/jev_request.json").exists())

    def test_missing_requirements_fields_language_and_oversize_fail_before_transport(self):
        invalid = []
        for field in ("operator_summary", "semantics", "case_groups", "implementation_constraints", "optimization_hint"):
            candidate = deepcopy(self.requirements)
            candidate[field] = []
            invalid.append(candidate)
        invalid.extend([{}, [], {**self.requirements, "language": "zh"},
                        {**self.requirements, "operator_summary": "x" * 6001}])
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                self.write_json(self.work / "fusion_requirements.en.json", candidate)
                with self.assertRaises(ValueError):
                    self.run_selection()
        self.evaluate.assert_not_called()

    def test_empty_optional_optimization_hint_and_pretty_json_are_accepted(self):
        self.requirements["optimization_hint"] = ""
        (self.work / "fusion_requirements.en.json").write_text(
            json.dumps(self.requirements, indent=200), encoding="utf-8",
        )
        self.assertGreater((self.work / "fusion_requirements.en.json").stat().st_size, 6000)
        self.run_selection()
        self.evaluate.assert_called_once()

    def test_legacy_consumers_without_new_stage_inputs_preserve_existing_route(self):
        (self.work / "fusion_requirements.en.json").unlink()
        for stage in ("stage3", "stage7", "stage8", "stage9"):
            self.assertIn("legacy task", format_fusion_library_for_prompt(self.work, stage))
        with self.assertRaises(FileNotFoundError):
            format_fusion_library_for_prompt(self.work, "stage2")
        (self.work / "fusion").mkdir()
        with self.assertRaises(FileNotFoundError):
            format_fusion_library_for_prompt(self.work, "stage9")

    def test_duplicate_catalog_ids_empty_hardware_and_bad_source_are_rejected(self):
        self.write_json(self.options_path, {"schema_version": 1, "methods": [self.catalog["methods"][0]] * 2})
        with self.assertRaisesRegex(ValueError, "unique"):
            self.run_selection()
        self.write_json(self.options_path, self.catalog)
        self.write_json(self.work / "device_info.json", {})
        with self.assertRaisesRegex(ValueError, "hardware"):
            self.run_selection()
        self.write_json(self.work / "device_info.json", self.hardware)
        self.methods_path.write_bytes(b"hello\x00")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            self.run_selection()
        self.evaluate.assert_not_called()

    def test_all_five_stages_receive_full_candidates_and_stage2_highest_guidance(self):
        library = self.run_selection()
        for stage in (2, 3, 7, 8, 9):
            with self.subTest(stage=stage):
                prompt = format_fusion_library_for_prompt(self.work, stage)
                self.assertIn(str(fusion_library_path(self.work).resolve()), prompt)
                self.assertIn(json.dumps(library, ensure_ascii=False, indent=2), prompt)
                self.assertIn("do not modify", prompt)
                if stage == 2:
                    self.assertIn("highest-probability", prompt)
        with self.assertRaises(ValueError):
            format_fusion_library_for_prompt(self.work, 6)

    def test_malformed_library_is_rejected_before_prompt_injection(self):
        for candidate in ([], {}, {"schema_version": 1, "top_n": 0, "candidates": []}):
            with self.subTest(candidate=candidate):
                self.run_selection()
                self.write_json(fusion_library_path(self.work), candidate)
                with self.assertRaises(ValueError):
                    format_fusion_library_for_prompt(self.work, 2)

    def test_real_catalog_and_source_fit_budget_with_representative_requirements(self):
        # The real source stays complete. This is a request-only budget check, no API call.
        catalog = json.loads((ROOT / "knowledge/fusion_options.json").read_text(encoding="utf-8"))
        self.assertEqual({m["id"] for m in catalog["methods"]}, {f"F{i}" for i in range(1, 11)})
        source = (ROOT / "knowledge/fusion_method.md").read_bytes().decode("utf-8-sig")
        request = _build_request(
            source, self.requirements, self.hardware, catalog["methods"], self.settings.model,
            catalog_context={key: value for key, value in catalog.items() if key != "methods"},
        )
        sizes = check_budget(request, self.settings)
        self.assertLessEqual(sizes["state_plus_longest_question_bytes"], 28000)
        self.assertLessEqual(sizes["request_bytes"], 56000)
        self.assertEqual(len(request["questions"]), 10)
        self.evaluate.assert_not_called()

    def test_individual_requirements_limit_does_not_bypass_real_combined_budget(self):
        catalog = json.loads((ROOT / "knowledge/fusion_options.json").read_text(encoding="utf-8"))
        source = (ROOT / "knowledge/fusion_method.md").read_bytes().decode("utf-8-sig")
        requirements = deepcopy(self.requirements)
        requirements["optimization_hint"] = ""
        serialized_size = len(json.dumps(requirements, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        requirements["optimization_hint"] = "x" * (6000 - serialized_size)
        request = _build_request(
            source, requirements, self.hardware, catalog["methods"], self.settings.model,
            catalog_context={key: value for key, value in catalog.items() if key != "methods"},
        )
        # A 6000-byte requirement can fit a shorter real source. The combined
        # limit must still reject it when configured below the actual request.
        sizes = check_budget(request, self.settings)
        constrained = replace(self.settings,
                              max_state_question_bytes=sizes["state_plus_longest_question_bytes"] - 1)
        with self.assertRaisesRegex(ValueError, "state \\+ longest question"):
            check_budget(request, constrained)
        self.evaluate.assert_not_called()

    def test_dynamic_requirements_budget_fits_real_request_and_rejects_next_byte(self):
        catalog = json.loads((ROOT / "knowledge/fusion_options.json").read_text(encoding="utf-8"))
        source = (ROOT / "knowledge/fusion_method.md").read_bytes().decode("utf-8-sig")
        hardware = {
            "chip_model": "Ascend 950", "soc_version": "Ascend950", "npu_arch": "dav-3510",
            "ai_core_num": 32, "ub_size_kb": 248, "l1_size_kb": 1024,
            "l0a_size_kb": 64, "l0b_size_kb": 64, "l0c_size_kb": 256,
            "device_id": 0, "detect_method": "synthetic hardware fixture",
        }
        empty = _build_request(
            source, {}, hardware, catalog["methods"], self.settings.model,
            catalog_context={key: value for key, value in catalog.items() if key != "methods"})
        empty_sizes = check_budget(empty, self.settings)
        limits = [(28000, 56000),
                  (empty_sizes["state_plus_longest_question_bytes"] + 2048, 56000),
                  (28000, empty_sizes["request_bytes"] + 2048)]
        for state_limit, request_limit in limits:
            with self.subTest(state_limit=state_limit, request_limit=request_limit):
                allowed = fusion_requirements_byte_budget(
                    hardware, max_state_question_bytes=state_limit, max_request_bytes=request_limit)
                requirements = deepcopy(self.requirements)
                requirements["optimization_hint"] = ""
                initial = len(json.dumps(requirements, ensure_ascii=False, sort_keys=True).encode("utf-8"))
                requirements["optimization_hint"] = "x" * (allowed - initial)
                request = _build_request(
                    source, requirements, hardware, catalog["methods"], self.settings.model,
                    catalog_context={key: value for key, value in catalog.items() if key != "methods"},
                )
                settings = replace(self.settings, max_state_question_bytes=state_limit,
                                   max_request_bytes=request_limit)
                sizes = check_budget(request, settings)
                _validate_requirements(requirements)
                self.assertLessEqual(sizes["state_plus_longest_question_bytes"], state_limit)
                self.assertLessEqual(sizes["request_bytes"], request_limit)
                self.assertTrue(allowed == 6000 or sizes["state_plus_longest_question_bytes"] == state_limit
                                or sizes["request_bytes"] == request_limit)
                request["state"]["operator_requirements"]["optimization_hint"] += "x"
                with self.assertRaisesRegex(ValueError, "byte budget"):
                    if allowed == 6000:
                        _validate_requirements(request["state"]["operator_requirements"])
                    else:
                        check_budget(request, settings)
        self.load_settings.assert_not_called()
        self.evaluate.assert_not_called()

    def test_dynamic_budget_refuses_insufficient_capacity_without_sdk_or_credentials(self):
        with self.assertRaisesRegex(ValueError, "insufficient"):
            fusion_requirements_byte_budget(self.hardware, max_state_question_bytes=100)
        self.load_settings.assert_not_called()
        self.evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
