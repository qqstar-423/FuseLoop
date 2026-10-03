"""Offline tests for lossless schema handling around English translation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lib.jev_translation import assert_english_payload, prepare_english_payload


class JevTranslationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.payload = {
            "fusion_methods_source": "# 一、融合 F1\n用 128MB HBM，输入 fp16。",
            "operator_requirements": {"language": "en", "semantics": "卷积之后激活", "shape": [1, 16]},
            "hardware": {"chip_model": "Ascend 950", "notes": "设备说明", "cores": 32, "enabled": True},
            "catalog": {"schema_version": 1, "methods": [{"id": "F1", "name": "Fission",
                        "name_zh": "半融合", "source_sections": ["第一节"],
                        "variants": [{"id": "F1.compute_split", "description": "拆分 F1 到 HBM"}]}]},
        }
        self.english = {
            "# 一、融合 F1\n用 128MB HBM，输入 fp16。": "# One: fusion F1\nUse 128MB HBM with fp16 inputs.",
            "卷积之后激活": "Activation after convolution", "设备说明": "Device notes",
            "半融合": "Partial fusion", "第一节": "Section one", "拆分 F1 到 HBM": "Split F1 into HBM",
        }
        mocked = patch("lib.kerminal_rpc.translate_fields", side_effect=self.translate)
        self.translate_fields = mocked.start()
        self.addCleanup(mocked.stop)

    def translate(self, cli, fields, directory, timeout=240):
        self.assertEqual(cli, "synthetic-kerminal")
        self.assertEqual(directory, self.work / "translation")
        return {key: self.english[value] for key, value in fields.items()}

    def prepare(self, **kwargs):
        return prepare_english_payload(self.payload, self.work, cli="synthetic-kerminal", **kwargs)

    def assert_not_published(self):
        self.assertFalse((self.work / "english_inputs.json").exists())
        self.assertFalse((self.work / "translation_manifest.json").exists())

    def test_translates_every_nested_value_with_program_owned_keys_and_structure(self):
        original = deepcopy(self.payload)
        result = self.prepare(timeout=71)
        self.translate_fields.assert_called_once()
        fields = self.translate_fields.call_args.args[1]
        self.assertEqual(set(fields.values()), set(self.english))
        self.assertTrue(all(key.startswith("field_") for key in fields))
        self.assertEqual(self.translate_fields.call_args.kwargs["timeout"], 71)
        self.assertEqual(self.payload, original)
        self.assertEqual(result["operator_requirements"]["shape"], [1, 16])
        self.assertIs(result["hardware"]["enabled"], True)
        method = result["catalog"]["methods"][0]
        self.assertEqual(method["id"], "F1")
        self.assertEqual(method["name_zh"], "Partial fusion")
        self.assertEqual(method["source_sections"], ["Section one"])
        self.assertEqual(method["variants"][0]["id"], "F1.compute_split")
        assert_english_payload(result)
        self.assertEqual(json.loads((self.work / "translation_input.json").read_text(encoding="utf-8")), original)
        self.assertEqual(json.loads((self.work / "english_inputs.json").read_text(encoding="utf-8")), result)

    def test_plain_english_requires_no_translator_or_cli(self):
        self.payload = {"model": "jev-1.13.0", "state": {"note": "Use HBM ≥ 128 MB; café ×2", "unset": None}}
        result = prepare_english_payload(self.payload, self.work, cli=None)
        self.assertEqual(result, self.payload)
        self.translate_fields.assert_not_called()

    def test_missing_cli_fails_without_published_result(self):
        with self.assertRaisesRegex(ValueError, "Kerminal CLI"):
            prepare_english_payload(self.payload, self.work, cli="")
        self.assert_not_published()
        self.translate_fields.assert_not_called()

    def test_keys_and_non_english_identifiers_are_never_translated(self):
        for value in ({"中文": "value"}, {"model": "模型"}, {"id": "方案F1"},
                      {"methods": [{"method_id": "融合"}]}, {"compatible_with": ["方案F1"]}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "keys|identifiers"):
                prepare_english_payload(value, self.work, cli="synthetic-kerminal")
            self.assert_not_published()
        self.translate_fields.assert_not_called()

    def test_rejects_missing_extra_non_string_and_blank_translations(self):
        self.translate_fields.side_effect = None
        for response in ({}, {"extra": "English"}, ["English"]):
            with self.subTest(response=response):
                self.translate_fields.return_value = response
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assert_not_published()
        self.payload = {"note": "半融合"}
        for value in (None, 2, [], {}, "", "  "):
            with self.subTest(value=value):
                self.translate_fields.return_value = {"field_000000": value}
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assert_not_published()

    def test_rejects_numeric_and_identifier_drift(self):
        original = self.payload["fusion_methods_source"]
        for replacement in ("# One: fusion F1\nUse 256MB HBM with fp16 inputs.",
                            "# One: fusion G1\nUse 128MB HBM with fp16 inputs.",
                            "# One: fusion F1\nUse 128MB DDR with fp16 inputs.",
                            "# 1: fusion F1\nUse 128MB HBM with fp16 inputs."):
            with self.subTest(replacement=replacement):
                self.english[original] = replacement
                with self.assertRaisesRegex(ValueError, "numeric|identifiers"):
                    self.prepare()
                self.assert_not_published()

    def test_rejects_remaining_scripts_even_with_en_flag_or_json_unicode_escapes(self):
        values = ["融合", "日本語", "한글", "данные", "مرحبا", r"\u4e2d\u6587",
                  r'{"note": "\u4e2d\u6587"}', r'{"note": "\\u4e2d\\u6587"}',
                  r"\u005cu4e2d\u005cu6587"]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                assert_english_payload({"language": "en", "deep": [{"text": value}]})
        self.english["半融合"] = r"Partial \u878d\u5408"
        with self.assertRaisesRegex(ValueError, "non-English"):
            self.prepare()
        self.assert_not_published()

    def test_numeric_signs_and_scientific_exponents_are_preserved(self):
        for source, incorrect in (("偏移 -1", "Offset 1"), ("偏移 -1", "Offset +1"),
                                  ("偏移 +1", "Offset -1"), ("误差 1e-5", "Error 1e5"),
                                  ("误差 1e+5", "Error 1e-5")):
            with self.subTest(source=source, incorrect=incorrect):
                self.payload = {"note": source}
                self.english[source] = incorrect
                with self.assertRaisesRegex(ValueError, "numeric"):
                    self.prepare()
                self.assert_not_published()

    def test_new_english_technical_terms_allowed_and_identifier_ranges_preserved(self):
        self.payload = {"note": "F1-F7 输入输出，使用矩阵单元；容差 1e-5。"}
        self.english[self.payload["note"]] = "F1-F7 InputOutput, using CUBE; tolerance 1e-5."
        result = self.prepare()
        self.assertEqual(result["note"], self.english[self.payload["note"]])

    def test_literal_escaped_input_is_decoded_before_translation(self):
        self.payload = {"note": r"\u534a\u878d\u5408"}
        result = self.prepare()
        self.assertEqual(result["note"], "Partial fusion")
        self.assertEqual(self.translate_fields.call_args.args[1], {"field_000000": "半融合"})

    def test_valid_cache_reused_and_original_changes_invalidate_it(self):
        result = self.prepare()
        self.assertEqual(self.prepare(), result)
        self.translate_fields.assert_called_once()
        self.payload["hardware"]["cores"] = 64
        self.assertEqual(self.prepare()["hardware"]["cores"], 64)
        self.assertEqual(self.translate_fields.call_count, 2)

    def test_cache_hash_tampering_and_version_changes_require_retranslation(self):
        self.prepare()
        output = self.work / "english_inputs.json"
        output.write_text('{"unexpected": "English"}', encoding="utf-8")
        self.prepare()
        self.assertEqual(self.translate_fields.call_count, 2)
        manifest_path = self.work / "translation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["version"] = -1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.prepare()
        self.assertEqual(self.translate_fields.call_count, 3)

    def test_cache_is_revalidated_even_when_output_hash_matches(self):
        self.prepare()
        output_path = self.work / "english_inputs.json"
        manifest_path = self.work / "translation_manifest.json"
        output = json.loads(output_path.read_text(encoding="utf-8"))
        output["catalog"]["methods"][0]["name_zh"] = "未翻译"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_sha256"] = hashlib.sha256(json.dumps(output, ensure_ascii=False, sort_keys=True,
                                                              allow_nan=False).encode("utf-8")).hexdigest()
        output_path.write_text(json.dumps(output), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.prepare()
        self.assertEqual(self.translate_fields.call_count, 2)

    def test_translation_failure_removes_previous_usable_artifacts(self):
        self.prepare()
        self.payload["hardware"]["cores"] = 64
        self.translate_fields.side_effect = RuntimeError("Synthetic translation failure")
        with self.assertRaisesRegex(RuntimeError, "translation failure"):
            self.prepare()
        self.assert_not_published()
        self.assertTrue((self.work / "translation_input.json").exists())

    def test_input_validation_failure_removes_previous_usable_artifacts(self):
        self.prepare()
        self.payload["硬件"] = "bad key"
        with self.assertRaises(ValueError):
            self.prepare()
        self.assert_not_published()

    def test_non_json_and_nonfinite_values_rejected(self):
        for value in (float("nan"), float("inf"), ("tuple",), {"set"}):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                prepare_english_payload({"value": value}, self.work, cli="synthetic-kerminal")
            self.assert_not_published()


if __name__ == "__main__":
    unittest.main()
