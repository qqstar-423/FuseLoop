"""Offline Jev transport checks; all credentials and SDK responses are synthetic."""

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from lib.jev_client import (
    JevSettings,
    build_request,
    check_budget,
    evaluate,
    load_settings,
    read_source,
)


class JevClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        transport = patch("lib.jev_client.TypeSafeClient", autospec=True)
        self.client_factory = transport.start()
        self.addCleanup(transport.stop)
        self.settings = JevSettings(api_key="synthetic-key-for-offline-tests")
        self.request = {
            "model": self.settings.model,
            "state": {"language": "en", "summary": "Three validated measurements."},
            "questions": {
                "ready": {"type": "noul", "instructions": "Is the evidence complete?"}
            },
        }

    def write_json(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def input_files(self):
        source = self.directory / "kernel.py"
        source.write_text("def kernel(x):\n    return x + 1\n", encoding="utf-8")
        evidence = self.write_json("evidence.en.json", {
            "language": "en",
            "profiling_analysis": {"bottleneck": "Memory traffic dominates."},
            "iteration_history": [{"direction": "Reduce memory traffic", "gain": 0.04}],
        })
        questions = self.write_json("questions.json", self.request["questions"])
        return source, evidence, questions

    def configure(self, *, inline_key=None, credentials_file="config.local.yaml"):
        config = self.directory / "config.yaml"
        content = "jev:\n  api_key_env: JEV_OFFLINE_TEST_KEY\n"
        if inline_key is not None:
            content += f"  api_key: {json.dumps(inline_key)}\n"
        if credentials_file is not None:
            content += f"  credentials_file: {credentials_file}\n"
        config.write_text(content, encoding="utf-8")
        return config

    def test_environment_key_precedes_inline_and_local_file_and_repr_hides_key(self):
        config = self.configure(inline_key="synthetic-inline-key")
        local = self.directory / "config.local.yaml"
        local.write_text("jev:\n  api_key: synthetic-local-key\n", encoding="utf-8")
        os.environ["JEV_OFFLINE_TEST_KEY"] = "  synthetic-environment-key  "
        original_read = Path.read_text

        def guarded_read(path, *args, **kwargs):
            if path == local:
                self.fail("An environment key must avoid reading local credentials.")
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=guarded_read):
            settings = load_settings(config)
        self.assertEqual(settings.api_key, "synthetic-environment-key")
        self.assertNotIn(settings.api_key, repr(settings))
        self.assertNotIn("synthetic-inline-key", repr(settings))
        self.assertNotIn("synthetic-local-key", repr(settings))

    def test_inline_key_works_without_a_second_config_file(self):
        config = self.configure(inline_key="  synthetic-inline-key  ", credentials_file=None)
        self.assertEqual(load_settings(config).api_key, "synthetic-inline-key")
        self.assertEqual(list(self.directory.iterdir()), [config])
        self.client_factory.assert_not_called()

    def test_inline_key_precedes_explicit_legacy_credentials_without_reading_them(self):
        config = self.configure(inline_key="synthetic-inline-key")
        local = self.directory / "config.local.yaml"
        local.write_text("jev:\n  api_key: synthetic-local-key\n", encoding="utf-8")
        original_read = Path.read_text

        def guarded_read(path, *args, **kwargs):
            if path == local:
                self.fail("An inline key must avoid reading legacy credentials.")
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=guarded_read):
            self.assertEqual(load_settings(config).api_key, "synthetic-inline-key")

    def test_legacy_credentials_are_relative_to_the_selected_config_directory(self):
        selected = self.directory / "selected"
        selected.mkdir()
        credentials = selected / "secrets" / "legacy.yaml"
        credentials.parent.mkdir()
        credentials.write_text("jev:\n  api_key: synthetic-relative-key\n", encoding="utf-8")
        config = selected / "settings.yaml"
        config.write_text("jev:\n  credentials_file: secrets/legacy.yaml\n", encoding="utf-8")
        self.assertEqual(load_settings(config).api_key, "synthetic-relative-key")

    def test_unspecified_credentials_file_never_reads_leftover_local_config(self):
        config = self.configure(credentials_file=None)
        local = self.directory / "config.local.yaml"
        local.write_text("jev:\n  api_key: synthetic-leftover-key\n", encoding="utf-8")
        original_read = Path.read_text

        def guarded_read(path, *args, **kwargs):
            if path == local:
                self.fail("Legacy credentials require an explicit credentials_file setting.")
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=guarded_read):
            with self.assertRaisesRegex(ValueError, "jev.api_key.*environment variable"):
                load_settings(config)
        self.client_factory.assert_not_called()

    def test_local_key_fallback_and_missing_key_are_explicit(self):
        config = self.configure()
        local = self.directory / "config.local.yaml"
        local.write_text("jev:\n  api_key: synthetic-local-key\n", encoding="utf-8")
        os.environ["JEV_OFFLINE_TEST_KEY"] = "   "
        self.assertEqual(load_settings(config).api_key, "synthetic-local-key")
        local.write_text("jev: {}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "API key is missing"):
            load_settings(config)
        self.client_factory.assert_not_called()

    def test_source_contains_actual_utf8_text_and_hash_of_original_bytes(self):
        path = self.directory / "kernel.py"
        source = "# 融合算子\r\ndef kernel(x):\r\n    return x + 1\r\n"
        original = b"\xef\xbb\xbf" + source.encode("utf-8")
        path.write_bytes(original)
        result = read_source(path)
        self.assertEqual(result, {
            "filename": "kernel.py",
            "sha256": hashlib.sha256(original).hexdigest(),
            "source": source,
        })

    def test_empty_whitespace_and_binary_sources_are_rejected(self):
        path = self.directory / "kernel.py"
        for data in (b"", b" \r\n\t", b"\xef\xbb\xbf", b"print(1)\x00", b"\xff\xfe"):
            with self.subTest(data=data):
                path.write_bytes(data)
                with self.assertRaises((ValueError, UnicodeError)):
                    read_source(path)
        self.client_factory.assert_not_called()

    def test_request_merges_english_evidence_with_source_and_questions(self):
        source, evidence, questions = self.input_files()
        result = build_request(source, evidence, questions, model=self.settings.model)
        self.assertEqual(result["model"], self.settings.model)
        self.assertEqual(result["questions"], self.request["questions"])
        self.assertEqual(result["state"]["language"], "en")
        self.assertEqual(result["state"]["profiling_analysis"], {
            "bottleneck": "Memory traffic dominates."
        })
        self.assertEqual(result["state"]["iteration_history"][0]["gain"], 0.04)
        self.assertEqual(result["state"]["kernel"]["source"], source.read_bytes().decode("utf-8"))
        self.assertEqual(result["state"]["kernel"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_evidence_requires_english_marker_and_cannot_replace_source(self):
        source, evidence, questions = self.input_files()
        for payload in ([], {}, {"language": "zh"}, {"language": "en", "kernel": {"source": "fake"}}):
            with self.subTest(evidence=payload):
                evidence.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_request(source, evidence, questions)
        self.client_factory.assert_not_called()

    def test_english_marker_cannot_hide_untranslated_request_content(self):
        for location in ("evidence", "source", "questions"):
            with self.subTest(location=location):
                source, evidence, questions = self.input_files()
                if location == "evidence":
                    evidence.write_text(json.dumps({
                        "language": "en", "profiling_analysis": {"bottleneck": "内存访问"},
                    }), encoding="utf-8")
                elif location == "source":
                    source.write_text("# 融合算子\ndef kernel(x):\n    return x + 1\n", encoding="utf-8")
                else:
                    questions.write_text(json.dumps({
                        "ready": {"type": "noul", "instructions": "证据是否完整？"},
                    }), encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_request(source, evidence, questions)
        self.client_factory.assert_not_called()

    def test_untranslated_payload_is_rejected_before_sdk_construction(self):
        mutations = {
            "nested_state": lambda request: request["state"].update({
                "history": [{"observations": [{"reason": "片上空间不足"}]}],
            }),
            "hardware": lambda request: request["state"].update({
                "hardware": {"name": "昇腾处理器", "memory": "片上缓存"},
            }),
            "instructions": lambda request: request["questions"]["ready"].update({
                "instructions": "选择最合适的融合方案",
            }),
            "escaped_json": lambda request: request["state"].update({
                "evidence_json": json.dumps({"summary": "需要更多数据"}),
            }),
        }
        for location, mutate in mutations.items():
            with self.subTest(location=location):
                request = copy.deepcopy(self.request)
                mutate(request)
                with self.assertRaises(ValueError):
                    evaluate(request, self.settings)
                self.client_factory.assert_not_called()

    def test_32k_pair_byte_guard_accepts_exact_limit_and_rejects_next_byte(self):
        # These are the application's byte limits, not exact provider token counts.
        request = copy.deepcopy(self.request)
        request["state"] = ""
        question_bytes = len(json.dumps(request["questions"], ensure_ascii=False).encode("utf-8"))
        remaining = self.settings.max_state_question_bytes - question_bytes - 2
        request["state"] = "中" * (remaining // 3) + "x" * (remaining % 3)
        accepted = check_budget(request, self.settings)
        self.assertEqual(accepted["state_plus_longest_question_bytes"], 28000)
        self.assertLess(accepted["request_bytes"], self.settings.max_request_bytes)
        self.assertEqual(accepted["method"], "conservative_utf8_json_bytes_not_exact_tokens")
        request["state"] += "x"
        with self.assertRaisesRegex(ValueError, "state \\+ longest question"):
            check_budget(request, self.settings)

    def test_64k_total_byte_guard_accepts_exact_limit_and_rejects_next_byte(self):
        request = {
            "model": self.settings.model,
            "state": "",
            "questions": {
                name: {"type": "noul", "instructions": "x" * length}
                for name, length in (("first", 18000), ("second", 18000), ("third", 0))
            },
        }
        overhead = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        request["questions"]["third"]["instructions"] = "x" * (56000 - overhead)
        accepted = check_budget(request, self.settings)
        self.assertEqual(accepted["request_bytes"], 56000)
        self.assertLess(accepted["state_plus_longest_question_bytes"], 28000)
        request["questions"]["third"]["instructions"] += "x"
        with self.assertRaisesRegex(ValueError, "request exceeds local byte budget"):
            check_budget(request, self.settings)

    def test_nan_and_infinity_in_evidence_are_rejected_before_sdk(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                request = copy.deepcopy(self.request)
                request["state"]["speedup"] = value
                with self.assertRaises(ValueError):
                    evaluate(request, self.settings)
        self.client_factory.assert_not_called()

    def test_either_budget_failure_never_constructs_sdk_or_sends_http(self):
        for settings in (
            replace(self.settings, max_state_question_bytes=1),
            replace(self.settings, max_request_bytes=1),
        ):
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, "byte budget"):
                    evaluate(self.request, settings)
        self.client_factory.assert_not_called()

    def test_sdk_response_preserves_native_answers_model_and_usage(self):
        request = copy.deepcopy(self.request)
        request["questions"].update({
            "direction": {"type": "choice", "instructions": "Which direction?", "criteria": {"memory": None, "tiling": None}},
            "readiness": {"type": "score", "instructions": "How complete?", "criteria": ["Low", "Medium", "High"]},
        })
        native = {
            "model": "jev-1.13.0",
            "answers": {
                "ready": {"type": "noul", "noul": 0.23},
                "direction": {"type": "choice", "choice": "memory", "probabilities": {"memory": 0.7, "tiling": 0.3}, "confidence": 0.81},
                "readiness": {"type": "score", "score": 1.4, "legend": {"0": "Low", "1": "Medium", "2": "High"}, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}, "confidence": 0.78},
            },
            "usage": {"input_tokens": 271, "output_tokens": 5},
        }
        result = Mock()
        result.model_dump.return_value = native
        sdk = self.client_factory.return_value.__enter__.return_value
        sdk.system_one.return_value = result
        self.assertEqual(evaluate(request, self.settings), native)
        self.client_factory.assert_called_once()
        self.assertEqual(self.client_factory.call_args.kwargs["api_key"], self.settings.api_key)
        self.assertEqual(self.client_factory.call_args.kwargs["model"], self.settings.model)
        sdk.system_one.assert_called_once_with(state=request["state"], questions=request["questions"])
        result.model_dump.assert_called_once_with(mode="json", exclude_none=True)
        self.assertNotIn("decision", native)
        self.assertNotIn("next_stage", native)

    def test_missing_extra_or_wrong_type_answers_are_rejected(self):
        sdk = self.client_factory.return_value.__enter__.return_value
        bad_responses = (
            {},
            {"answers": {}},
            {"answers": {"ready": {"type": "noul", "noul": 0.9}, "unknown": {"type": "noul", "noul": 0.8}}},
            {"answers": {"ready": {"type": "choice", "choice": "yes"}}},
        )
        for payload in bad_responses:
            with self.subTest(response=payload):
                result = Mock()
                result.model_dump.return_value = payload
                sdk.system_one.return_value = result
                with self.assertRaises(ValueError):
                    evaluate(self.request, self.settings)


if __name__ == "__main__":
    unittest.main()
