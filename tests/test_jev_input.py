"""English-input validation preserves source values and rejects invalid evidence."""

from copy import deepcopy
import unittest

from lib.jev_input import assert_english_payload


class JevInputTests(unittest.TestCase):
    def test_nested_json_values_numbers_and_identifiers_are_unchanged(self):
        value = {"model": "jev-1.13.0", "note": "HBM 128 MB; offsets -1 and +1; tolerance 1e-5",
                 "catalog": [{"id": "F1-F7", "shape": [1, 16], "offset": -1, "error": 1e-5,
                              "enabled": True, "unset": None, "labels": ["café ×2", "CUBE ≥ 1"]}]}
        before = deepcopy(value)
        self.assertIsNone(assert_english_payload(value))
        self.assertEqual(value, before)
        self.assertIs(value["catalog"][0]["enabled"], True)

    def test_language_marker_does_not_bypass_nested_script_validation(self):
        for text in ("融合", "日本語", "한글", "данные", "مرحبا"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "non-English"):
                assert_english_payload({"language": "en", "nested": [{"note": text}]})

    def test_escaped_non_english_text_is_rejected_without_mutation(self):
        values = [r"\u4e2d\u6587", r'{"note": "\u4e2d\u6587"}',
                  r'{"note": "\\u4e2d\\u6587"}', r"\u005cu4e2d\u005cu6587"]
        for text in values:
            value = {"note": text}
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "non-English"):
                assert_english_payload(value)
            self.assertEqual(value["note"], text)

    def test_keys_must_be_english_strings(self):
        for key in ("中文", 2, None, ("tuple",)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "keys"):
                assert_english_payload({key: "English"})

    def test_non_json_and_nonfinite_values_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), ("tuple",), {"set"}, b"bytes"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                assert_english_payload({"note": value})

    def test_windows_paths_and_literal_english_escapes_are_preserved(self):
        value = {"path": r"C:\work\new\kernel.py", "note": r"\u0048BM"}
        before = deepcopy(value)
        assert_english_payload(value)
        self.assertEqual(value, before)


if __name__ == "__main__":
    unittest.main()
