"""Validate English JSON evidence without rewriting its content."""

import json
import re
import unicodedata


_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")


def _decoded(text):
    # Inspect escaped JSON strings without decoding ordinary Windows paths.
    for _ in range(8):
        decoded = _ESCAPE.sub(lambda match: chr(int(match.group(1) or match.group(2), 16)), text)
        if decoded == text:
            return text
        text = decoded
    if _ESCAPE.search(text):
        raise ValueError("Jev text has excessive Unicode escape nesting.")
    return text


def _non_english(text):
    for char in _decoded(text):
        category = unicodedata.category(char)
        name = unicodedata.name(char, "")
        if (category.startswith("L") and not char.isascii() and "LATIN" not in name
                or "IDEOGRAPHIC" in name or category in {"Cs", "Co"}):
            return True
    return False


def assert_english_payload(value):
    """Require JSON values and English-script keys/text, leaving them unchanged.

    Latin technical names, finite numbers, punctuation and mathematical symbols
    are allowed. Script checking cannot establish the meaning or truth of prose.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or _non_english(key):
                raise ValueError("Jev JSON keys must use English identifiers.")
            assert_english_payload(item)
    elif isinstance(value, list):
        for item in value:
            assert_english_payload(item)
    elif isinstance(value, str):
        if _non_english(value):
            raise ValueError("Jev payload contains non-English text; provide English inputs.")
    elif value is not None and type(value) not in {int, float, bool}:
        raise ValueError("Jev payload must contain JSON-compatible values only.")
    if isinstance(value, float):
        json.dumps(value, allow_nan=False)
