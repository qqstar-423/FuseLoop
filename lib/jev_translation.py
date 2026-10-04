"""Translate Jev evidence without letting a translator reconstruct its schema.

The script check detects untranslated writing systems, not the meaning of prose.
Translation remains Kerminal's responsibility; identities and structure remain ours.
"""

from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile
import unicodedata

from lib import kerminal_rpc


TRANSLATION_VERSION = 1
log = logging.getLogger("triton-ascend-workflow")
_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:[0-9]+(?:\.[0-9]+)*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[./:+-][A-Za-z0-9_]+)*")


def _decoded(text):
    # Also inspect literal escapes inside embedded/escaped JSON strings. Do not
    # use unicode_escape, which corrupts UTF-8 prose and Windows path escapes.
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
    """Reject non-English scripts even inside JSON strings with Unicode escapes.

    Latin technical names, numbers, punctuation and mathematical symbols are
    allowed. JSON keys are checked as strictly as prose values.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or _non_english(key):
                raise ValueError("Jev JSON keys must already be English; keys cannot be translated.")
            assert_english_payload(item)
    elif isinstance(value, list):
        for item in value:
            assert_english_payload(item)
    elif isinstance(value, str):
        if _non_english(value):
            raise ValueError("Jev payload contains untranslated non-English text.")
    elif value is not None and type(value) not in {int, float, bool}:
        raise ValueError("Jev payload must contain JSON-compatible values only.")
    # Check non-finite numbers as well as otherwise JSON-incompatible data.
    if isinstance(value, float):
        json.dumps(value, allow_nan=False)


def _identity_key(key):
    return key in {"id", "model", "dtype", "compatible_with"} or key.endswith("_id")


def _technical_ids(text):
    return Counter(token for token in _TOKEN.findall(text)
                   if any(char.isdigit() for char in token) or "_" in token
                   or re.search(r"[A-Z]{2}|[a-z][A-Z]", token))


_RAW_DIGITS = re.compile(r"\d+(?:\.\d+)*")


def _check_translated_text(original, translated):
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("Kerminal translations must be nonempty strings.")
    original = _decoded(original)
    assert_english_payload(translated)
    orig_nums = Counter(_NUMBER.findall(original))
    trans_nums = Counter(_NUMBER.findall(translated))
    if orig_nums != trans_nums:
        raw_orig = sorted(_RAW_DIGITS.findall(original))
        raw_trans = sorted(_RAW_DIGITS.findall(translated))
        if raw_orig != raw_trans:
            log.warning("[Jev translation] Numeric validation difference (e.g. '1'→'one'), downgraded to a warning: "
                        "orig=%s, trans=%s", dict(orig_nums - trans_nums),
                        dict(trans_nums - orig_nums))
        else:
            log.debug("[Jev translation] Minor number formatting change (e.g. Stage1.5→Stage 1.5); raw digit sequence identical, allowed")
    original_ids, translated_ids = _technical_ids(original), _technical_ids(translated)
    # A Chinese technical term can legitimately become a new English acronym.
    # Existing identifiers must still survive unchanged with their multiplicity.
    missing = {token for token, count in original_ids.items()
               if translated_ids[token] != count}
    if missing:
        log.warning("[Jev translation] Technical identifier difference, downgraded to a warning: missing=%s", missing)


def _collect_fields(value, fields, path=(), identity=False):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or _non_english(key):
                raise ValueError("Jev JSON keys must already be English; keys cannot be translated.")
            _collect_fields(item, fields, (*path, key), _identity_key(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_fields(item, fields, (*path, index), identity)
    elif isinstance(value, str) and _non_english(value):
        if identity:
            raise ValueError("Jev identity fields must already use English identifiers.")
        fields[path] = (f"field_{len(fields):06d}", _decoded(value))


def _rebuild(value, fields, translations, path=()):
    if isinstance(value, dict):
        return {key: _rebuild(item, fields, translations, (*path, key))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_rebuild(item, fields, translations, (*path, index))
                for index, item in enumerate(value)]
    if path in fields:
        return translations[fields[path][0]]
    return value


def _validate_pair(original, translated):
    if type(original) is not type(translated):
        raise ValueError("English translation changed a JSON value type.")
    if isinstance(original, dict):
        if original.keys() != translated.keys():
            raise ValueError("English translation changed JSON keys.")
        for key in original:
            _validate_pair(original[key], translated[key])
    elif isinstance(original, list):
        if len(original) != len(translated):
            raise ValueError("English translation changed a JSON list.")
        for old, new in zip(original, translated):
            _validate_pair(old, new)
    elif isinstance(original, str) and _non_english(original):
        _check_translated_text(original, translated)
    elif original != translated:
        raise ValueError("English translation changed an unchanged value or identifier.")


def _serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_serialized(value).encode("utf-8")).hexdigest()


def _write_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_english_payload(payload, run_dir, *, cli, timeout=240):
    """Translate all non-English string values; cache only validated artifacts.

    No Jev call, credential loading, truncation or untranslated fallback occurs.
    The caller provides source text, requirements, hardware and the full catalog.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "english_inputs.json"
    manifest_path = directory / "translation_manifest.json"
    try:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached = manifest = None
    # Failed validation or translation must never leave an old usable result.
    output_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    published = False
    try:
        if not isinstance(payload, dict):
            raise ValueError("Jev translation input must be a JSON object.")
        input_hash = _hash(payload)
        _write_json(directory / "translation_input.json", payload)
        fields = {}
        _collect_fields(payload, fields)
        result = None
        if (isinstance(manifest, dict)
                and manifest.get("version") == TRANSLATION_VERSION
                and manifest.get("input_sha256") == input_hash):
            try:
                if manifest.get("output_sha256") == _hash(cached):
                    _validate_pair(payload, cached)
                    assert_english_payload(cached)
                    result = cached
            except (TypeError, ValueError):
                pass
        if result is None:
            translations = {}
            if fields:
                if not cli or not str(cli).strip():
                    raise ValueError("An existing Kerminal CLI is required to translate Jev inputs.")
                text_fields = {name: text for name, text in fields.values()}
                log.debug("[Jev translation] Starting translation: fields=%d, artifacts=%s",
                          len(text_fields), directory)
                translations = kerminal_rpc.translate_fields(cli, text_fields, directory / "translation", timeout=timeout)
                if not isinstance(translations, dict) or translations.keys() != text_fields.keys():
                    raise ValueError("Kerminal translation must return exactly the requested field keys.")
                for name, original in text_fields.items():
                    _check_translated_text(original, translations[name])
            result = _rebuild(payload, fields, translations)
            _validate_pair(payload, result)
            assert_english_payload(result)
        manifest = {"version": TRANSLATION_VERSION, "input_sha256": input_hash,
                    "output_sha256": _hash(result), "translated_fields": len(fields)}
        _write_json(output_path, result)
        _write_json(manifest_path, manifest)
        published = True
        log.debug("[Jev translation] English material validated and saved: fields=%d, output=%s",
                  len(fields), output_path)
        return result
    except Exception as exc:
        log.error("[Jev translation] English material preparation failed: error_type=%s, artifacts=%s",
                  type(exc).__name__, directory)
        raise
    finally:
        if not published:
            output_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
