"""Shared Stage9 performance-experience contract, without I/O or data repair.

Schema, empty worksheets and validation use the same field definitions. The
model supplies conclusions; validation reports every independently checkable
missing or invalid field without inventing a replacement.
"""

import json


_WHY_FIELDS = {"proven_pattern": "why_it_worked", "regression_pattern": "why_it_failed"}
_DETAIL_FIELDS = {
    "applicability": "text",
    "next_action": "text",
    "evidence": "evidence",
    "case_analysis": "cases",
}
_CASE_FIELDS = ("case_id", "observation", "explanation")


class PerformancePatternValidationError(ValueError):
    """All field errors from one performance-experience object."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _fields(name):
    fields = {}
    why = _WHY_FIELDS.get(name)
    if why:
        fields.update({"what_changed": "text", why: "text", "fusion_related": "boolean"})
    fields.update(_DETAIL_FIELDS)
    return fields


def _schema(kind):
    if kind == "text":
        return {"type": "string", "minLength": 1, "pattern": r"\S"}
    if kind == "boolean":
        return {"type": "boolean"}
    if kind == "evidence":
        return {"anyOf": [_schema("text"), {"type": "object", "minProperties": 1},
                          {"type": "array", "minItems": 1}]}
    if kind == "cases":
        return {"type": "array", "minItems": 1, "items": {
            "type": "object", "properties": {key: _schema("text") for key in _CASE_FIELDS},
            "required": list(_CASE_FIELDS),
        }}
    raise ValueError(f"Unknown performance-pattern field kind: {kind}")


def build_performance_pattern_schema(name):
    """Return a fresh schema; unspecified extra fields remain permitted."""
    fields = _fields(name)
    return {"type": "object", "properties": {key: _schema(kind) for key, kind in fields.items()},
            "required": list(fields)}


def build_performance_pattern_template(name):
    """Return an incomplete worksheet, never a fabricated valid experience."""
    result = {}
    for field, kind in _fields(name).items():
        if kind == "cases":
            result[field] = [{key: "" for key in _CASE_FIELDS}]
        elif kind == "boolean":
            # None prevents the worksheet from asserting a model conclusion.
            result[field] = None
        else:
            result[field] = ""
    return result


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_performance_pattern_details(pattern, name="performance_pattern"):
    """Validate all known fields and raise one error with every bad field path.

    The default name preserves the historical details-only contract. The concrete
    proven/regression names additionally require their respective basic fields.
    Evidence containers retain the existing shallow nonempty check.
    """
    if not isinstance(pattern, dict):
        raise PerformancePatternValidationError([f"{name} must be an object"])
    errors = []
    for field, kind in _fields(name).items():
        value = pattern.get(field)
        path = f"{name}.{field}"
        if kind == "text":
            if not _nonempty_text(value):
                errors.append(f"{path} must be a nonempty string")
        elif kind == "boolean":
            if type(value) is not bool:
                errors.append(f"{path} must be a boolean")
        elif kind == "evidence":
            if not (_nonempty_text(value) or isinstance(value, (dict, list)) and bool(value)):
                errors.append(f"{path} must identify supporting evidence: a nonempty string, object or array")
        elif kind == "cases":
            if not isinstance(value, list) or not value:
                errors.append(f"{path} must contain affected case analysis (a nonempty array)")
                continue
            for index, analysis in enumerate(value):
                item_path = f"{path}[{index}]"
                if not isinstance(analysis, dict):
                    errors.append(f"{item_path} must be an object")
                    continue
                case_id = analysis.get("case_id")
                # Escape newlines/control characters so diagnostics stay on one log line.
                context = (f"case_id={json.dumps(case_id, ensure_ascii=False)}"
                           if isinstance(case_id, str) else "case_id missing or invalid")
                for key in _CASE_FIELDS:
                    if not _nonempty_text(analysis.get(key)):
                        errors.append(f"{item_path}.{key} ({context}) must be a nonempty string")
    if errors:
        raise PerformancePatternValidationError(errors)
