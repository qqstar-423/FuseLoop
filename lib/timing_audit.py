"""Read-only kernel_details.csv diagnostics; never an operator performance gate.

Only dedicated CANN cache-clean/warmup records establish invocation boundaries.
Task IDs, kernel counts and timestamp gaps are deliberately not used to guess them.
"""

from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import io
import math
from pathlib import Path
from statistics import median


_TOOL_TOKENS = ("cannbenchcacheclean", "cannbenchwarmup")


def _key(value):
    return "".join(c for c in value.lower() if c.isalnum())


def _number(value, label, positive=False):
    try:
        result = Decimal(value.strip())
    except (AttributeError, InvalidOperation):
        raise ValueError(f"{label}: missing or invalid number") from None
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError(f"{label}: expected a finite {'positive' if positive else 'nonnegative'} number")
    # Keep JSON results finite even for syntactically valid Decimal inputs.
    if not math.isfinite(float(result)):
        raise ValueError(f"{label}: outside the supported numeric range")
    return result


def _json_number(value):
    result = float(value)
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError("metric outside the supported JSON numeric range")
    return result


def _summary(values):
    return {
        "mean": _json_number(sum(values) / len(values)),
        "median": _json_number(median(values)),
    }


def audit_kernel_csv(path, expected_repeats=3):
    """Return diagnostic metrics, or ``valid=False`` with explicit errors.

    Times must be in microseconds. Missing/ambiguous groups, malformed rows and
    multiple devices invalidate the audit. Missing device IDs and overlapping
    device events remain diagnosable, with explicit limitations; no returned
    metric is asserted to be complete operator end-to-end time.

    ``expected_repeats`` is a caller-supplied protocol fact, never inferred.
    An invalid parameter raises ValueError; data and file errors return a report.
    """
    if isinstance(expected_repeats, bool) or not isinstance(expected_repeats, int) or expected_repeats <= 0:
        raise ValueError("expected_repeats must be a positive integer")
    report = {
        "schema_version": "ascend-kernel-timing-audit/v2",
        "diagnostic_only": True,
        "valid": False,
        "status": "invalid",
        "source_csv": str(Path(path).resolve()),
        "expected_repeats": expected_repeats,
        "errors": [],
        "limitations": [
            "Kernel durations and the first-to-last device activity span are not complete operator end-to-end latency.",
            "Host work, launch overhead outside the device span and unrecorded work are not measured.",
            "The CSV alone does not establish candidate identity, report freshness or baseline compatibility.",
            "All non-tool kernels are included; names/types do not independently establish implementation ownership.",
        ],
    }
    try:
        raw = Path(path).read_bytes()
        report["source_sha256"] = hashlib.sha256(raw).hexdigest()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        columns = {_key(name): name for name in reader.fieldnames}
        if len(columns) != len(reader.fieldnames):
            raise ValueError("CSV has duplicate or ambiguous column names")
        for required in ("starttimeus", "durationus"):
            if required not in columns:
                raise ValueError(f"CSV is missing required column: {required}")
        if "name" not in columns and "type" not in columns:
            raise ValueError("CSV needs a Name or Type column")
        device_col = columns.get("deviceid") or columns.get("device")
        stream_col = columns.get("streamid") or columns.get("stream")
        groups = []
        devices = set()
        missing_devices = False
        row_count = 0
        # Profiler timestamps can be large. Arithmetic must not round away a
        # sub-microsecond kernel before subtracting its start timestamp.
        with localcontext() as context:
            context.prec = 80
            for row in reader:
                row_count += 1
                line = reader.line_num
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"line {line}: row does not match the CSV header")
                name = (row.get(columns.get("name"), "") or "").strip()
                kind = (row.get(columns.get("type"), "") or "").strip()
                name = name or kind
                if not name:
                    raise ValueError(f"line {line}: kernel name/type is empty")
                start = _number(row[columns["starttimeus"]], f"line {line} start")
                duration = _number(row[columns["durationus"]], f"line {line} duration", positive=True)
                end = start + duration
                if end - start != duration:
                    raise ValueError(f"line {line}: timestamp precision exceeds the supported range")
                device = (row.get(device_col, "") or "").strip()
                if device:
                    devices.add(device)
                else:
                    missing_devices = True
                if len(devices) > 1:
                    raise ValueError("CSV contains multiple devices; invocation grouping is ambiguous")
                event = {
                    "name": name, "type": kind, "start": start, "end": end,
                    "duration": duration, "line": line,
                    "stream": (row.get(stream_col, "") or "").strip(),
                }
                is_tool = any(token in (name + " " + kind).lower() for token in _TOOL_TOKENS)
                if is_tool:
                    if groups:
                        previous = groups[-1]
                        if not previous["events"]:
                            raise ValueError(f"line {line}: empty invocation between tool boundaries")
                        if start < max(item["end"] for item in previous["events"]):
                            raise ValueError(f"line {line}: tool boundary overlaps the previous invocation")
                    groups.append({"boundary": event, "events": []})
                else:
                    if not groups:
                        raise ValueError(f"line {line}: target work has no preceding dedicated tool boundary")
                    if start < groups[-1]["boundary"]["end"]:
                        raise ValueError(f"line {line}: target work precedes/overlaps its tool boundary")
                    groups[-1]["events"].append(event)

            report["csv_rows"] = row_count
            report["observed_repeats"] = len(groups)
            if not groups:
                raise ValueError("CSV contains no dedicated tool boundaries")
            if any(not group["events"] for group in groups):
                raise ValueError("CSV ends with an empty invocation")
            if len(groups) != expected_repeats:
                raise ValueError(f"expected {expected_repeats} complete invocations, found {len(groups)}")

            totals, spans = [], []
            by_name = defaultdict(list)
            signatures = set()
            repeats = []
            all_names = Counter()
            has_overlap = False
            has_multiple_streams = False
            for index, group in enumerate(groups, 1):
                events = group["events"]
                names = Counter(event["name"] for event in events)
                signatures.add(tuple(sorted(Counter((event["name"], event["type"]) for event in events).items())))
                per_name = defaultdict(Decimal)
                for event in events:
                    per_name[event["name"]] += event["duration"]
                    all_names[event["name"]] += 1
                for name, value in per_name.items():
                    by_name[name].append(value)
                total = sum((event["duration"] for event in events), Decimal(0))
                ordered = sorted(events, key=lambda event: event["start"])
                first, last = ordered[0]["start"], max(event["end"] for event in events)
                overlap = False
                furthest_end = ordered[0]["end"]
                for event in ordered[1:]:
                    overlap |= event["start"] < furthest_end
                    furthest_end = max(furthest_end, event["end"])
                streams = sorted({event["stream"] for event in events if event["stream"]})
                has_overlap |= overlap
                has_multiple_streams |= len(streams) > 1
                totals.append(total)
                spans.append(last - first)
                repeats.append({
                    "repeat_index": index,
                    "boundary_name": group["boundary"]["name"],
                    "boundary_csv_line": group["boundary"]["line"],
                    "kernel_count": len(events),
                    "kernel_counts": dict(sorted(names.items())),
                    "device_kernel_sum_us": _json_number(total),
                    "device_activity_span_us": _json_number(last - first),
                    "first_start_us_decimal": str(first),
                    "last_end_us_decimal": str(last),
                    "overlapping_device_events": overlap,
                    "stream_ids": streams,
                })
            if missing_devices:
                report["limitations"].append("Device identity is missing/incomplete; a shared device timeline cannot be independently verified.")
            if has_overlap:
                report["limitations"].append("Some device events overlap; summed durations double-count overlapping time and are not elapsed wall-clock time.")
            if has_multiple_streams:
                report["limitations"].append("Multiple streams are present; first-to-last device span does not establish full operator completion.")
            report.update({
                "valid": True,
                "status": "ok",
                "device_ids": sorted(devices),
                "device_identity_complete": not missing_devices,
                "repeats": repeats,
                "kernel_counts_all_repeats": dict(sorted(all_names.items())),
                "topology_changed_across_repeats": len(signatures) > 1,
                "metrics": {
                    "device_kernel_sum_us": _summary(totals),
                    "device_activity_span_us": _summary(spans),
                },
                "tool_compatible": {
                    "sum_of_per_kernel_medians_us": round(_json_number(sum((median(values) for values in by_name.values()), Decimal(0))), 2),
                    "per_kernel_medians_us": {name: round(_json_number(median(values)), 2) for name, values in sorted(by_name.items())},
                    "semantics": "Sum repeated launches per name within a group, then take the median only over groups where that name appears; round the final sum and per-name medians separately to two decimal places, as the external tool does. This is a tool-compatible diagnostic, not end-to-end time.",
                },
            })
    except (OSError, UnicodeError, csv.Error, ValueError, InvalidOperation, OverflowError) as exc:
        report["errors"].append(str(exc))
        report["valid"] = False
        report["status"] = "invalid"
        # Never leave a partial aggregate behind after a data/range failure.
        report.pop("metrics", None)
        report.pop("tool_compatible", None)
    return report
