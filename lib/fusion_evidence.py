"""Bind a developer's fusion choice and self-tests to the evaluated source tree.

This module records evidence, never performance winners or routing decisions.
Missing or stale evidence is ineligible, without interrupting the existing
compile/precision failure routes.
"""

from copy import deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path


SCHEMA_VERSION = 1
DECISION_FILE = "fusion_scheme_rationale.md"
_IGNORED_DIRS = {".git", "__pycache__", "build", "dist"}
log = logging.getLogger("triton-ascend-workflow")


def implementation_hash(impl_dir):
    """Hash source/configuration paths and bytes, excluding generated caches."""
    directory = Path(impl_dir)
    if not directory.is_dir():
        raise ValueError("Implementation directory is missing.")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(directory.rglob("*"), key=lambda p: p.relative_to(directory).as_posix()):
        relative = path.relative_to(directory)
        if (any(part in _IGNORED_DIRS or part.endswith(".egg-info") for part in relative.parts)
                or path.suffix in {".pyc", ".pyo"} or not path.is_file()):
            continue
        if path.is_symlink():
            raise ValueError("Implementation evidence cannot use symlinked source files.")
        name, data = relative.as_posix().encode("utf-8"), path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        count += 1
    if not count:
        raise ValueError("Implementation has no source files.")
    return digest.hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _initial_library_hash(path):
    library = _read_json(path)
    candidates = _candidate_map(library.get("candidates"), original=True)
    projection = {"schema_version": library.get("schema_version"), "candidates": [
        {key: candidate.get(key) for key in ("rank", "method", "probability")}
        for candidate in candidates.values()
    ]}
    return hashlib.sha256(json.dumps(projection, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _legacy_without_initial(work):
    return not (work / "fusion").exists() and not (work / "fusion_requirements.en.json").exists()


def _initial_library(work):
    if _legacy_without_initial(work):
        return None, "legacy-unscored-fusion-library/v1"
    path = _within(work / "fusion/fusion_library.json", work)
    return _read_json(path), _initial_library_hash(path)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _strings(value):
    return isinstance(value, list) and bool(value) and all(_text(item) for item in value)


def _within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Evidence files must stay within the workflow directory.")
    if not path.is_file() or not path.read_bytes().strip():
        raise ValueError(f"Evidence file is missing or empty: {path.name}")
    return path


def _candidate_map(candidates, *, original):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Fusion library needs a nonempty candidates list.")
    result = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("method"), dict):
            raise ValueError("Fusion candidate requires a method object.")
        method, probability = candidate["method"], candidate.get("probability")
        identity = method.get("id")
        if not _text(identity) or not _text(method.get("name")) or identity in result:
            raise ValueError("Fusion method IDs must be unique and have a name.")
        if "probability" not in candidate or (probability is None and original):
            raise ValueError("Initial Jev probabilities must be preserved.")
        if probability is not None and (isinstance(probability, bool)
                or not isinstance(probability, (int, float)) or not math.isfinite(probability)
                or not 0 <= probability <= 1):
            raise ValueError("Fusion probability must be null or a finite number in [0, 1].")
        result[identity] = candidate
    return result


def _validate_library(library, initial):
    if not isinstance(library, dict) or type(library.get("schema_version")) is not int or library["schema_version"] != 1:
        raise ValueError("Round fusion library requires schema_version=1.")
    if initial is not None and not isinstance(initial, dict):
        raise ValueError("Initial Jev library is missing.")
    json.dumps(library, allow_nan=False)
    json.dumps(initial, allow_nan=False)
    originals = {} if initial is None else _candidate_map(initial.get("candidates"), original=True)
    candidates = _candidate_map(library.get("candidates"), original=False)
    if not originals.keys() <= candidates.keys():
        raise ValueError("Round library must retain all initial Jev candidates.")
    for identity, candidate in candidates.items():
        if identity in originals:
            if (candidate["method"] != originals[identity]["method"]
                    or candidate["probability"] != originals[identity]["probability"]):
                raise ValueError("Round library cannot rewrite initial methods or Jev probabilities.")
        elif candidate["probability"] is not None:
            raise ValueError("New fusion methods have no Jev score: probability must be null.")
    selection = library.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Round fusion library requires a selection object.")
    ids = selection.get("method_ids")
    if not _strings(ids) or len(set(ids)) != len(ids) or not set(ids) <= candidates.keys():
        raise ValueError("Selection method_ids must name existing, unique candidates.")
    for field in ("implementation_plan", "reason"):
        if not _text(selection.get(field)):
            raise ValueError(f"Fusion selection requires {field}.")
    for field in ("target_cases", "actual_changes", "expected_benefits"):
        if not _strings(selection.get(field)):
            raise ValueError(f"Fusion selection requires a nonempty {field} list.")
    return selection["implementation_plan"], {
        **deepcopy(selection), "methods": [deepcopy(candidates[identity]) for identity in ids],
    }


def _validate_self_test(result, develop, work):
    if not isinstance(result, dict) or type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise ValueError("Self-test result requires schema_version=1.")
    provided, continuous = result.get("provided_cases"), result.get("continuous_calls")
    if not isinstance(provided, dict) or not isinstance(continuous, dict):
        raise ValueError("Self-tests must include provided_cases and continuous_calls.")
    for entry in (provided, continuous):
        if entry.get("executed") is not True or entry.get("passed") is not True:
            raise ValueError("Provided-case and consecutive-call self-tests must actually pass.")
    if (type(provided.get("total")) is not int or provided["total"] <= 0
            or type(provided.get("passed_cases")) is not int or provided["passed_cases"] != provided["total"]):
        raise ValueError("Every provided self-test case must pass.")
    if (continuous.get("same_shape") is not True
            or continuous.get("reference_checked_each_call") is not True
            or type(continuous.get("call_count")) is not int or continuous["call_count"] < 2):
        raise ValueError("Consecutive calls must keep shapes and check each result against the reference.")
    changes = continuous.get("changes")
    if not isinstance(changes, dict):
        raise ValueError("Consecutive calls require input/weight/bias change records.")
    applicable = 0
    for name in ("inputs", "weights", "bias"):
        entry = changes.get(name)
        if not isinstance(entry, dict) or type(entry.get("applicable")) is not bool:
            raise ValueError(f"Self-test must explicitly describe {name} applicability.")
        if entry["applicable"]:
            applicable += 1
            if entry.get("changed") is not True or entry.get("passed") is not True:
                raise ValueError(f"Self-test must validate fresh {name} values.")
        elif not _text(entry.get("reason")):
            raise ValueError(f"Inapplicable {name} needs a reason backed by the test log.")
    if not applicable:
        raise ValueError("Consecutive-call testing must change at least one applicable argument.")
    paths = {}
    for name, entry in (("provided", provided), ("continuous", continuous)):
        if not _text(entry.get("evidence_path")):
            raise ValueError("Self-test evidence_path must identify an actual nonempty log.")
        path = Path(entry["evidence_path"])
        paths[f"selftest_log_{name}"] = str(_within(path if path.is_absolute() else develop / path, work))
    return paths


def _invalid(reason):
    return {"eligible": False, "reason": str(reason), "implementation_plan": "",
            "fusion_scheme": {}, "impl_sha256": "", "evidence_paths": {}}


def _write_binding(work, develop, stage, result):
    pointer = work / "selection/current_implementation.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "stage": stage,
                                    "develop_dir": str(develop), **result},
                                   ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(pointer)


def _revision(path):
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "size": stat.st_size, "inode": stat.st_ino, "sha256": _sha(path)}


def _path_key(path):
    return os.path.normcase(str(Path(path).resolve()))


def begin_development(work_dir, develop_dir):
    """Invalidate the old pointer and capture artifacts before a developer runs."""
    work, develop = Path(work_dir).resolve(), Path(develop_dir).resolve()
    token = {"work_dir": str(work), "develop_dir": str(develop), "files": {}}
    _write_binding(work, develop, None, _invalid("Development is pending; previous evidence cannot be reused."))
    try:
        if not develop.is_relative_to(work):
            raise ValueError("Development directory must stay inside the workflow directory.")
        paths = {path.resolve() for path in develop.rglob("*") if path.is_file()}
        prior_result = develop / "self_test_result.json"
        if prior_result.is_file():
            try:
                result = _read_json(prior_result)
            except (OSError, ValueError):
                result = {}  # The malformed file itself is still revision-tracked.
            if isinstance(result, dict):
                for field in ("provided_cases", "continuous_calls"):
                    entry = result.get(field)
                    if isinstance(entry, dict) and _text(entry.get("evidence_path")):
                        path = Path(entry["evidence_path"])
                        path = (path if path.is_absolute() else develop / path).resolve()
                        if path.is_file():
                            paths.add(path)
        for path in paths:
            if not path.is_relative_to(work):
                raise ValueError("Existing development evidence points outside the workflow directory.")
            token["files"][_path_key(path)] = _revision(path)
    except (OSError, ValueError, TypeError) as exc:
        token["error"] = str(exc)
    return token


def _assert_updated(paths, previous, work, develop):
    if previous is None:
        return  # Compatibility for callers which have not opted into a run boundary.
    if (not isinstance(previous, dict) or previous.get("error")
            or previous.get("work_dir") != str(work) or previous.get("develop_dir") != str(develop)
            or not isinstance(previous.get("files"), dict)):
        raise ValueError("Development artifact revisions could not be verified.")
    for path in paths.values():
        old = previous["files"].get(_path_key(path))
        if old is not None and old == _revision(Path(path)):
            raise ValueError(f"Stale development evidence was not updated by this run: {Path(path).name}")


def finalize_development(work_dir, develop_dir, stage, agent_ok=True, previous_revisions=None):
    """Bind returned artifacts to current source, without aborting failure routes."""
    work, develop = Path(work_dir).resolve(), Path(develop_dir).resolve()
    try:
        if agent_ok is not True:
            raise ValueError("Development agent did not complete successfully.")
        if stage not in (2, 3) or not develop.is_relative_to(work):
            raise ValueError("Development evidence requires Stage2/3 inside the work directory.")
        names = {"fusion_library": "fusion_library.json",
                 "decision_rationale": "design_rationale.md" if stage == 2 else DECISION_FILE}
        paths = {key: str(_within(develop / name, work)) for key, name in names.items()}
        initial, initial_hash = _initial_library(work)
        plan, scheme = _validate_library(_read_json(paths["fusion_library"]), initial)
        result = {"eligible": False, "reason": "Self-tests have not been validated.",
                  "implementation_plan": plan, "fusion_scheme": scheme,
                  "impl_sha256": implementation_hash(work / "impl"), "evidence_paths": paths,
                  "initial_library_sha256": initial_hash}
        try:
            for key in ("self_test_report", "self_test_result"):
                extension = ".md" if key == "self_test_report" else ".json"
                paths[key] = str(_within(develop / (key + extension), work))
            paths.update(_validate_self_test(_read_json(paths["self_test_result"]), develop, work))
            result.update(eligible=True, reason="Source, fusion decision and self-tests are bound.")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # Reviewers still need the current choice when correctness is unproven.
            result["reason"] = str(exc)
        # A rerun must produce fresh decisions and test artifacts before they can
        # be associated with new code. Never rebind untouched historical files.
        _assert_updated(paths, previous_revisions, work, develop)
        result["document_sha256"] = {key: _sha(path) for key, path in paths.items()}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = _invalid(exc)
    _write_binding(work, develop, stage, result)
    if result["eligible"]:
        log.debug("fusion evidence: Stage%s bound this round\'s scheme and self-tests, directory=%s, code SHA256=%s",
                  stage, develop, result["impl_sha256"])
    else:
        log.debug("fusion evidence: Stage%s is not yet eligible for the best implementation library, directory=%s, reason=%s",
                  stage, develop, result["reason"])
    return result


def load_evidence(work_dir):
    """Revalidate the bound version; later build-time code edits invalidate it."""
    work = Path(work_dir).resolve()
    try:
        result = _read_json(work / "selection/current_implementation.json")
        if not isinstance(result, dict) or not result.get("impl_sha256"):
            return _invalid(result.get("reason", "No valid development evidence.") if isinstance(result, dict) else "Invalid evidence pointer.")
        if result.get("impl_sha256") != implementation_hash(work / "impl"):
            raise ValueError("Implementation changed after self-testing; rerun developer self-tests.")
        paths, hashes = result.get("evidence_paths"), result.get("document_sha256")
        if not isinstance(paths, dict) or not isinstance(hashes, dict) or set(paths) != set(hashes):
            raise ValueError("Incomplete evidence document hashes.")
        for key, path in paths.items():
            if _sha(_within(path, work)) != hashes[key]:
                raise ValueError(f"Bound evidence document changed: {key}.")
        initial, initial_hash = _initial_library(work)
        if initial_hash != result.get("initial_library_sha256"):
            raise ValueError("Initial Jev library changed after development.")
        # Re-read the bound artifacts; do not trust a hand-edited pointer summary.
        plan, scheme = _validate_library(_read_json(paths["fusion_library"]), initial)
        if plan != result.get("implementation_plan") or scheme != result.get("fusion_scheme"):
            raise ValueError("Evidence summary differs from the bound developer output.")
        try:
            _within(paths["self_test_report"], work)
            _validate_self_test(_read_json(paths["self_test_result"]), Path(result["develop_dir"]), work)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result.update(eligible=False, reason=str(exc))
        return result
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _invalid(exc)


def restore_imported_evidence(work_dir, manifest):
    """Restore only a matching developer binding, never old performance state.

    The importer may relocate paths in copied documents. Its manifest records
    both the original and relocated hashes; verify both before treating those
    historical self-tests as evidence for the copied, unchanged implementation.
    """
    work = Path(work_dir).resolve()
    develop = work / "develop/iter0"
    context = {"source_work_dir": "", "source_impl_dir": "",
               "source_develop_dir": "", "source_stage": None,
               "manifest_path": "init_impl_manifest.json", "optimize_hint": ""}
    stage = None
    try:
        if not isinstance(manifest, dict):
            raise ValueError("Imported development manifest is missing or invalid.")
        for key in ("source_work_dir", "source_impl_dir", "source_develop_dir"):
            if _text(manifest.get(key)):
                context[key] = manifest[key]
        if _text(manifest.get("optimize_hint")):
            context["optimize_hint"] = manifest["optimize_hint"]
        source = manifest.get("source_development_binding")
        if not isinstance(source, dict) or source.get("eligible") is not True:
            raise ValueError("The imported source has no eligible developer self-test binding.")
        stage = source.get("stage")
        context["source_stage"] = stage
        if type(stage) is not int or stage not in (2, 3):
            stage = None
            raise ValueError("Imported evidence must come from Stage2 or Stage3.")
        if not all(_text(context[key]) for key in
                   ("source_work_dir", "source_impl_dir", "source_develop_dir")):
            raise ValueError("The imported source directories are incomplete.")
        source_work = Path(context["source_work_dir"]).resolve()
        source_develop = Path(context["source_develop_dir"]).resolve()
        if (not source_develop.is_relative_to(source_work / "develop")
                or not _text(source.get("develop_dir"))
                or _path_key(source["develop_dir"]) != _path_key(source_develop)):
            raise ValueError("The copied developer directory does not match its original binding.")
        if source.get("impl_sha256") != implementation_hash(work / "impl"):
            raise ValueError("Imported implementation differs from the source self-tested version.")
        _, initial_hash = _initial_library(work)
        if source.get("initial_library_sha256") != initial_hash:
            raise ValueError("Imported Jev candidates differ from the developer's original evidence.")
        paths, hashes = source.get("evidence_paths"), source.get("document_sha256")
        if (not isinstance(paths, dict) or not paths or not isinstance(hashes, dict)
                or set(paths) != set(hashes)):
            raise ValueError("Original developer evidence hashes are incomplete.")
        copied = manifest.get("copied_files")
        if not isinstance(copied, list):
            raise ValueError("The imported file checksum list is missing.")
        by_source = {}
        for item in copied:
            if not isinstance(item, dict) or not _text(item.get("source")):
                raise ValueError("The imported file checksum list contains an invalid entry.")
            key = _path_key(item["source"])
            if key in by_source:
                raise ValueError("The imported file checksum list contains duplicate sources.")
            by_source[key] = item
        relocated = {}
        for name, original in paths.items():
            if not _text(original) or not Path(original).resolve().is_relative_to(source_develop):
                raise ValueError("Developer evidence points outside the copied developer directory.")
            item = by_source.get(_path_key(original))
            if not item or item.get("source_sha256") != hashes[name]:
                raise ValueError(f"Original developer evidence changed before import: {name}.")
            relative = item.get("destination")
            if not _text(relative) or Path(relative).is_absolute():
                raise ValueError(f"Imported evidence destination is invalid: {name}.")
            target = _within(work / relative, work)
            expected = develop / Path(original).resolve().relative_to(source_develop)
            if target != expected.resolve() or not target.is_relative_to(develop):
                raise ValueError(f"Imported evidence left its developer directory: {name}.")
            if item.get("sha256") != _sha(target):
                raise ValueError(f"Copied developer evidence changed after import: {name}.")
            relocated[name] = str(target)
        # This call is safe only after the source/code/checksum gates above.
        # It also rechecks the relocated self-test log paths and fusion schema.
        result = finalize_development(work, develop, stage)
        if not result.get("eligible"):
            raise ValueError(f"Copied developer self-tests are invalid: {result.get('reason', '')}")
        if result.get("impl_sha256") != source.get("impl_sha256"):
            raise ValueError("Imported implementation changed while restoring its evidence.")
        if result.get("evidence_paths") != relocated:
            raise ValueError("Relocated developer evidence differs from the original binding.")
        # Use the importer's exact path-only mapping: prose may legitimately
        # refer to the implementation or developer directory that just moved.
        from lib.init_impl import _rebaser
        rebase = _rebaser(source_work, Path(context["source_impl_dir"]).resolve(), work, source_develop)
        if (result.get("implementation_plan") != rebase(source.get("implementation_plan"))
                or result.get("fusion_scheme") != rebase(source.get("fusion_scheme"))):
            raise ValueError("Copied fusion decisions differ from the original binding.")
        result["reason"] = "Imported unchanged implementation and developer evidence were verified."
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = _invalid(exc)
    context["restored"] = result.get("eligible") is True
    result["imported_context"] = context
    _write_binding(work, develop, stage, result)
    checked = load_evidence(work)
    if result["eligible"] and not checked.get("eligible"):
        result = {**_invalid(checked["reason"]), "imported_context": {**context, "restored": False}}
        _write_binding(work, develop, stage, result)
    log.log(logging.INFO if result["eligible"] else logging.WARNING,
            "[emergency import evidence] %s; source development directory=%s; new development directory=%s; code and self-test eligibility=%s; reason=%s",
            "restored the development evidence for the corresponding code" if result["eligible"] else "kept the development material for reference only",
            context["source_develop_dir"], develop, result["eligible"], result["reason"])
    return result


def _imported_evidence_prompt(work_dir, result):
    """Describe imported documents even when their old passing result is stale."""
    from lib.prompt_files import file_hint

    work = Path(work_dir).resolve()
    try:
        pointer = _read_json(work / "selection/current_implementation.json")
        context = pointer.get("imported_context") if isinstance(pointer, dict) else None
    except (OSError, ValueError, TypeError):
        return ""
    if not isinstance(context, dict):
        return ""
    text = ("\n\n## Emergency-imported development material\n"
            f"Source implementation: {context.get('source_impl_dir', '')}; "
            f"source development directory: {context.get('source_develop_dir', '')}.\n"
            "This is a new work directory; old evaluations, best versions, the history ledger and human feedback were not inherited.\n")
    if result.get("eligible"):
        text += "The corresponding code and original development evidence were verified as consistent; this work must still re-run build, precision and performance evaluation before a best record can be produced.\n"
    else:
        text += ("The original self-test results are historical reference only and must not be treated as the current code passing; this round can still evaluate the specified implementation first."
                 "Stage9 should arrange for Stage3 to verify the current fusion scheme and actually run the missing tests, then re-evaluate once evidence is complete; "
                 "performance results before that are excluded from the best record and stagnation window.\n")
    if _text(context.get("optimize_hint")):
        text += ("Follow-up optimization direction given at startup (measure first in the first round; it is not an approved P0; "
                 "Stage9 should form concrete suggestions after analyzing this round's evidence):\n"
                 + context["optimize_hint"] + "\n")
    descriptions = {
        "design_rationale.md": ("Design rationale of the imported implementation", "Understand the algorithm and dataflow; compare with the current code and do not assume it still fully applies"),
        DECISION_FILE: ("Fusion selection rationale of the imported implementation", "Look at the fusion selection and reasons; when eligibility has lapsed it is historical reference only"),
        "fusion_library.json": ("Fusion scheme library of the imported development round", "Look at the actual selection and initial Jev probabilities; do not confuse it with the initial candidate library"),
        "self_test_report.md": ("Self-test report of the imported development round", "Look at the original test scope; whether it supports the current code is decided by program binding verification"),
        "self_test_result.json": ("Self-test results of the imported development round", "Verify the actual execution and repeated-call records; historical passing must not be rewritten as new passing"),
    }
    for filename, description in descriptions.items():
        path = work / "develop/iter0" / filename
        if path.is_file():
            text += file_hint(work, path, *description)
    manifest = work / "init_impl_manifest.json"
    if manifest.is_file():
        text += file_hint(work, manifest, "Source and per-file verification manifest of this import",
                          "Verify the source development round, copy scope and source/target checksums; this is not evaluation history")
    return text


def development_prompt(work_dir, develop_dir, stage):
    from lib.prompt_files import file_hint

    work, develop = Path(work_dir).resolve(), Path(develop_dir).resolve()
    rationale = "design_rationale.md" if stage == 2 else DECISION_FILE
    legacy = _legacy_without_initial(work)
    initial_hint = (
        "This is a legacy task not yet integrated with Stage1.5: there is no fusion directory or fusion_requirements.en.json; "
        "keep the original routing and do not call Jev. Build this round's candidates from the current implementation and evidence; every probability must be null.\n"
        if legacy else file_hint(work, work / "fusion/fusion_library.json", "Read-only initial Jev fusion candidate library",
                                 "Look at the methods and probabilities in candidates; the original methods and probabilities must be preserved")
    )
    candidate_hint = ("candidates record unscored schemes; every probability must be null; "
                      if legacy else "candidates keep the initial library's complete candidates, methods and original Jev probabilities; ")
    example = {"schema_version": 1, "provided_cases": {
        "executed": True, "passed": True, "total": 1, "passed_cases": 1, "evidence_path": "self_test.log"},
        "continuous_calls": {"executed": True, "passed": True, "same_shape": True, "call_count": 2,
            "reference_checked_each_call": True, "evidence_path": "self_test.log", "changes": {
                "inputs": {"applicable": True, "changed": True, "passed": True},
                "weights": {"applicable": False, "reason": "fill in only when the interface truly has no such parameter, and provide evidence in the log"},
                "bias": {"applicable": False, "reason": "fill in only when the interface truly has no such parameter, and provide evidence in the log"}}}}
    return (
        f"\n\n## This round's fusion selection and self-test evidence (Stage{stage})\n"
        + initial_hint +
        f"This round's output directory: {develop}. Output fusion_library.json, {rationale}, self_test_report.md, "
        "self_test_result.json and the actual test logs. Write the scheme selection rationale separately from the test results.\n"
        + file_hint(work, develop / "fusion_library.json", "This round's actual fusion selection and implementation scheme",
                    "Fill in selection's method_ids, implementation_plan and selection reasons; distinguish initial candidates from newly added methods")
        + file_hint(work, develop / rationale, "This round's fusion scheme selection rationale",
                    "Explain what was chosen, why and which cases it affects; the analysis node checks the scheme against this, and self-test reports are not a substitute")
        + file_hint(work, develop / "self_test_report.md", "This round's self-test process and results",
                    "Describe the execution results for the given cases and same-shape repeated calls separately, pointing at the actual test logs")
        + file_hint(work, develop / "self_test_result.json", "Structured result the program uses to validate this round's self-test eligibility",
                    "Fill in provided_cases, continuous_calls and evidence_path following the structure below; unexecuted tests must not be marked as passing")
        + file_hint(work, develop / "self_test.log", "Example location of the raw self-test log",
                    "Save the real execution output; the filename may differ, but self_test_result.json's evidence_path must point at the real log")
        +
        "This round's fusion_library.json format: schema_version=1; " + candidate_hint +
        "This round's attempts may be recorded in attempts; when appending new methods the probability must be null and you must not score them yourself. selection must contain "
        "method_ids (array of actually selected candidate IDs, combinable), implementation_plan (implementation scheme text), reason (selection rationale), "
        "target_cases, actual_changes, expected_benefits (the last three are nonempty arrays of text). Explain why the highest probability was not chosen or why an old scheme was kept.\n"
        "Finish the code first, then run every given case; for same-shape repeated calls change applicable parameters such as inputs, weights and bias, comparing each call against the reference implementation "
        "to confirm no stale values are reused. Inapplicable parameters must be explained with log evidence; already-passing cases cannot substitute for repeated calls."
        "If code changes after testing, retest. Write false honestly for unexecuted or failed items; do not copy the example's true.\n"
        "When resuming or rerunning the same round, regenerate this round's selection, rationale, self-test results and real logs; the program rejects stale unupdated files.\n"
        "self_test_result.json structure example (counts and results must be replaced with measured values; evidence_path may be a log path relative to this directory):\n"
        + json.dumps(example, ensure_ascii=False, indent=2) + "\n"
        "The program binds code and document hashes after the agent returns; an implementation with incomplete self-tests or records still follows the original build/precision flow but cannot enter the best implementation library.\n"
    )


def format_evidence_for_prompt(work_dir):
    from lib.prompt_files import file_hint

    result = load_evidence(work_dir)
    imported_hint = _imported_evidence_prompt(work_dir, result)
    if not result.get("impl_sha256") or not result.get("fusion_scheme"):
        return imported_hint + f"\n\n## Fusion selection evidence for the current implementation\nNot eligible for the best implementation library: {result['reason']}. Do not attach old rationale/old self-tests to the current code; Stage3 must update and retest.\n"
    paths = result["evidence_paths"]
    rationale = Path(paths["decision_rationale"]).read_text(encoding="utf-8-sig")
    descriptions = {
        "decision_rationale": ("Fusion selection rationale for the current code", "Look first at the chosen scheme, why others were not chosen and the target cases, then verify against measurements"),
        "fusion_library": ("This round's actual fusion selection and scheme library", "Look at selection and newly added methods; the original Jev probability is only a prior reference"),
        "self_test_report": ("Self-test report for the current code", "Check whether the given cases and repeated calls actually ran; distinguish failed, unexecuted and passed"),
        "self_test_result": ("Self-test JSON used by the program's validation", "Look at provided_cases and continuous_calls pass statuses and evidence_path"),
        "selftest_log_provided": ("Raw self-test log for the given cases", "Verify whether the actual commands, case counts and results support the self-test report"),
        "selftest_log_continuous": ("Raw self-test log for repeated calls", "Verify the per-call results against the reference implementation for the same shape with changed parameters"),
    }
    hints = "".join(file_hint(work_dir, path, *descriptions.get(
        key, ("Additional evidence for the current implementation", "Cross-check against the fusion selection and self-test results"))) for key, path in paths.items())
    return (imported_hint + "\n\n## Fusion selection rationale bound to the current code\n"
            f"Implementation SHA256: {result['impl_sha256']}\n"
            + hints +
            f"Best implementation library eligibility: {result['eligible']}; {result['reason']}\n"
            "This round's library is distinct from the read-only initial Jev library; real evaluation takes priority over initial probabilities.\n"
            + json.dumps(result["fusion_scheme"], ensure_ascii=False, indent=2) + "\n\n" + rationale + "\n")
