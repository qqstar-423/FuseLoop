"""Import Stage1/1.5 inputs and matching development evidence for a build restart.

No provider, configuration, historical evaluation or best-version lookup is used.
The caller owns stage routing and whether old self-test evidence is still valid.
"""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import stat


MANIFEST = "init_impl_manifest.json"
_ROOT_FILES = ("ANALYSIS.md", "fusion_requirements.en.json", "device_info.json")
_REQUIRED = (*_ROOT_FILES, "fusion/fusion_library.json", "fusion/ranking.json",
             "fusion/jev_request.json", "fusion/jev_response.json")
_FUSION_FILES = {"fusion_library.json", "ranking.json", "jev_request.json", "jev_response.json", "english_inputs.json",
                 "translation/english_inputs.json"}
_DEVELOP_REQUIRED = ("fusion_library.json", "self_test_report.md", "self_test_result.json")
_IGNORED = {".git", "__pycache__", "build", "dist", ".pytest_cache"}
_EXCLUDED_DEVELOP = {"human_feedback.json", "question.md"}
_DEVICE_FIELDS = ("chip_model", "soc_version", "npu_arch", "ai_core_num", "cube_core_num",
                  "vector_core_num", "ub_size_kb", "l1_size_kb", "l0a_size_kb",
                  "l0b_size_kb", "l0c_size_kb", "l2_size_kb", "programming_model",
                  "driver_backend", "target_arch", "runtime_versions", "framework", "backend", "toolchain")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"--init-impl material cannot be read as JSON: {path}") from exc


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _linked(path):
    return path.is_symlink() or bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT) \
        if os.name == "nt" else path.is_symlink()


def _safe_path(path, root):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"--init-impl material path escapes the source directory: {path}")
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.exists() and _linked(parent):
            raise ValueError(f"--init-impl material must not pass through symbolic links or junctions: {path}")


def _files(directory, *, ignore_generated=False, ignored_dirs=None):
    """Do not follow symlinks, junctions or reparse points in imported trees."""
    if _linked(directory):
        raise ValueError(f"--init-impl cannot copy symbolic links or junctions: {directory}")
    ignored_dirs = _IGNORED if ignored_dirs is None else ignored_dirs
    for item in sorted(directory.iterdir()):
        if ignore_generated and (item.name in ignored_dirs or item.name.endswith(".egg-info")
                                 or item.suffix in {".pyc", ".pyo"}):
            continue
        if _linked(item):
            raise ValueError(f"--init-impl cannot copy symbolic links or junctions: {item}")
        if item.is_dir():
            yield from _files(item, ignore_generated=ignore_generated, ignored_dirs=ignored_dirs)
        elif item.is_file():
            yield item
        else:
            raise ValueError(f"--init-impl does not support special files: {item}")


def _task_hash(task_dir):
    directory = Path(task_dir).resolve(strict=True)  # task/ itself is normally a link.
    if not directory.is_dir():
        raise ValueError("--init-impl task directory does not exist.")
    digest = hashlib.sha256()
    count = 0
    for path in _files(directory, ignore_generated=True):
        if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".md", ".txt"}:
            continue
        digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
        count += 1
    if not count:
        raise ValueError("--init-impl task directory has no verifiable requirement/case/reference implementation files.")
    return digest.hexdigest()


def _source_work(source):
    for parent in source.parents:
        if ((parent / "ANALYSIS.md").is_file() or (parent / ".state.json").exists()
                or ((parent / "device_info.json").is_file()
                    and (parent / "fusion/fusion_library.json").is_file())):
            return parent
    raise ValueError("--init-impl must be inside an existing workflow work directory; Stage1/1.5/2 material was not found.")


def _scoring_inputs_relative(work):
    from lib.fusion_selection import PROTOCOL, LEGACY_PROTOCOL
    library_path = work / "fusion/fusion_library.json"
    _safe_path(library_path, work)
    library = _json(library_path)
    protocol = library.get("protocol") if isinstance(library, dict) else None
    if protocol == PROTOCOL:
        return "fusion/english_inputs.json"
    if protocol == LEGACY_PROTOCOL:
        return "fusion/translation/english_inputs.json"
    raise ValueError("--init-impl scoring protocol is unsupported; provide verified English scoring material.")


def _required(work, develop=None):
    for relative in (*_REQUIRED, _scoring_inputs_relative(work)):
        path = work / relative
        _safe_path(path, work)
        if not path.is_file() or not path.read_bytes().strip():
            raise ValueError(f"--init-impl is missing required material from completed Stage1/1.5/2: {relative}; these stages are not rerun automatically.")
    if develop is not None:
        for name in _DEVELOP_REQUIRED:
            path = develop / name
            _safe_path(path, work)
            if not path.is_file() or not path.read_bytes().strip():
                raise ValueError(f"--init-impl is missing the development material for the corresponding implementation: {path}")
        rationales = [develop / name for name in ("design_rationale.md", "fusion_scheme_rationale.md")]
        for path in rationales:
            _safe_path(path, work)
        if not any(path.is_file() and path.read_bytes().strip() for path in rationales):
            raise ValueError("--init-impl is missing the corresponding implementation\'s design/fusion selection rationale.")


def _device_signature(device):
    if not isinstance(device, dict) or not device.get("chip_model"):
        raise ValueError("--init-impl chip info lacks chip_model.")
    return {key: device[key] for key in _DEVICE_FIELDS if key in device}


def _validate_fusion(work, original_hashes=None, develop=None):
    """Check native probabilities and their source binding without loading Jev."""
    from lib.fusion_selection import (PROTOCOL, LEGACY_PROTOCOL, LEGACY_CAVEATS, SUPPORTED_PROTOCOLS,
                                      _build_request, _validate_catalog, _validate_requirements)
    from lib.fusion_evidence import _validate_library
    from lib.jev_input import assert_english_payload

    requirements = _json(work / "fusion_requirements.en.json")
    _validate_requirements(requirements)
    assert_english_payload(requirements)
    library, ranking = (_json(work / "fusion" / name) for name in ("fusion_library.json", "ranking.json"))
    response, request = (_json(work / "fusion" / name) for name in ("jev_response.json", "jev_request.json"))
    if not all(isinstance(value, dict) for value in (library, ranking, response, request)):
        raise ValueError("--init-impl fusion scoring material must be JSON objects.")
    protocol = library.get("protocol")
    if protocol not in SUPPORTED_PROTOCOLS or ranking.get("protocol") != protocol:
        raise ValueError("--init-impl scoring protocol is unsupported or inconsistent.")
    input_relative = _scoring_inputs_relative(work)
    inputs = _json(work / input_relative)
    if not isinstance(inputs, dict) or set(inputs) != {
            "fusion_methods_source", "operator_requirements", "hardware", "catalog"}:
        raise ValueError("--init-impl English input evidence is incomplete.")
    assert_english_payload(inputs)
    assert_english_payload(request)
    methods = _validate_catalog(inputs["catalog"])
    expected_request = _build_request(
        inputs["fusion_methods_source"], inputs["operator_requirements"], inputs["hardware"],
        methods, request.get("model"),
        catalog_context={key: value for key, value in inputs["catalog"].items() if key != "methods"})
    if protocol == LEGACY_PROTOCOL:
        expected_request["state"]["caveats"] = LEGACY_CAVEATS
    canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if canonical(request) != canonical(expected_request):
        raise ValueError("--init-impl English input evidence disagrees with the Jev request.")
    input_sha = _sha(json.dumps(inputs, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"))
    for document in (library, ranking):
        digest = document.get("english_inputs_sha256")
        if ((protocol == PROTOCOL or digest is not None) and digest != input_sha
                or not isinstance(document.get("english_inputs_path"), str)
                or Path(document["english_inputs_path"]).resolve() != (work / input_relative).resolve()):
            raise ValueError("--init-impl English input evidence path or digest disagrees with scoring.")
    assert_english_payload(ranking.get("candidates"))
    candidates, top_n = ranking.get("candidates"), library.get("top_n")
    if (ranking.get("schema_version") != 1 or library.get("schema_version") != 1
            or not isinstance(candidates, list) or not candidates or type(top_n) is not int
            or not 1 <= top_n <= len(candidates) or library.get("candidates") != candidates[:top_n]):
        raise ValueError("--init-impl fusion library disagrees with the full ranking/Top N.")
    answers, questions = response.get("answers"), request.get("questions")
    if not isinstance(answers, dict) or not isinstance(questions, dict):
        raise ValueError("--init-impl Jev raw request or response is missing options.")
    ids, previous = set(), float("inf")
    for index, candidate in enumerate(candidates, 1):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("method"), dict):
            raise ValueError("--init-impl fusion ranking entry is invalid.")
        method, probability = candidate["method"], candidate.get("probability")
        identity = method.get("id")
        if (not isinstance(identity, str) or not identity or identity in ids
                or not isinstance(method.get("name"), str) or not method["name"].strip()
                or type(candidate.get("rank")) is not int or candidate["rank"] != index
                or isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1 or probability > previous):
            raise ValueError("--init-impl fusion methods, ordering or probabilities are invalid.")
        answer = answers.get(identity)
        if (not isinstance(answer, dict) or answer.get("type") != "noul"
                or isinstance(answer.get("noul"), bool) or answer.get("noul") != probability):
            raise ValueError("--init-impl fusion probabilities disagree with the raw Jev response.")
        ids.add(identity)
        previous = probability
    if set(answers) != ids or set(questions) != ids:
        raise ValueError("--init-impl Jev request, response and full ranking options disagree.")
    response_sha = _sha(json.dumps(response, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"))
    for document in (library, ranking):
        if document.get("response_sha256") != response_sha:
            raise ValueError("--init-impl Jev response digests disagree.")
        sources = document.get("sources", {})
        if not isinstance(sources, dict) or not all(isinstance(value, dict) for value in sources.values()):
            raise ValueError("--init-impl Jev scoring source records are invalid.")
        for key, relative in (("stage1_analysis", "ANALYSIS.md"),
                              ("stage1_requirements", "fusion_requirements.en.json"),
                              ("hardware", "device_info.json")):
            expected = (original_hashes or {}).get(relative) or _sha((work / relative).read_bytes())
            if not isinstance(sources.get(key), dict) or sources[key].get("sha256") != expected:
                raise ValueError(f"--init-impl Jev scoring disagrees with the original input: {relative}")
        fingerprint = _sha(json.dumps({
            "protocol": document.get("protocol"), "request": request,
            "jev_config": document.get("jev_config"),
            "source_sha256": {key: value.get("sha256") for key, value in sources.items()},
        }, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"))
        if document.get("fingerprint_sha256") != fingerprint:
            raise ValueError("--init-impl Jev request and source digests disagree.")
    if library.get("fingerprint_sha256") != ranking.get("fingerprint_sha256"):
        raise ValueError("--init-impl the fusion library and full ranking come from different scorings.")
    if develop is not None:
        _validate_library(_json(develop / "fusion_library.json"), library)


def _rebaser(source_work, source_impl, work, source_develop):
    pairs = [(source_impl, work / "impl"), (source_work / "fusion", work / "fusion"),
             (source_develop, work / "develop/iter0"),
             (source_work / "task", work / "task")]
    pairs.extend((source_work / name, work / name) for name in _ROOT_FILES)
    replacements = {}
    for old, new in pairs:
        replacements[str(old)] = str(new)
        replacements[old.as_posix()] = new.as_posix()
    expression = re.compile("(" + "|".join(re.escape(old) for old in sorted(replacements, key=len, reverse=True))
                            + r")(?=$|[\\/\s\"'<>),;:#\]])")

    def rebase(value):
        if isinstance(value, str):
            return expression.sub(lambda match: replacements[match.group(1)], value)
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        return value
    return rebase


def _select_development(source_work, source):
    """Use a verifiable binding, never the highest iteration directory name."""
    from lib.fusion_evidence import implementation_hash
    pointer = source_work / "selection/current_implementation.json"
    if pointer.is_file() and not _linked(pointer):
        try:
            _safe_path(pointer, source_work)
            binding = _json(pointer)
            develop = Path(binding.get("develop_dir", "")).resolve()
            paths, hashes = binding.get("evidence_paths"), binding.get("document_sha256")
            if (binding.get("stage") in (2, 3) and develop.is_relative_to(source_work / "develop")
                    and develop.is_dir() and not _linked(develop)
                    and binding.get("impl_sha256") == implementation_hash(source)
                    and isinstance(paths, dict) and paths and isinstance(hashes, dict)
                    and set(paths) == set(hashes)
                    and all(Path(path).is_file() and Path(path).resolve().is_relative_to(source_work)
                            and not _linked(Path(path)) and _sha(Path(path).read_bytes()) == hashes[key]
                            for key, path in paths.items())):
                return develop, binding
        except (ValueError, OSError, TypeError, AttributeError):
            pass
    return source_work / "develop/iter0", None


def load_init_impl_manifest(work_dir):
    path = Path(work_dir) / MANIFEST
    if not path.exists():
        return None
    result = _json(path)
    if isinstance(result, dict) and result.get("status") == "preparing":
        raise ValueError("--init-impl the previous import did not finish; retry with the original --init-impl command into a new work directory; do not restore a half-copied directory.")
    if (not isinstance(result, dict) or result.get("schema_version") != 1
            or result.get("status") != "prepared" or result.get("skipped_stages") != [1, 1.5, 2]
            or not isinstance(result.get("copied_files"), list) or not result["copied_files"]):
        raise ValueError("--init-impl the import manifest is incomplete and cannot justify skipping stages.")
    return result


def prepare_init_impl(source_impl, work_dir, task_dir, *, optimize_hint="", log=None):
    """Copy only allowlisted inputs into a new work; refuse any overwrite."""
    log = log or logging.getLogger("triton-ascend-workflow")
    source = Path(source_impl).absolute()
    if not source.is_dir() or _linked(source):
        raise ValueError("--init-impl must point to an actual implementation directory, not a symbolic link.")
    source, work = source.resolve(), Path(work_dir).resolve()
    source_work = _source_work(source)
    if work.is_relative_to(source_work) or source_work.is_relative_to(work):
        raise ValueError("--init-impl the source and new work must not contain each other.")
    if (work / MANIFEST).exists() or (work / ".state.json").exists():
        raise ValueError("--init-impl can only import into a new work directory; existing task state must not be overwritten.")
    work.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "status": "preparing", "source_work_dir": str(source_work),
              "source_impl_dir": str(source), "work_dir": str(work), "optimize_hint": optimize_hint,
              "skipped_stages": [1, 1.5, 2], "copied_files": []}
    with (work / MANIFEST).open("xb") as output:
        output.write(_encoded(result))
    source_develop, binding = _select_development(source_work, source)
    _required(source_work, source_develop)
    device = _json(source_work / "device_info.json")
    _device_signature(device)
    previous_import = load_init_impl_manifest(source_work)
    if previous_import is not None:
        validate_init_impl_inputs(source_work, device, source_work / "task")
        fusion_input_hashes = previous_import.get("fusion_input_hashes") or {
            entry["destination"]: entry["source_sha256"]
            for entry in previous_import["copied_files"] if entry["destination"] in _ROOT_FILES}
    else:
        fusion_input_hashes = {name: _sha((source_work / name).read_bytes()) for name in _ROOT_FILES}
    _validate_fusion(source_work, fusion_input_hashes, develop=source_develop)
    task_sha = _task_hash(source_work / "task")
    if task_sha != _task_hash(task_dir):
        raise ValueError("--init-impl the original task disagrees with --task-dir\'s requirements/cases/reference implementation.")
    files = [(source_work / name, name) for name in _ROOT_FILES]
    roots = ((source, "impl"), (source_work / "fusion", "fusion"),
             (source_develop, "develop/iter0"))
    for directory, destination in roots:
        files.extend((path, (Path(destination) / path.relative_to(directory)).as_posix())
                     for path in _files(directory, ignore_generated=destination == "impl",
                                        ignored_dirs=_IGNORED - {".pytest_cache"})
                     if not (destination == "develop/iter0" and path.name in _EXCLUDED_DEVELOP)
                     and (destination != "fusion" or path.relative_to(directory).as_posix() in _FUSION_FILES))
    if not any(relative.startswith("impl/") for _, relative in files):
        raise ValueError("--init-impl the implementation directory is empty.")
    for _, relative in files:
        target = work / relative
        for parent in (target, *target.parents):
            if parent == work.parent:
                break
            if parent.exists() and _linked(parent):
                raise ValueError(f"--init-impl the target must not pass through a symbolic link: {parent}")
        if target.exists():
            raise ValueError(f"--init-impl does not overwrite existing files: {target}")
    for _, relative in roots:
        target = work / relative
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise ValueError(f"--init-impl the target directory must be empty: {target}")
    result.update(source_develop_dir=str(source_develop), source_device_info=device,
                  task_sha256=task_sha, source_development_binding=binding,
                  fusion_input_hashes=fusion_input_hashes)
    rebase = _rebaser(source_work, source, work, source_develop)
    copied = []
    for path, relative in files:
        if _linked(path):
            raise ValueError(f"--init-impl cannot copy symbolic links: {path}")
        original = path.read_bytes()
        data = original
        # Implementation and provider payloads are immutable source evidence.
        preserve = relative.startswith("impl/") or relative in {
            "fusion/english_inputs.json", "fusion/translation/english_inputs.json",
            "fusion/jev_request.json", "fusion/jev_response.json", "device_info.json"}
        if not preserve:
            if path.suffix.lower() == ".json":
                value = json.loads(original.decode("utf-8-sig"))
                rebased = rebase(value)
                if rebased != value:
                    data = _encoded(rebased)
            elif path.suffix.lower() in {".md", ".txt", ".log", ".yaml", ".yml"}:
                try:
                    text = original.decode("utf-8-sig")
                    changed = rebase(text)
                    if changed != text:
                        data = changed.encode("utf-8")
                except UnicodeDecodeError:
                    pass
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(data)
        target.chmod(stat.S_IMODE(path.stat().st_mode))
        copied.append({"source": str(path), "destination": relative,
                       "source_sha256": _sha(original), "sha256": _sha(data)})
    _validate_fusion(work, fusion_input_hashes, work / "develop/iter0")
    result.update(status="prepared", copied_files=copied)
    temporary = work / (MANIFEST + ".tmp")
    with temporary.open("xb") as output:
        output.write(_encoded(result))
    temporary.replace(work / MANIFEST)
    log.info("===== Emergency implementation import: skipping Stage1, Stage1.5 and Stage2; entering Stage4 build directly =====")
    log.info("[emergency import] specified implementation=%s; original work=%s; new work=%s; file count=%d; manifest=%s",
             source, source_work, work, len(copied), work / MANIFEST)
    for relative in (*_ROOT_FILES, "impl/", "fusion/", "develop/iter0/"):
        origin = source if relative == "impl/" else source_develop if relative == "develop/iter0/" else source_work / relative
        log.info("[emergency import] copied %s -> %s", origin, work / relative)
    if binding is None:
        log.warning("[emergency import] cannot confirm the version binding between the specified code and development material; the imported first-version material is for reference only and old self-tests are not proof of passing this time.")
    log.info("[emergency import] old evaluations, history, best scores, exit counters and unrelated round material are not inherited.")
    return result


def validate_init_impl_inputs(work_dir, current_device_info, task_dir):
    """Validate imported immutable inputs on startup/resume; never call Jev."""
    work = Path(work_dir).resolve()
    manifest = load_init_impl_manifest(work)
    if manifest is None:
        raise ValueError("--init-impl the import manifest does not exist.")
    if _task_hash(task_dir) != manifest.get("task_sha256"):
        raise ValueError("--init-impl the current task differs from the imported one; the old analysis and fusion scores cannot keep being used.")
    original_device = _device_signature(manifest.get("source_device_info"))
    current_device = _device_signature(current_device_info)
    if original_device != current_device:
        changed = sorted(key for key in original_device.keys() | current_device.keys()
                         if original_device.get(key) != current_device.get(key))
        raise ValueError("--init-impl current chip/framework parameters disagree with the original scoring environment: " + ", ".join(changed))
    original_hashes = {}
    seen = set()
    for entry in manifest["copied_files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("destination"), str):
            raise ValueError("--init-impl a file manifest entry is invalid.")
        relative = Path(entry["destination"])
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() in seen):
            raise ValueError("--init-impl the file manifest contains out-of-scope or duplicate paths.")
        seen.add(relative.as_posix())
        if relative.parts[0] == "impl":
            continue  # Stage3 may legitimately modify the implementation later.
        path = work / relative
        _safe_path(path, work)
        if (not path.is_file() or not path.resolve().is_relative_to(work)
                or _linked(path) or _sha(path.read_bytes()) != entry.get("sha256")):
            raise ValueError(f"--init-impl imported material is missing or was modified: {relative.as_posix()}")
        original_hashes[relative.as_posix()] = entry.get("source_sha256")
    if not set((*_REQUIRED, _scoring_inputs_relative(work))) <= seen:
        raise ValueError("--init-impl the file manifest lacks required Stage1/1.5/2 material.")
    _required(work, work / "develop/iter0")
    # Immediate copy provenance differs from original Jev inputs after a prior
    # import has relocated paths in ANALYSIS or requirements. Keep both hashes.
    fusion_input_hashes = manifest.get("fusion_input_hashes", original_hashes)
    if (not isinstance(fusion_input_hashes, dict)
            or not set(_ROOT_FILES) <= fusion_input_hashes.keys()
            or not all(isinstance(fusion_input_hashes[name], str)
                       and re.fullmatch(r"[0-9a-f]{64}", fusion_input_hashes[name])
                       for name in _ROOT_FILES)):
        raise ValueError("--init-impl is missing the original Jev scoring input digests.")
    _validate_fusion(work, fusion_input_hashes, work / "develop/iter0")
