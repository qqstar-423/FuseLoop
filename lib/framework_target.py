"""Triton Ascend target identity, implementation and runtime validation."""

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tokenize

from .handoff import atomic_write_json


FRAMEWORK = "Triton"
BACKEND = "triton-ascend"
PROGRAMMING_MODEL = "Triton block program (SPMD)"
TARGET_FILE = "workflow_target.json"
TARGET_IDENTITY = {"schema_version": 1, "framework": FRAMEWORK, "backend": BACKEND}


def validate_device_id(device_id):
    """The configured ID is logical inside the caller's existing visibility map."""
    if type(device_id) is not int or device_id < 0:
        raise ValueError("hardware.device_id must be a nonnegative integer")
    return device_id


def _imports_triton(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(name.name.split(".", 1)[0] == "triton" for name in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if (node.module or "").split(".", 1)[0] == "triton":
                return True
    return False


def _unfinished_source_imports_triton(source):
    """Read import statements even when Stage4 still needs to repair syntax."""
    statement = []

    def matches():
        try:
            return _imports_triton(ast.parse(tokenize.untokenize(statement)))
        except SyntaxError:
            return False

    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in {tokenize.NEWLINE, tokenize.ENDMARKER} or token.string == ";":
                if statement and matches():
                    return True
                statement = []
            elif statement:
                statement.append((token.type, token.string))
            elif token.type == tokenize.NAME and token.string in {"from", "import"}:
                statement.append((token.type, token.string))
    except (tokenize.TokenError, IndentationError):
        pass
    return bool(statement and matches())


def assert_target_implementation(directory):
    """Require Triton source imports across an implemented submission directory.

    Imports may live in a kernel module separate from package exports and host
    wrappers. Empty packages, docstrings and pass statements are valid before
    development; packaging metadata alone does not establish an implementation.
    This is a source identity check, not proof of kernel execution or correctness.
    """
    root = Path(directory)
    if not root.is_dir():
        return
    has_implementation = False
    imports_triton = False
    for path in root.rglob("*.py"):
        if any(part in {".git", "__pycache__", "build", "dist"} for part in path.relative_to(root).parts):
            continue
        if path == root / "setup.py":
            continue
        source = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source)
            has_implementation |= any(not isinstance(node, ast.Pass)
                and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                         and isinstance(node.value.value, str)) for node in tree.body)
            imports_triton |= _imports_triton(tree)
        except SyntaxError:
            has_implementation = True
            imports_triton |= _unfinished_source_imports_triton(source)
    if has_implementation and not imports_triton:
        raise ValueError(f"The implementation has no Triton source import: {root}. Provide a Triton Ascend implementation; existing files were not modified.")


def ensure_workflow_target(work_dir):
    """Only new empty runs or explicitly marked Triton runs may proceed."""
    work = Path(work_dir)
    marker = work / TARGET_FILE
    if marker.is_file():
        try:
            target = json.loads(marker.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot verify the working directory\'s target framework: {marker}; use a new work directory") from exc
        if not isinstance(target, dict) or any(target.get(key) != value for key, value in TARGET_IDENTITY.items()):
            raise ValueError(f"The working directory is not a Triton Ascend task: {marker}. Target identity must match this workflow; use a new work directory.")
        assert_target_implementation(work / "impl")
        return marker
    existing = [work / name for name in (".state.json", "history.json", "device_info.json", "ANALYSIS.md")
                if (work / name).exists()]
    for name in ("impl", "eval", "develop", "fusion", "knowledge"):
        folder = work / name
        if folder.is_dir() and any(path.is_file() for path in folder.rglob("*")):
            existing.append(folder)
    if existing:
        raise ValueError(f"The existing work directory lacks {TARGET_FILE}, so its Triton Ascend target cannot be verified. Create a new work directory. Existing files were not modified.")
    work.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, TARGET_IDENTITY)
    return marker


def ensure_example_link(work_dir, cannbench_repo):
    """Require the declared cann-bench example, never retain an unrelated old link."""
    source = Path(cannbench_repo).resolve() / "examples" / "triton_ascend_cann_example"
    link = Path(work_dir) / "example"
    if not (source / "cann_bench" / "__init__.py").is_file():
        raise ValueError(f"Missing the standard Triton Ascend example: {source}; update the repository at paths.cannbench_repo and confirm cann_bench/__init__.py exists.")
    assert_target_implementation(source)
    if os.path.lexists(link):
        if not link.is_dir() or link.resolve() != source:
            raise ValueError(f"The working directory\'s example source does not match: {link}; it must point to {source}. Use a new work directory or the matching example source.")
    else:
        os.symlink(source, link, target_is_directory=True)
    return source


def ensure_task_link(work_dir, task_dir):
    """The agent's read-only task must be exactly the evaluator's task directory."""
    source = Path(task_dir).resolve()
    link = Path(work_dir) / "task"
    if not source.is_dir():
        raise ValueError(f"Operator task directory does not exist: {source}")
    if os.path.lexists(link):
        if not link.is_dir() or link.resolve() != source:
            raise ValueError(f"The working directory\'s task does not match --task-dir: {link}; {source} is required. Use the matching task or a new work directory; mixing requirements with evaluation cases is forbidden.")
    else:
        os.symlink(source, link, target_is_directory=True)
    return source


def read_cann_toolchain(env):
    """Fingerprint installed CANN version metadata from the active environment.

    Compiler metadata is the layout used by cann-bench's Docker self-test. A
    toolkit-level version.info is accepted for other supported CANN layouts.
    Never derive the version from a directory name or from the current date.
    """
    # Preserve an environment's "latest" symlink here so the next comparison
    # follows a toolkit switch; each version file below records its real path.
    roots = {key: os.path.abspath(env[key])
             for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "CANN_PATH")
             if isinstance(env.get(key), str) and env[key].strip()}
    files = {}
    for directory in roots.values():
        root = Path(directory)
        for relative in ("compiler/version.info", "version.info"):
            path = root / relative
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if not raw.strip():
                raise RuntimeError(f"CANN version info is empty: {path}")
            content = raw.decode("utf-8-sig", errors="replace")
            versions = re.findall(r"(?mi)^\s*version\s*=\s*(.+?)\s*$", content)
            resolved = str(path.resolve())
            files[resolved] = {"path": resolved, "sha256": hashlib.sha256(raw).hexdigest(),
                               "version": versions[0] if versions else "see_version_file"}
    if not files:
        raise RuntimeError("Cannot verify the CANN toolchain version: check the current CANN environment variables and compiler/version.info or version.info; mixing performance records without version evidence is not allowed.")
    return {"verified": True, "cann_roots": roots,
            "version_files": [files[key] for key in sorted(files)]}


def detect_triton_runtime(device_id=0, config_path=None):
    """Probe the installed active NPU backend, not merely whether Triton imports.

    No CUDA/NPU visibility variable is rewritten. Torch and the evaluator receive
    the same logical device ID. The subprocess uses the workflow's CANN setup.
    """
    validate_device_id(device_id)
    from .cann_env import build_cann_env
    script = (
        "import json, torch, torch_npu, triton\n"
        "from importlib.metadata import version\n"
        "from triton.runtime import driver\n"
        f"torch.npu.set_device({device_id})\n"
        "target = driver.active.get_current_target()\n"
        "print('WORKFLOW_TRITON_RUNTIME=' + json.dumps({\n"
        " 'framework': 'Triton', 'backend': 'triton-ascend',\n"
        " 'driver_backend': str(target.backend), 'target_arch': str(target.arch),\n"
        " 'runtime_versions': {'triton': str(triton.__version__),\n"
        "  'triton_ascend': version('triton-ascend'),\n"
        "  'torch': str(torch.__version__), 'torch_npu': str(torch_npu.__version__)}}))\n"
    )
    env = build_cann_env({"WORKFLOW_NPU_DEVICE_ID": str(device_id)}, config_path=config_path)
    toolchain = read_cann_toolchain(env)
    try:
        result = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                                timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Triton Ascend runtime environment detection could not complete: {exc}") from exc
    if result.returncode:
        raise RuntimeError("Triton Ascend runtime environment detection failed; check version compatibility among triton-ascend, torch_npu and CANN.\n"
                           + (result.stderr or result.stdout)[-1200:])
    lines = [line.split("=", 1)[1] for line in result.stdout.splitlines()
             if line.startswith("WORKFLOW_TRITON_RUNTIME=")]
    try:
        runtime = json.loads(lines[-1])
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("Triton Ascend detection returned no valid backend information") from exc
    if not isinstance(runtime, dict):
        raise RuntimeError("Triton Ascend detection returned no backend object")
    # triton-ascend registers the active compiler target as "npu". This is
    # distinct from the human-readable distribution/backend label above.
    if runtime.get("driver_backend") != "npu":
        raise RuntimeError(f"This project requires Triton Ascend, but the actual active backend is {runtime.get('driver_backend', 'unknown')}; a CUDA/HIP backend cannot substitute.")
    versions = runtime.get("runtime_versions")
    if (not isinstance(versions, dict) or any(not isinstance(versions.get(name), str)
            or not versions[name].strip() or versions[name].lower() == "unknown"
            for name in ("triton", "triton_ascend", "torch", "torch_npu"))):
        raise RuntimeError("Triton Ascend detection lacks actual triton/triton-ascend/torch/torch_npu versions")
    runtime.update(framework=FRAMEWORK, backend=BACKEND, programming_model=PROGRAMMING_MODEL,
                   toolchain=toolchain)
    return runtime
