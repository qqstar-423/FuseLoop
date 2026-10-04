"""
CANN environment variable builder — reads from config.yaml and sources set_env.sh for the full environment.

Core idea: CANN ships its own set_env.sh; sourcing it yields exactly the environment of a fresh terminal.
No more hand-built path lists; switching environments only requires changing toolkit_path in config.yaml.

Usage:
    from lib.cann_env import build_cann_env, build_cann_shell_prefix, load_cann_config
"""

import os
import shlex
import subprocess
import sys
import yaml

_cached_cann_cfg = None
_cached_cann_env = None
_cached_config_path = None


def reset_cann_cache():
    """Re-read config and the parent environment for each workflow invocation."""
    global _cached_cann_cfg, _cached_cann_env, _cached_config_path
    _cached_cann_cfg = None
    _cached_cann_env = None
    _cached_config_path = None


def _find_config_yaml():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    return os.path.join(repo_root, "config.yaml")


def load_cann_config(config_path=None):
    global _cached_cann_cfg, _cached_cann_env, _cached_config_path
    config_path = os.path.realpath(config_path or _find_config_yaml())
    if _cached_cann_cfg is not None and _cached_config_path == config_path:
        return _cached_cann_cfg

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cann = cfg.get("cann", {})
    _cached_cann_cfg = {
        "toolkit_path": cann.get("toolkit_path", ""),
        "driver_path": cann.get("driver_path", "/usr/local/Ascend/driver"),
        "arch": cann.get("arch", "aarch64-linux"),
        "node_bin": cann.get("node_bin", ""),
    }
    _cached_config_path = config_path
    _cached_cann_env = None
    return _cached_cann_cfg


def _source_set_env_sh(cfg):
    """
    Start a subshell with bash --login (matching a fresh user terminal),
    source set_env.sh, then dump all environment variables.
    This captures every ASCEND_*/CANN_*/TOOLCHAIN_* variable set by set_env.sh.
    """
    set_env_sh = os.path.join(cfg["toolkit_path"], "set_env.sh")
    if not os.path.isfile(set_env_sh):
        return {}

    script = f'source {shlex.quote(set_env_sh)} >/dev/null 2>&1 && env -0'
    try:
        result = subprocess.run(
            ["bash", "--login", "-c", script],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CANN set_env.sh execution failed: {set_env_sh}; another Toolkit\'s environment cannot be used to continue")
        env_from_sh = {}
        for entry in result.stdout.split("\0"):
            if "=" in entry:
                key, val = entry.split("=", 1)
                env_from_sh[key] = val
        if not env_from_sh:
            raise RuntimeError(f"CANN set_env.sh returned no environment: {set_env_sh}")
        return env_from_sh
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"CANN set_env.sh could not be executed: {set_env_sh}") from exc


def build_cann_env(extra=None, config_path=None):
    """
    Build an env dict containing the CANN environment variables for subprocess use.
    The full environment is obtained via bash --login + source set_env.sh (matching a fresh terminal).
    """
    global _cached_cann_env
    cfg = load_cann_config(config_path)
    if _cached_cann_env is not None:
        env = _cached_cann_env.copy()
        if extra:
            env.update(extra)
        return env

    sourced = _source_set_env_sh(cfg)
    if sourced:
        env = sourced.copy()
    else:
        env = os.environ.copy()
        tk = cfg["toolkit_path"]
        dr = cfg["driver_path"]
        arch = cfg["arch"]
        driver_libs = [f"{dr}/lib64", f"{dr}/lib64/common", f"{dr}/lib64/driver"]
        ld_parts = [
            f"{tk}/lib64", f"{tk}/{arch}/lib64",
            f"{tk}/lib64/plugin/opskernel", f"{tk}/lib64/plugin/nnengine",
        ] + driver_libs
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
        env["PYTHONPATH"] = f"{tk}/python/site-packages:{tk}/opp/built-in/op_impl/ai_core/tbe"
        env["PATH"] = f"{tk}/bin:{tk}/tools/ccec_compiler/bin:{env.get('PATH', '')}"
        env["ASCEND_HOME_PATH"] = tk
        env["ASCEND_OPP_PATH"] = f"{tk}/opp"
        env["ASCEND_TOOLKIT_HOME"] = tk
        env["ASCEND_AICPU_PATH"] = tk
        if "CANN_PATH" in env:
            env["CANN_PATH"] = tk

    # Keep the evaluator's existing driver-library compatibility paths, now
    # shared by probes and agents instead of added only for performance runs.
    driver = cfg["driver_path"]
    libraries = [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
    for path in (f"{driver}/lib64", f"{driver}/lib64/common", f"{driver}/lib64/driver"):
        if path not in libraries:
            libraries.append(path)
    env["LD_LIBRARY_PATH"] = ":".join(libraries)

    # A login shell may reset the parent's visibility map. Device IDs are
    # logical within that map, so all probes/agents/evaluators must preserve it.
    for key in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        if key in os.environ:
            env[key] = os.environ[key]

    if cfg.get("node_bin") and cfg["node_bin"] not in env.get("PATH", ""):
        env["PATH"] = cfg["node_bin"] + ":" + env.get("PATH", "")

    env.pop("LD_PRELOAD", None)

    _cached_cann_env = env.copy()
    if extra:
        env.update(extra)
    return env


def build_cann_shell_prefix(config_path=None):
    """
    Generate a shell prefix string usable with bash -c.
    It sources set_env.sh directly, giving exactly the environment of a fresh terminal.
    """
    cfg = load_cann_config(config_path)
    set_env_sh = os.path.join(cfg["toolkit_path"], "set_env.sh")
    dr = cfg["driver_path"]

    parts = [f'source "{set_env_sh}" 2>/dev/null']
    parts.append(
        f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{dr}/lib64:{dr}/lib64/common:{dr}/lib64/driver"
    )
    parts.append("unset LD_PRELOAD")
    return "; ".join(parts)


def _ld_library_paths(cfg):
    """Compatibility interface for direct calls from bench_parser etc. Prefers values from set_env.sh."""
    sourced = _source_set_env_sh(cfg)
    if sourced and sourced.get("LD_LIBRARY_PATH"):
        dr = cfg["driver_path"]
        paths = sourced["LD_LIBRARY_PATH"].split(":")
        for p in [f"{dr}/lib64", f"{dr}/lib64/common", f"{dr}/lib64/driver"]:
            if p not in paths:
                paths.append(p)
        return paths
    tk = cfg["toolkit_path"]
    dr = cfg["driver_path"]
    arch = cfg["arch"]
    return [
        f"{tk}/lib64", f"{tk}/{arch}/lib64",
        f"{tk}/lib64/plugin/opskernel", f"{tk}/lib64/plugin/nnengine",
        f"{dr}/lib64", f"{dr}/lib64/common", f"{dr}/lib64/driver",
    ]


def _pythonpath_dirs(cfg):
    """Compatibility interface."""
    sourced = _source_set_env_sh(cfg)
    if sourced and sourced.get("PYTHONPATH"):
        return [p for p in sourced["PYTHONPATH"].split(":") if p]
    tk = cfg["toolkit_path"]
    return [f"{tk}/python/site-packages", f"{tk}/opp/built-in/op_impl/ai_core/tbe"]


def _path_dirs(cfg):
    """Compatibility interface."""
    sourced = _source_set_env_sh(cfg)
    if sourced and sourced.get("PATH"):
        paths = sourced["PATH"].split(":")
        if cfg.get("node_bin") and cfg["node_bin"] not in paths:
            paths.insert(0, cfg["node_bin"])
        return paths
    tk = cfg["toolkit_path"]
    dirs = [f"{tk}/bin", f"{tk}/tools/ccec_compiler/bin"]
    if cfg.get("node_bin"):
        dirs.append(cfg["node_bin"])
    return dirs
