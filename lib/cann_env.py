"""
CANN 环境变量构建模块 — 从 config.yaml 读取，通过 source set_env.sh 获取完整环境。

核心思路：CANN 自带 set_env.sh，直接 source 它得到的环境和新开终端一模一样。
不再手动拼路径列表，换环境只需改 config.yaml 中的 toolkit_path。

用法:
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
    用 bash --login 启动子 shell（和用户新开终端一致），
    再 source set_env.sh，然后 dump 全部环境变量。
    这样能捕获 set_env.sh 设的所有 ASCEND_*/CANN_*/TOOLCHAIN_* 等变量。
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
            raise RuntimeError(f"CANN set_env.sh 执行失败：{set_env_sh}；不能继续使用其他 Toolkit 的环境")
        env_from_sh = {}
        for entry in result.stdout.split("\0"):
            if "=" in entry:
                key, val = entry.split("=", 1)
                env_from_sh[key] = val
        if not env_from_sh:
            raise RuntimeError(f"CANN set_env.sh 未返回环境：{set_env_sh}")
        return env_from_sh
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"CANN set_env.sh 无法执行：{set_env_sh}") from exc


def build_cann_env(extra=None, config_path=None):
    """
    构建包含 CANN 环境变量的 env dict，供 subprocess 使用。
    通过 bash --login + source set_env.sh 获取完整环境（和新开终端一致）。
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
    生成 bash -c 可用的 shell 前缀字符串。
    直接 source set_env.sh，和新开终端的环境一模一样。
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
    """兼容接口：供 bench_parser 等直接调用。优先从 set_env.sh 获取。"""
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
    """兼容接口。"""
    sourced = _source_set_env_sh(cfg)
    if sourced and sourced.get("PYTHONPATH"):
        return [p for p in sourced["PYTHONPATH"].split(":") if p]
    tk = cfg["toolkit_path"]
    return [f"{tk}/python/site-packages", f"{tk}/opp/built-in/op_impl/ai_core/tbe"]


def _path_dirs(cfg):
    """兼容接口。"""
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
