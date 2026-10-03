import logging
import sys
from pathlib import Path


def setup_logger(work_dir: str = None, level: str = "DEBUG", console_level: str = "INFO") -> logging.Logger:
    """设置全局日志 + 状态切换日志 + 节点日志"""
    log = logging.getLogger("triton-ascend-workflow")
    if log.handlers:
        return log
    log.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 终端输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if work_dir:
        log_dir = Path(work_dir) / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        # 全局日志（所有内容）
        fh = logging.FileHandler(log_dir / "workflow.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


def setup_state_logger(work_dir: str) -> logging.Logger:
    """状态切换专用日志"""
    log = logging.getLogger("triton-ascend-state")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir = Path(work_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / "state_transitions.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


def setup_history_logger(work_dir: str) -> logging.Logger:
    """history 变化专用日志，每轮记录完整 JSON 快照"""
    log = logging.getLogger("triton-ascend-history")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir = Path(work_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / "history.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


def setup_node_logger(work_dir: str, node: str) -> logging.Logger:
    """
    节点专用日志。
    node: "N1" / "N2" / "N3" / "N4"
    每个节点一个日志文件，记录该节点 agent 的所有活动。
    """
    logger_name = f"triton-ascend-{node}"
    log = logging.getLogger(logger_name)
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir = Path(work_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / f"{node}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log
