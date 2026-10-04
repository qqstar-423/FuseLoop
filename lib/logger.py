import logging
import sys
from pathlib import Path


def setup_logger(work_dir: str = None, level: str = "DEBUG", console_level: str = "INFO") -> logging.Logger:
    """Set up the global log + state-transition log + node logs"""
    log = logging.getLogger("triton-ascend-workflow")
    if log.handlers:
        return log
    log.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if work_dir:
        log_dir = Path(work_dir) / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        # global log (everything)
        fh = logging.FileHandler(log_dir / "workflow.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


def setup_state_logger(work_dir: str) -> logging.Logger:
    """Dedicated log for state transitions"""
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
    """Dedicated log for history changes; records the full JSON snapshot each round"""
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
    Per-node log.
    node: "N1" / "N2" / "N3" / "N4"
    One log file per node, recording all of that node's agent activity.
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
