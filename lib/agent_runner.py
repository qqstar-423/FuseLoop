"""
Agent Runner — invokes the agent as a fresh process each time; sessions are never continued.
Context is supplied entirely by the orchestrator-injected prompt (including file paths); agents read files themselves.
"""

import errno
import json
from contextlib import nullcontext
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import logging
import os
try:
    import pty
except ImportError:  # Importable on Windows; NPU agent execution uses POSIX PTYs.
    pty = None
import select
from queue import Empty, Queue
import subprocess
import sys
import tempfile
from threading import Thread
import time
from pathlib import Path

from .handoff import read_file_safe

log = logging.getLogger("triton-ascend-workflow")

HEARTBEAT_INTERVAL = 30
DEVICE_ID = os.environ.get("WORKFLOW_NPU_DEVICE_ID", "0")
_workflow_agent_configs = ContextVar("workflow_agent_configs", default=None)


def isolated_agent_configuration(function):
    """Keep one workflow invocation's agent settings out of other runs/tests."""
    @wraps(function)
    def scoped(*args, **kwargs):
        token = _workflow_agent_configs.set(None)
        try:
            return function(*args, **kwargs)
        finally:
            _workflow_agent_configs.reset(token)
    return scoped


def bind_workflow_agent_configuration(config, config_path=None):
    """Make every stage/helper use the selected config and logical NPU device."""
    from .cann_env import _find_config_yaml
    from .framework_target import validate_device_id
    device_id = validate_device_id(config.get("hardware", {}).get("device_id", 0))
    agents = deepcopy(config.get("agents", {}))
    for agent in agents.values():
        agent["workflow_config_path"] = os.path.abspath(config_path or _find_config_yaml())
        agent["device_id"] = device_id
    _workflow_agent_configs.set(agents)


def _build_env(extra: dict = None, config_path=None) -> dict:
    from .cann_env import build_cann_env
    env = build_cann_env(config_path=config_path)
    env["WORKFLOW_NPU_DEVICE_ID"] = DEVICE_ID

    if extra:
        env.update(extra)
    return env


def _agent_env(config):
    return _build_env({"WORKFLOW_NPU_DEVICE_ID": str(config.get("device_id", DEVICE_ID))},
                      config_path=config.get("workflow_config_path"))


def _load_agent_config(agent_name: str) -> dict:
    """Read the named agent\'s configuration (CLI path etc.) from config.yaml."""
    bound = _workflow_agent_configs.get()
    if bound is not None:
        return deepcopy(bound.get(agent_name, {}))
    from .cann_env import load_cann_config, _find_config_yaml
    import yaml
    config_path = _find_config_yaml()
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        agent = dict(cfg.get("agents", {}).get(agent_name, {}))
        agent["workflow_config_path"] = os.path.abspath(config_path)
        agent["device_id"] = cfg.get("hardware", {}).get("device_id", 0)
        return agent
    except Exception:
        return {}


GLOBAL_CONSTRAINT = (
    "\n\n⛔ Prohibition: all files under the task/ directory (including golden.py, proto.yaml, cases.yaml, desc.md, etc.) "
    "are evaluation benchmark data; modifying, overwriting or deleting them is strictly forbidden. Read-only, never write. Violating this rule invalidates the evaluation results.\n"
)


def run_agent(agent: str, role_file: str, work_dir: str, prompt: str = "", config: dict = None, node_log: logging.Logger = None) -> bool:
    config = _load_agent_config(agent) if config is None else config
    prompt = prompt + GLOBAL_CONSTRAINT
    if agent == "cannbot":
        return run_cannbot(role_file, work_dir, prompt, config, node_log)
    elif agent == "kerminal":
        return run_kerminal(role_file, work_dir, prompt, config, node_log)
    elif agent == "hermes":
        return run_hermes(role_file, work_dir, prompt, config, node_log)
    else:
        log.error(f"Unknown agent: {agent}")
        return False


def run_cannbot(role_file: str, work_dir: str, prompt: str, config: dict = None, node_log: logging.Logger = None) -> bool:
    """
    Call CANNBot — a fresh process every time; sessions are never continued.
    Context comes from the injected prompt (including file paths; the agent reads them itself).
    """
    config = config or {}
    system_prompt = read_file_safe(role_file)
    full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

    cli = config.get("cli")
    if not cli:
        raise ValueError("cannbot cli path is not configured; check config.yaml agents.cannbot.cli")
    cwd = config.get("cwd", work_dir)
    # CANNBot run reads non-TTY stdin as the message. A file-backed stdin avoids
    # both execve's argv limit and a pipe-write/output-read deadlock for long input.
    prompt_path = _save_agent_prompt(work_dir, role_file, full_prompt, agent="cannbot",
                                     transport="stdin", node_log=node_log)
    cmd = [cli, "run", "--format", "json", "--dangerously-skip-permissions"]

    log.info(f"[N1 CANNBot] starting, cwd={cwd}")
    result = _run_with_heartbeat(cmd, cwd=cwd, env=_agent_env(config), agent_name="N1 CANNBot", node_log=node_log,
                                 stdin_path=prompt_path)

    if result.returncode != 0:
        log.error(f"[N1 CANNBot] failed, returncode={result.returncode}")
        if "Prompt exceeds max length" in result.stdout + (result.stderr or ""):
            log.error("[N1 CANNBot] model context limit exceeded; the prompt was passed in full via stdin, "
                      "which differs from the operating system's command-line argument length limit. Check this prompt file and the model's context capacity.")
    return result.returncode == 0


def run_kerminal(role_file: str, work_dir: str, prompt: str = "", config: dict = None, node_log: logging.Logger = None) -> bool:
    """
    Call Kerminal — a fresh process every time; sessions are never continued.
    The native exec mode reads the full task from stdin; -a never and the configured environment are kept.
    """
    config = config or {}
    system_prompt = read_file_safe(role_file)
    full_prompt = f"{system_prompt}\n\n---\n\n{prompt}" if prompt else system_prompt

    cli = config.get("cli")
    if not cli:
        raise ValueError("kerminal cli path is not configured; check config.yaml agents.kerminal.cli")
    cwd = config.get("cwd", work_dir)
    prompt_path = _save_agent_prompt(work_dir, role_file, full_prompt, agent="kerminal",
                                     transport="exec stdin", node_log=node_log)
    # Verified with Kerminal 0.8.12 help exec. -a is a top-level option;
    # work directories need not be Git repositories. Do not add sandbox bypass.
    cmd = [cli, "-a", "never", "exec", "--skip-git-repo-check", "-C", cwd, "-"]

    log.info(f"[N2 Kerminal] starting, role={os.path.basename(role_file)}, cwd={cwd}")
    if node_log:
        node_log.info(f"starting, role={os.path.basename(role_file)}")
    result = _run_with_heartbeat(cmd, cwd=cwd, env=_agent_env(config), agent_name="N2 Kerminal",
                                 node_log=node_log, plain_text=True, stdin_path=prompt_path)
    return result.returncode == 0


def run_hermes(role_file: str, work_dir: str, prompt: str = "", config: dict = None, node_log: logging.Logger = None) -> bool:
    """
    Call the Hermes CLI — a fresh process every time; sessions are never continued.
    Like cannbot/kerminal, it is invoked through a CLI.
    """
    config = config or {}
    system_prompt = read_file_safe(role_file)
    full_prompt = f"{system_prompt}\n\n---\n\n{prompt}" if prompt else system_prompt

    cli = config.get("cli")
    if not cli:
        raise ValueError("hermes cli path is not configured; check config.yaml agents.hermes.cli")
    cwd = config.get("cwd", work_dir)
    prompt_path = _save_agent_prompt(work_dir, role_file, full_prompt, agent="hermes",
                                     transport="file read inside the Python process", node_log=node_log)
    # Hermes 0.20.4 -z does not read stdin. Keep its existing oneshot behavior:
    # the bridge reads the file after process creation and runs the same entrypoint.
    from .hermes_prompt import build_command
    env = _agent_env(config)
    cmd = build_command(cli, prompt_path, cwd, env, ["-t", "web,file", "--yolo", "--in", cwd])

    log.info(f"[N3 Hermes] starting, cwd={cwd}")
    if node_log:
        node_log.info(f"starting, cwd={cwd}")
    result = _run_with_heartbeat(cmd, cwd=cwd, env=env, agent_name="N3 Hermes", timeout=0, node_log=node_log, plain_text=True)

    # A transport/CLI failure must not be masked by a report from an older round.
    if result.returncode != 0:
        log.error(f"[N3 Hermes] failed, returncode={result.returncode}; an old report must not be treated as success")
        return False

    # Check whether the output file was generated
    iter_search_dirs = [d for d in os.listdir(os.path.join(work_dir, "search")) if os.path.isdir(os.path.join(work_dir, "search", d))] if os.path.isdir(os.path.join(work_dir, "search")) else []
    search_report = None
    for d in sorted(iter_search_dirs, reverse=True):
        candidate = os.path.join(work_dir, "search", d, "SEARCH_REPORT.md")
        if os.path.exists(candidate):
            search_report = candidate
            break
    if not search_report:
        search_report = os.path.join(work_dir, "search", "SEARCH_REPORT.md")

    if os.path.exists(search_report):
        log.info(f"[N3 Hermes] SEARCH_REPORT.md generated: {search_report}")
        return True
    else:
        log.warning("[N3 Hermes] search did not complete (SEARCH_REPORT.md was not generated)")
        return result.returncode == 0


# ═══════════════════════════════════════════════════════════════
# low-level helpers
# ═══════════════════════════════════════════════════════════════


def _save_agent_prompt(work_dir, role_file, full_prompt, *, agent, transport, node_log=None) -> str:
    """Keep the exact UTF-8 input per invocation, including retries in one stage."""
    directory = Path(work_dir).resolve() / "log" / "prompts"
    directory.mkdir(parents=True, exist_ok=True)
    payload = full_prompt.encode("utf-8")
    # mkstemp creates a unique file with owner-only permissions on POSIX.
    fd, path = tempfile.mkstemp(prefix=f"{agent}_{Path(role_file).stem}_", suffix=".txt", dir=directory)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
    label = {"cannbot": "N1 CANNBot", "kerminal": "N2 Kerminal", "hermes": "N3 Hermes"}[agent]
    message = (f"[{label}] prompt transport={transport}, UTF-8 bytes={len(payload)}, "
               f"full prompt file={path}")
    log.info(message)
    if node_log and node_log is not log:
        node_log.info(message)
    return path


def _start_process(cmd, *, env, agent_name, **kwargs):
    """Diagnose launch limits without printing the prompt or environment values."""
    try:
        return subprocess.Popen(cmd, env=env, **kwargs)
    except OSError as exc:
        if exc.errno == errno.E2BIG:
            argv_sizes = [len(os.fsencode(value)) + 1 for value in cmd]
            env_sizes = [len(os.fsencode(key)) + len(os.fsencode(value)) + 2
                         for key, value in env.items()]
            log.error(f"[{agent_name}] process never started: the system rejected the command-line arguments or environment as too long (E2BIG); "
                      f"argv bytes={sum(argv_sizes)}, largest single argument={max(argv_sizes, default=0)}, "
                      f"env bytes={sum(env_sizes)}, largest single environment variable={max(env_sizes, default=0)}. "
                      "This is not a model context error; check the argument passing method or environment variable sizes.")
        raise


def _run_with_heartbeat(cmd: list, cwd: str, env: dict, agent_name: str, timeout: int = 0, node_log: logging.Logger = None, plain_text: bool = False, stdin_path: str = None) -> subprocess.CompletedProcess:
    """Read each output pipe through one UTF-8 decoder, including its final tail.

    Reader threads keep partial lines from blocking heartbeat/timeout handling.
    Never mix TextIOWrapper reads with communicate(): its raw drain can start
    halfway through a UTF-8 character already buffered by the text reader.
    """
    nlog = node_log or log  # fallback to global log
    # The child keeps its own handle after Popen. Close the parent's copy even
    # when startup fails; no writer thread or buffering of a giant stdin pipe.
    with (open(stdin_path, "rb") if stdin_path is not None else nullcontext(None)) as prompt_input:
        proc = _start_process(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="backslashreplace", env=env,
            bufsize=1, stdin=prompt_input, agent_name=agent_name,
        )
    start = time.monotonic()
    last_heartbeat = start
    chunks = {"stdout": [], "stderr": []}
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5
    output = Queue()

    def read_stream(name, stream):
        try:
            for line in iter(stream.readline, ""):
                output.put((name, line, None))
        except Exception as exc:
            output.put((name, None, exc))
        finally:
            try:
                stream.close()
            finally:
                output.put((name, None, None))

    readers = [Thread(target=read_stream, args=(name, stream),
                      name=f"agent-output-{name}", daemon=True)
               for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr))]
    open_streams = {"stdout", "stderr"}
    stop_requested = False

    try:
        for reader in readers:
            reader.start()
        # Even a child that has already exited may have unread lines in either
        # pipe. Consume both EOF markers before returning the complete result.
        while open_streams or proc.poll() is None:
            now = time.monotonic()
            elapsed = now - start
            if not stop_requested and proc.poll() is None:
                if timeout > 0 and elapsed > timeout:
                    proc.kill()
                    stop_requested = True
                    log.error(f"[{agent_name}] timed out ({timeout}s); killing the process")
                elif consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    proc.kill()
                    stop_requested = True
                    log.error(f"[{agent_name}] {MAX_CONSECUTIVE_ERRORS} consecutive API errors; terminating the process")
                    if nlog is not log:
                        nlog.error(f"[{agent_name}] {MAX_CONSECUTIVE_ERRORS} consecutive API errors; terminating the process")
            if not stop_requested and now - last_heartbeat >= HEARTBEAT_INTERVAL:
                log.info(f"[{agent_name}] still running... ({elapsed:.0f}s)")
                last_heartbeat = now

            try:
                name, line, read_error = output.get(timeout=0.2)
            except Empty:
                continue
            if read_error is not None:
                raise RuntimeError(f"[{agent_name}] failed to read the {name} log: {read_error}") from read_error
            if line is None:
                open_streams.discard(name)
                continue
            if line:
                chunks[name].append(line)
                stripped = line.strip()
                if not stripped:
                    continue
                # Parse cannbot JSON output and print the key information
                if '"type":"text"' in stripped or '"type":"error"' in stripped:
                    try:
                        import json as _json
                        data = _json.loads(stripped)
                        if data.get("type") == "error":
                            err_msg = data.get("error", {}).get("data", {}).get("message", str(data.get("error", "")))
                            log.error(f"[{agent_name}] API error: {err_msg}")
                            nlog.error(f"[{agent_name}] API error: {err_msg}")
                            consecutive_errors += 1
                        elif data.get("type") == "text":
                            text = data.get("part", {}).get("text", "")
                            if text.strip():
                                log.info(f"[{agent_name}] output: {text.strip()[:200]}")
                                nlog.info(f"[{agent_name}] output: {text.strip()[:200]}")
                                consecutive_errors = 0  # reset when normal output arrives
                    except Exception:
                        pass
                elif '"tool":"' in stripped:
                    try:
                        import json as _json
                        data = _json.loads(stripped)
                        tool = data.get("part", {}).get("tool", "")
                        title = data.get("part", {}).get("state", {}).get("title", "")
                        status = data.get("part", {}).get("state", {}).get("status", "")
                        if tool and status == "completed":
                            log.info(f"[{agent_name}] tool:{tool} → {title}")
                    except Exception:
                        pass
                elif "error" in stripped.lower() or "ERROR" in stripped:
                    log.warning(f"[{agent_name}] {stripped[:200]}")
                elif plain_text and len(stripped) > 5:
                    # plain-text agents such as hermes: print non-JSON lines directly
                    log.info(f"[{agent_name}] output: {stripped[:200]}")
                    nlog.info(f"[{agent_name}] output: {stripped[:200]}")

        proc.wait()
    finally:
        # Preserve interruption/startup error behavior and do not leave the
        # directly launched CLI running if processing its output fails.
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=1.0)

    stdout_all = "".join(chunks["stdout"])
    stderr_all = "".join(chunks["stderr"])
    elapsed = time.monotonic() - start
    log.info(f"[{agent_name}] finished, elapsed={elapsed:.0f}s, returncode={proc.returncode}")
    if stderr_all:
        log.debug(f"[{agent_name}] stderr (last 500):\n{stderr_all[-500:]}")

    return subprocess.CompletedProcess(args=cmd, returncode=proc.returncode, stdout=stdout_all, stderr=stderr_all)


def _run_with_pty(cmd: list, cwd: str, env: dict, timeout: int = 0, node_log: logging.Logger = None, agent_label: str = "N2 Kerminal") -> tuple:
    """
    Run commands that require a TTY (kerminal).
    Emulates terminal responses (CSI 6n) + presses Enter automatically (welcome screen) + exits on idle detection.
    """
    if pty is None:
        raise RuntimeError("NPU workflow agent execution requires a POSIX host. Standalone Jev/Kerminal RPC tools support Windows.")
    import struct
    import fcntl
    import termios
    import re

    _ansi_re = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\x1b[()][0-9A-B]|\x1b[M]|\x0f|\x0e|\r|\x1b\[\?[0-9]*[hl]')

    def _extract_lines(raw):
        """Extract readable lines from raw TUI output"""
        clean = _ansi_re.sub('', raw)
        clean = clean.replace('•', '').replace('›', '')
        lines = []
        for l in clean.split('\n'):
            l = l.strip()
            if len(l) > 10 and not l.startswith('╭') and not l.startswith('│') and not l.startswith('╰') and not l.startswith('├'):
                lines.append(l)
        return lines

    nlog = node_log or log
    output_chunks = []
    _tui_buffer = ""
    _logged_lines = set()
    master_fd, slave_fd = pty.openpty()

    try:
        winsize = struct.pack('HHHH', 50, 200, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        run_env = env.copy()
        run_env['TERM'] = 'xterm-256color'
        run_env['COLUMNS'] = '200'
        run_env['LINES'] = '50'

        proc = _start_process(
            cmd, cwd=cwd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env=run_env, close_fds=True, agent_name=agent_label,
        )
    except BaseException:
        os.close(master_fd)
        raise
    finally:
        os.close(slave_fd)

    start = time.time()
    initial_enter_sent = False
    last_data_time = time.time()
    last_heartbeat = start
    task_started = False
    idle_exit_seconds = 15

    try:
        while True:
            if timeout > 0 and (time.time() - start) > timeout:
                proc.kill()
                break
            ready, _, _ = select.select([master_fd], [], [], 1.0)
            if ready:
                try:
                    data = os.read(master_fd, 8192)
                    if not data:
                        break
                    text = data.decode("utf-8", errors="replace")
                    output_chunks.append(text)
                    last_data_time = time.time()
                    # accumulate TUI output into the buffer
                    _tui_buffer += text
                    if '\x1b[6n' in text:
                        os.write(master_fd, b'\x1b[1;1R')
                    if not initial_enter_sent and 'Press Enter to continue' in text:
                        os.write(master_fd, b'\r')
                        initial_enter_sent = True
                    if 'Worked for' in text or 'context left' in text:
                        task_started = True
                except OSError:
                    break
            else:
                # on each select timeout (1 second), process the accumulated TUI buffer
                if _tui_buffer:
                    lines = _extract_lines(_tui_buffer)
                    for line in lines:
                        if line not in _logged_lines and 'Worked for' not in line and 'context left' not in line and 'Press Enter' not in line:
                            nlog.info(f"output: {line[:200]}")
                            log.info(f"[{agent_label}] output: {line[:200]}")
                            _logged_lines.add(line)
                    _tui_buffer = ""
                if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL:
                    log.info(f"[{agent_label}] still running... ({time.time()-start:.0f}s)")
                    nlog.info(f"still running... ({time.time()-start:.0f}s)")
                    last_heartbeat = time.time()
                if task_started and (time.time() - last_data_time) > idle_exit_seconds:
                    proc.terminate()
                    time.sleep(1)
                    if proc.poll() is None:
                        proc.kill()
                    break
            if proc.poll() is not None:
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0.1)
                    if not ready:
                        break
                    try:
                        data = os.read(master_fd, 8192)
                        if not data:
                            break
                        output_chunks.append(data.decode("utf-8", errors="replace"))
                    except OSError:
                        break
                break
    finally:
        os.close(master_fd)

    proc.wait()
    elapsed = time.time() - start
    log.info(f"[{agent_label}] finished, elapsed={elapsed:.0f}s, returncode={proc.returncode}")
    nlog.info(f"finished, elapsed={elapsed:.0f}s, returncode={proc.returncode}")
    return proc.returncode, "".join(output_chunks)
