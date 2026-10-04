"""Pass long prompts to the installed Hermes Python CLI without exec argv limits.

Hermes 0.20.4's ``-z`` accepts a string, not stdin or a prompt file.  Launch the
same console script in its own interpreter and construct that string only after
the process starts.  This module deliberately does not import Hermes itself.
"""

import ast
import os
from pathlib import Path
import re
import runpy
import shlex
import shutil
import sys


_PYTHON_NAME = re.compile(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?(?:w)?(?:\.exe)?$", re.I)
_MAX_ENTRY_BYTES = 64 * 1024
_REEXEC_PROMPT_BYTES = 64 * 1024


def _error(detail):
    return ValueError(
        "Cannot confirm the Hermes long-prompt entrypoint: " + detail
        + ". Set agents.hermes.cli to the hermes entrypoint produced by pip install in the Python environment where Hermes lives; "
        "the program will not fall back to putting the full prompt on the command line."
    )


def _executable(value, cwd, env):
    path = Path(value)
    if not path.is_absolute() and not any(separator in value for separator in ("/", "\\")):
        child_cwd = os.path.abspath(cwd)
        search_path = os.pathsep.join(
            item if os.path.isabs(item) else os.path.join(child_cwd, item)
            for item in env.get("PATH", os.defpath).split(os.pathsep)
        )
        located = shutil.which(value, path=search_path)
        if not located:
            raise _error(f"executable not found: {value}")
        path = Path(located)
    if not path.is_absolute():
        path = Path(cwd) / path
    # Do not resolve this path: resolving venv/bin/python's symlink selects the
    # system interpreter and loses the virtual environment's installed packages.
    path = Path(os.path.abspath(path))
    if not path.is_file() or (os.name != "nt" and not os.access(path, os.X_OK)):
        raise _error(f"the executable does not exist or is not executable: {path}")
    return str(path)


def _entry_source(path):
    with open(path, "rb") as stream:
        raw = stream.read(_MAX_ENTRY_BYTES + 1)
    if len(raw) > _MAX_ENTRY_BYTES or b"\x00" in raw:
        raise _error(f"{path} is not a recognizable Python console script")
    try:
        source = raw.decode("utf-8-sig")
        tree = ast.parse(source, filename=str(path))
    except (UnicodeError, SyntaxError) as exc:
        raise _error(f"{path} is not a recognizable Python console script") from exc
    names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "hermes_cli.main"
        for alias in node.names if alias.name == "main"
    }
    if not names or not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
        for node in ast.walk(tree)
    ):
        raise _error(f"{path} does not call the official hermes_cli.main:main entrypoint")
    return source


def _python_options(options):
    """Allow interpreter settings, never a second script, -c or -m program."""
    index = 0
    while index < len(options):
        option = options[index]
        if re.fullmatch(r"-[BEIOPsuS]+", option) or option.startswith(("-X", "-W")):
            if option in {"-X", "-W"}:
                index += 1
                if index == len(options):
                    raise _error("the Python shebang\'s -X/-W is missing its value")
            index += 1
            continue
        raise _error(f"unsupported Python shebang option {option}")
    return options


def _interpreter(source, cwd, env):
    lines = source.splitlines()
    if not lines or not lines[0].startswith("#!"):
        raise _error("the entrypoint has no Python shebang")
    try:
        launcher = shlex.split(lines[0][2:].strip())
        if not launcher:
            raise _error("the entrypoint\'s shebang is empty")
        if launcher[0] in {"/bin/sh", "/usr/bin/sh"}:
            # distlib's exact polyglot wrapper for Python paths containing
            # spaces or exceeding the operating system's shebang length.
            if (len(launcher) != 1 or len(lines) < 3
                    or not lines[1].startswith("'''exec' ") or lines[2] != "' '''"):
                raise _error("not a standard pip Python shell wrapper")
            wrapper = shlex.split(lines[1][len("'''exec' "):])
            if len(wrapper) != 3 or wrapper[-2:] != ["$0", "$@"]:
                raise _error("the pip shell wrapper contains unsupported extra commands or arguments")
            launcher = wrapper[:1]
        elif Path(launcher[0]).name in {"env", "env.exe"}:
            launcher = launcher[1:]
            if launcher[:1] in (["-S"], ["--split-string"]):
                launcher = launcher[1:]
                if len(launcher) == 1:
                    launcher = shlex.split(launcher[0])
            elif len(launcher) != 1:
                raise _error("multiple env shebang arguments require explicitly using -S")
        if not launcher or not _PYTHON_NAME.fullmatch(Path(launcher[0]).name):
            raise _error("the shebang does not point to a recognizable Python interpreter")
        return [_executable(launcher[0], cwd, env), *_python_options(launcher[1:])]
    except ValueError as exc:
        if str(exc).startswith("Cannot confirm the Hermes long-prompt entrypoint"):
            raise
        raise _error("cannot parse the Python shebang") from exc


def build_command(cli, prompt_path, cwd, env, extra_args):
    """Return a small command; the complete prompt appears only inside the child."""
    entrypoint = Path(_executable(os.fspath(cli), cwd, env)).resolve(strict=True)
    source = _entry_source(entrypoint)
    python = _interpreter(source, cwd, env)
    prompt = Path(prompt_path).absolute()
    if not prompt.is_file():
        raise _error(f"the prompt file does not exist: {prompt}")
    return [*python, str(Path(__file__).resolve()), str(entrypoint), str(prompt), *extra_args]


def _guard_prompt_reexec(prompt):
    """Catch Hermes managed-container/re-exec paths without reading its config."""
    if len(prompt.encode("utf-8")) < _REEXEC_PROMPT_BYTES:
        return

    def audit(event, args):
        if event not in {"os.exec", "subprocess.Popen"} or len(args) < 2:
            return
        command = args[1]
        if isinstance(command, (list, tuple)) and any(part == prompt for part in command):
            raise RuntimeError(
                "Hermes tried to pass the full long prompt into a subprocess command line again; stopped to avoid E2BIG."
                "A Hermes managed container or re-exec mode may be enabled; run its Python CLI directly inside the target environment."
            )

    sys.addaudithook(audit)


def _main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2:
        raise _error("the bridge is missing the entrypoint or prompt file path")
    entrypoint, prompt_path, *extra_args = args
    _entry_source(entrypoint)
    # Binary decode preserves CRLF and all prompt bytes; no universal-newline
    # conversion, text stripping, interpolation, or shell expansion.
    prompt = Path(prompt_path).read_bytes().decode("utf-8")
    _guard_prompt_reexec(prompt)
    sys.argv = [entrypoint, "-z", prompt, *extra_args]
    if not getattr(sys.flags, "safe_path", False):
        # run_path(file) does not supply the console script's import directory.
        # Replace the bridge's lib/ path so sibling imports behave as before.
        if sys.path:
            sys.path[0] = str(Path(entrypoint).parent)
        else:
            sys.path.insert(0, str(Path(entrypoint).parent))
    runpy.run_path(entrypoint, run_name="__main__")


if __name__ == "__main__":
    _main()
