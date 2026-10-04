"""Offline Hermes transport checks using only temporary in-process CLI stubs."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from lib import hermes_prompt


CONSOLE = '''import sys
from hermes_cli.main import main
if __name__ == "__main__":
    sys.exit(main())
'''


class HermesPromptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "hermes 中文 with spaces"  # non-ASCII path is intentional
        self.bin = self.root / "bin"
        self.bin.mkdir(parents=True)
        self.entry = self.bin / "hermes"
        self.prompt = self.root / "prompt 中文.txt"  # non-ASCII filename is intentional
        self.prompt.write_bytes("initial full prompt\r\n".encode("utf-8"))
        self.cwd = self.root / "working directory"
        self.cwd.mkdir()
        # Never inherit a user's provider credentials, HOME or Hermes config.
        self.env = {"PATH": str(Path(sys.executable).parent), "PYTHONIOENCODING": "utf-8"}
        if os.name == "nt":
            self.env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        self.write_entry()

    def write_entry(self, header=None, body=CONSOLE):
        if header is None:
            # Real pip/distlib wrapper; supports spaces in the interpreter path.
            header = "#!/bin/sh\n'''exec' " + shlex.quote(Path(sys.executable).as_posix()) + ' "$0" "$@"\n\' \'\'\'\n'
        self.entry.write_text(header + body, encoding="utf-8", newline="\n")
        self.entry.chmod(0o755)

    def command(self, extra_args=()):
        return hermes_prompt.build_command(str(self.entry), self.prompt, self.cwd,
                                          self.env, list(extra_args))

    def install_stub(self, *, returncode=0):
        package = self.bin / "hermes_cli"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (self.bin / "sibling_marker.py").write_text("MARKER = 'entrypoint sibling'\n", encoding="utf-8")
        (package / "main.py").write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "from sibling_marker import MARKER\n"
            "def main():\n"
            "    Path(os.environ['CAPTURE']).write_text(json.dumps({\n"
            "        'argv': sys.argv, 'cwd': os.getcwd(), 'marker': MARKER,\n"
            "        'executable': sys.executable, 'path0': sys.path[0]}, ensure_ascii=False), encoding='utf-8')\n"
            "    print('synthetic output 中文', flush=True)  # non-ASCII output is intentional\n"
            f"    return {returncode}\n", encoding="utf-8")
        self.env["CAPTURE"] = str(self.root / "capture.json")

    def run_bridge(self, command):
        """Exercise real runpy, without starting any child or loading real Hermes."""
        script_index = command.index(str(Path(hermes_prompt.__file__).resolve()))
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", command[script_index:]))
            stack.enter_context(patch.object(sys, "path", list(sys.path)))
            stack.enter_context(patch.dict(sys.modules))
            for name in ("hermes_cli", "hermes_cli.main", "sibling_marker"):
                sys.modules.pop(name, None)
            stack.enter_context(patch.dict(os.environ, self.env, clear=True))
            stack.enter_context(patch.object(sys, "addaudithook"))
            stack.enter_context(redirect_stdout(output))
            stack.callback(os.chdir, os.getcwd())
            os.chdir(self.cwd)
            with self.assertRaises(SystemExit) as stopped:
                hermes_prompt._main()
        return stopped.exception.code, output.getvalue()

    def test_multimegabyte_input_reaches_same_cli_with_all_flags_and_newlines(self):
        self.install_stub()
        prompt = "中文原文 '引号' $(no-shell) `no-shell`\r\n" * 50000  # non-ASCII payload is intentional
        self.assertGreater(len(prompt.encode("utf-8")), 2 * 1024 * 1024)
        self.prompt.write_bytes(prompt.encode("utf-8"))
        extra = ["-t", "web,file", "--yolo", "--in", str(self.cwd)]
        command = self.command(extra)
        self.assertNotIn(prompt, command)
        self.assertLess(sum(len(part.encode("utf-8")) for part in command), 4096)
        returncode, output = self.run_bridge(command)
        self.assertEqual(returncode, 0)
        self.assertIn("synthetic output 中文", output)
        actual = json.loads(Path(self.env["CAPTURE"]).read_text(encoding="utf-8"))
        self.assertEqual(actual["argv"], [str(self.entry.resolve()), "-z", prompt, *extra])
        self.assertEqual(actual["cwd"], str(self.cwd))
        self.assertEqual(actual["marker"], "entrypoint sibling")
        self.assertEqual(actual["path0"], str(self.bin))
        self.assertEqual(actual["executable"], sys.executable)

    def test_original_console_exit_code_propagates(self):
        self.install_stub(returncode=7)
        returncode, _ = self.run_bridge(self.command())
        self.assertEqual(returncode, 7)

    def test_direct_python_shebang_preserves_interpreter_and_options(self):
        self.write_entry("#!" + shlex.quote(sys.executable) + " -u\n")
        command = self.command()
        self.assertEqual(command[:2], [sys.executable, "-u"])

    def test_env_python_and_split_string_use_supplied_path(self):
        for shebang, flags in [
            ("#!/usr/bin/env python3\n", []),
            ("#!/usr/bin/env -S python3 -u -X utf8\n", ["-u", "-X", "utf8"]),
        ]:
            with self.subTest(shebang=shebang):
                self.write_entry(shebang)
                with patch.object(hermes_prompt.shutil, "which", return_value=sys.executable) as which:
                    command = self.command()
                which.assert_called_once_with("python3", path=self.env["PATH"])
                self.assertEqual(command[:1 + len(flags)], [sys.executable, *flags])

    def test_venv_interpreter_is_not_resolved_to_system_python(self):
        python = self.root / "venv" / "bin" / ("python.exe" if os.name == "nt" else "python3")
        python.parent.mkdir(parents=True)
        python.write_text("synthetic interpreter", encoding="utf-8")
        python.chmod(0o755)
        self.write_entry("#!" + shlex.quote(str(python)) + "\n")
        original = Path.resolve

        def no_interpreter_resolve(path, *args, **kwargs):
            self.assertNotEqual(path, python)
            return original(path, *args, **kwargs)

        with patch.object(Path, "resolve", no_interpreter_resolve):
            command = self.command()
        self.assertEqual(command[0], str(python))

    def test_cli_symlink_uses_the_original_console_script(self):
        link = self.root / "hermes_link"
        try:
            link.symlink_to(self.entry)
        except OSError as exc:
            self.skipTest(f"Host does not permit test symlinks: {exc}")
        command = hermes_prompt.build_command(str(link), self.prompt, self.cwd, self.env, [])
        self.assertEqual(command[-2:], [str(self.entry.resolve()), str(self.prompt)])

    def test_bare_cli_is_found_on_the_supplied_path(self):
        with patch.object(hermes_prompt.shutil, "which", return_value=str(self.entry)) as which:
            command = hermes_prompt.build_command("hermes", self.prompt, self.cwd, self.env, [])
        which.assert_called_once_with("hermes", path=self.env["PATH"])
        self.assertEqual(command[-2], str(self.entry.resolve()))

    def test_relative_path_entries_use_child_cwd_for_cli_and_env_python(self):
        environment = dict(self.env, PATH=os.pathsep.join(["bin", "", str(self.bin)]))
        expected = os.pathsep.join([str(self.cwd / "bin"), str(self.cwd) + os.sep, str(self.bin)])
        with patch.object(hermes_prompt.shutil, "which", return_value=str(self.entry)) as which:
            hermes_prompt.build_command("hermes", self.prompt, self.cwd, environment, [])
        which.assert_called_once_with("hermes", path=expected)
        self.write_entry("#!/usr/bin/env python3\n")
        with patch.object(hermes_prompt.shutil, "which", return_value=sys.executable) as which:
            hermes_prompt.build_command(str(self.entry), self.prompt, self.cwd, environment, [])
        which.assert_called_once_with("python3", path=expected)

    def test_unknown_binary_shell_or_other_console_entry_is_rejected(self):
        samples = [
            b"\x7fELF\x00binary",
            b"#!/bin/sh\nexec unknown-command\n",
            ("#!" + shlex.quote(sys.executable) + "\nfrom another_package import main\nmain()\n").encode(),
            ("#!/bin/sh\n'''exec' " + shlex.quote(sys.executable)
             + ' "$0" "$@"; echo unexpected\n\' \'\'\'\n' + CONSOLE).encode(),
        ]
        for source in samples:
            with self.subTest(source=source[:30]):
                self.entry.write_bytes(source)
                with self.assertRaisesRegex(ValueError, "will not fall back"):
                    self.command()

    def test_env_assignments_and_python_script_modes_are_not_guessed(self):
        for header in (
            "#!/usr/bin/env TOKEN=not-a-secret python3\n",
            "#!/usr/bin/env python3 -u\n",
            "#!/usr/bin/env -S python3 -m unrelated\n",
            "#!/usr/bin/env -S python3 -c unrelated\n",
        ):
            with self.subTest(header=header):
                self.write_entry(header)
                with patch.object(hermes_prompt.shutil, "which", return_value=sys.executable):
                    with self.assertRaises(ValueError):
                        self.command()

    def test_missing_prompt_is_reported_before_launch(self):
        self.prompt.unlink()
        with self.assertRaisesRegex(ValueError, "the prompt file does not exist"):
            self.command()

    def test_reexec_guard_only_blocks_the_same_large_prompt(self):
        prompt = "long prompt " * 30000
        with patch.object(hermes_prompt.sys, "addaudithook") as add:
            hermes_prompt._guard_prompt_reexec(prompt)
        audit = add.call_args.args[0]
        for event in ("os.exec", "subprocess.Popen"):
            audit(event, ("python", ["python", "small unrelated tool argument"], {}))
            audit(event, ("python", ["python", "unrelated" * 30000], {}))
            with self.assertRaisesRegex(RuntimeError, "into a subprocess command line again"):
                audit(event, ("hermes", ["hermes", "-z", prompt], {}))
        audit("unrelated.audit.event", ("hermes", [prompt]))
        with patch.object(hermes_prompt.sys, "addaudithook") as add:
            hermes_prompt._guard_prompt_reexec("small prompt")
        add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
