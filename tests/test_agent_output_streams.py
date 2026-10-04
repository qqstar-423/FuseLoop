"""Regression coverage for real, local subprocess stdout/stderr transport."""

import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

from lib import agent_runner as runner


class AgentOutputStreamsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def run_script(self, body, *, timeout=0, plain_text=False):
        # A separate child thread bounds even a regression that blocks readline()
        # before the runner can enforce its own timeout. No agent or API is used.
        script = (
            "import os, sys, threading, time\n"
            "guard = threading.Timer(4.0, lambda: os._exit(97))\n"
            "guard.daemon = True\n"
            "guard.start()\n"
            + textwrap.dedent(body)
        )
        self.assertTrue(script.isascii(), "Keep child command encoding independent of locale")
        result = runner._run_with_heartbeat(
            [sys.executable, "-u", "-c", script],
            cwd=str(self.work),
            env=os.environ.copy(),
            agent_name="local-stream-test",
            timeout=timeout,
            plain_text=plain_text,
        )
        self.assertNotEqual(result.returncode, 97, "Child guard fired: output transport stalled")
        return result

    def test_utf8_survives_split_writes_and_8192_byte_boundary(self):
        stdout = "a" * 8190 + "夜间输出：完整中文\n"
        stderr = "diagnostic info: Chinese retained as well\n"
        out_bytes = stdout.encode("utf-8")
        err_bytes = stderr.encode("utf-8")
        # The third byte of the first Chinese character is 0x9c, just beyond an 8192-byte read.
        self.assertEqual(out_bytes[8192], 0x9C)
        result = self.run_script(f"""
            out = {out_bytes!r}
            err = {err_bytes!r}
            sys.stdout.buffer.write(out[:8192])
            sys.stderr.buffer.write(err[:1])
            time.sleep(0.03)
            sys.stdout.buffer.write(out[8192:8193])
            sys.stderr.buffer.write(err[1:2])
            time.sleep(0.03)
            sys.stdout.buffer.write(out[8193:])
            sys.stderr.buffer.write(err[2:])
        """)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, stdout)
        self.assertEqual(result.stderr, stderr)

    def test_unterminated_line_does_not_block_large_other_stream(self):
        for partial_stream, busy_stream in (("stdout", "stderr"), ("stderr", "stdout")):
            with self.subTest(partial_stream=partial_stream):
                result = self.run_script(f"""
                    sys.{partial_stream}.buffer.write(b'partial line')
                    time.sleep(0.03)
                    sys.{busy_stream}.buffer.write(b'x' * (256 * 1024) + b'\\n')
                    sys.{partial_stream}.buffer.write(b' completed\\n')
                """)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(getattr(result, partial_stream), "partial line completed\n")
                self.assertEqual(getattr(result, busy_stream), "x" * (256 * 1024) + "\n")

    def test_fast_exit_collects_all_lines_and_unterminated_tails(self):
        result = self.run_script("""
            for index in range(3000):
                sys.stdout.buffer.write(('stdout:%d\\n' % index).encode('ascii'))
                sys.stderr.buffer.write(('stderr:%d\\n' % index).encode('ascii'))
            sys.stdout.buffer.write(b'stdout-tail')
            sys.stderr.buffer.write(b'stderr-tail')
            os._exit(0)
        """)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "".join(f"stdout:{index}\n" for index in range(3000))
                         + "stdout-tail")
        self.assertEqual(result.stderr, "".join(f"stderr:{index}\n" for index in range(3000))
                         + "stderr-tail")

    def test_invalid_stderr_bytes_are_visible_and_keep_nonzero_status(self):
        stderr_bytes = b"invalid=\x9c\nvalid=" + "中文".encode("utf-8") + b"\npartial=\xe4\xb8"
        result = self.run_script(f"""
            sys.stdout.buffer.write(b'standard output stays separate\\n')
            sys.stderr.buffer.write({stderr_bytes!r})
            raise SystemExit(23)
        """)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(result.stdout, "standard output stays separate\n")
        self.assertEqual(result.stderr, "invalid=\\x9c\nvalid=中文\npartial=\\xe4\\xb8")

    def test_timeout_and_heartbeat_continue_during_unterminated_output(self):
        start = time.monotonic()
        with patch.object(runner, "HEARTBEAT_INTERVAL", 0.05), \
                self.assertLogs(runner.log, level="INFO") as logs:
            result = self.run_script("""
                while True:
                    sys.stdout.buffer.write(b'ongoing stdout ')
                    sys.stderr.buffer.write(b'ongoing stderr ')
                    time.sleep(0.03)
            """, timeout=0.4)
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stdout.startswith("ongoing stdout "))
        self.assertTrue(result.stderr.startswith("ongoing stderr "))
        logged = "\n".join(logs.output)
        self.assertIn("still running", logged)
        self.assertIn("timed out", logged)

    def test_closed_output_streams_still_wait_for_process_and_enforce_timeout(self):
        start = time.monotonic()
        with patch.object(runner, "HEARTBEAT_INTERVAL", 0.05), \
                self.assertLogs(runner.log, level="INFO") as logs:
            result = self.run_script("""
                os.close(1)
                os.close(2)
                time.sleep(3.5)
            """, timeout=0.4)
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        logged = "\n".join(logs.output)
        self.assertIn("still running", logged)
        self.assertIn("timed out", logged)

    @staticmethod
    def api_error_line():
        return json.dumps({"type": "error", "error": {"data": {"message": "synthetic API error"}}},
                          separators=(",", ":")) + "\n"

    def test_five_consecutive_api_errors_terminate_child(self):
        errors = self.api_error_line() * 5
        with self.assertLogs(runner.log, level="ERROR") as logs:
            result = self.run_script(f"""
                sys.stdout.buffer.write({errors.encode('utf-8')!r})
                time.sleep(3.5)
            """, timeout=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, errors)
        self.assertEqual(result.stderr, "")
        self.assertIn("5 consecutive API errors", "\n".join(logs.output))
        self.assertNotIn("timed out", "\n".join(logs.output))

    def test_normal_json_output_resets_consecutive_api_error_count(self):
        normal = json.dumps({"type": "text", "part": {"text": "synthetic recovery"}},
                            separators=(",", ":")) + "\n"
        output = self.api_error_line() * 4 + normal + self.api_error_line() * 4
        with self.assertLogs(runner.log, level="INFO") as logs:
            result = self.run_script(f"""
                sys.stdout.buffer.write({output.encode('utf-8')!r})
                time.sleep(0.15)
            """)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, output)
        self.assertEqual(result.stderr, "")
        self.assertIn("synthetic recovery", "\n".join(logs.output))
        self.assertNotIn("terminating the process", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
