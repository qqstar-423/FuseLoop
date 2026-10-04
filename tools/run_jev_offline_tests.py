"""Run the repository tests with network, process and real-secret reads blocked.

Tests may install narrower mocks over these guards. Only synthetic credential
files created inside this invocation's private temporary directory are readable.
The public report contains test IDs/statuses, never exception text or tracebacks.
Real local subprocess transport tests are explicitly skipped and reported; run
them separately with the command recorded in the report.
"""

import argparse
import builtins
from contextlib import contextmanager, ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SECRET_NAMES = {"config.yaml", "config.local.yaml", "all_api_key.md"}
LOCAL_PROCESS_TEST_MODULES = frozenset({"test_agent_output_streams", "tests.test_agent_output_streams"})
LOCAL_PROCESS_SKIP_REASON = (
    "Requires real local Python subprocesses; the strict offline runner blocks subprocess.Popen."
)
LOCAL_PROCESS_TEST_COMMAND = (
    "python3 -m unittest discover -s tests -p 'test_agent_output_streams.py' -v"
)


@contextmanager
def offline_guards(fixture_root, violations):
    """Deny external effects without inspecting environment or credential values."""
    fixture_root = Path(fixture_root).resolve()

    def deny(kind):
        def blocked(*args, **kwargs):
            violations[kind] += 1
            raise AssertionError("Offline test guard blocked " + kind)
        return blocked

    def guarded_open(original):
        def open_file(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(file)).absolute()
                resolved = path.resolve()
                if (path.name.lower() in SECRET_NAMES or resolved.name.lower() in SECRET_NAMES) \
                        and not resolved.is_relative_to(fixture_root):
                    violations["real_secret_access"] += 1
                    raise AssertionError("Offline test guard blocked real credential file access")
            return original(file, *args, **kwargs)
        return open_file

    with ExitStack() as stack:
        # TypeSafe SDK constructs its User-Agent during import with machine().
        # On Windows this calls platform._syscmd_ver -> subprocess.check_output.
        # Stub only that read-only metadata query; never exempt an actual Popen.
        stack.enter_context(patch.object(platform, "machine", return_value="offline-test-machine"))
        stack.enter_context(patch.object(socket.socket, "connect", deny("network")))
        stack.enter_context(patch.object(socket.socket, "connect_ex", deny("network")))
        stack.enter_context(patch.object(socket, "create_connection", deny("network")))
        stack.enter_context(patch.object(socket, "getaddrinfo", deny("network")))
        stack.enter_context(patch.object(socket, "gethostbyname", deny("network")))
        stack.enter_context(patch.object(socket, "gethostbyname_ex", deny("network")))
        stack.enter_context(patch.object(socket, "gethostbyaddr", deny("network")))
        stack.enter_context(patch.object(socket.socket, "sendto", deny("network")))
        stack.enter_context(patch.object(subprocess, "Popen", deny("subprocess")))
        stack.enter_context(patch.object(builtins, "open", guarded_open(builtins.open)))
        stack.enter_context(patch.object(io, "open", guarded_open(io.open)))
        stack.enter_context(patch.object(os, "open", guarded_open(os.open)))
        stack.enter_context(patch.object(tempfile, "tempdir", str(fixture_root)))
        yield


@contextmanager
def skip_local_process_tests(suite):
    """Temporarily exclude declared process integration tests, keeping every guard."""
    def test_classes(node):
        if isinstance(node, unittest.TestSuite):
            for child in node:
                yield from test_classes(child)
        elif node.__class__.__module__ in LOCAL_PROCESS_TEST_MODULES:
            yield node.__class__

    with ExitStack() as stack:
        for test_class in set(test_classes(suite)):
            stack.enter_context(patch.object(test_class, "__unittest_skip__", True, create=True))
            stack.enter_context(patch.object(test_class, "__unittest_skip_why__",
                                            LOCAL_PROCESS_SKIP_REASON, create=True))
        yield


class SafeResults(unittest.TestResult):
    """Keep individual outcomes without storing arbitrary failure payloads."""

    def __init__(self):
        super().__init__()
        self.records = {}

    def _identity(self, test):
        name = test.id()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
            name = "nonstandard_test_" + hashlib.sha256(name.encode()).hexdigest()[:16]
        return name

    def startTest(self, test):
        super().startTest(test)
        self.records[self._identity(test)] = {"name": self._identity(test), "status": "running"}

    def _record(self, test, status, error=None):
        identity = self._identity(test)
        # setUpClass/tearDownClass errors have no preceding startTest callback.
        record = self.records.setdefault(identity, {"name": identity})
        record["status"] = status
        if error:
            record["error_type"] = error[0].__name__

    def addSuccess(self, test):
        self._record(test, "passed")

    def addFailure(self, test, err):
        self.failures.append((test, "Failure details withheld from the public report."))
        self._record(test, "failed", err)

    def addError(self, test, err):
        self.errors.append((test, "Error details withheld from the public report."))
        self._record(test, "error", err)

    def addSkip(self, test, reason):
        self.skipped.append((test, "Skip reason withheld from the public report."))
        if hasattr(test, "test_case"):
            record = self.records[self._identity(test.test_case)]
            record["skipped_subtests"] = record.get("skipped_subtests", 0) + 1
            return
        self._record(test, "skipped")
        # Publish only our fixed exclusion reason, never arbitrary test payloads.
        if reason == LOCAL_PROCESS_SKIP_REASON:
            record = self.records[self._identity(test)]
            record["skip_reason"] = LOCAL_PROCESS_SKIP_REASON
            record["run_separately"] = LOCAL_PROCESS_TEST_COMMAND

    def addExpectedFailure(self, test, err):
        self.expectedFailures.append((test, "Expected failure details withheld."))
        self._record(test, "expected_failure")

    def addUnexpectedSuccess(self, test):
        self.unexpectedSuccesses.append(test)
        self._record(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        # Subtest parameter repr may contain a local path or credential fixture.
        record = self.records[self._identity(test)]
        record["subtests"] = record.get("subtests", 0) + 1
        if err:
            record["failed_subtests"] = record.get("failed_subtests", 0) + 1
            if issubclass(err[0], test.failureException):
                self.addFailure(test, err)
            else:
                self.addError(test, err)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/jev_tests/results.json")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    violations = {"network": 0, "subprocess": 0, "real_secret_access": 0}
    results = SafeResults()
    discovery_error = None
    with tempfile.TemporaryDirectory(prefix="jev_library_offline_") as fixture_root:
        # Test stdout/stderr may contain fixture values. Do not save or replay it.
        with offline_guards(fixture_root, violations), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                suite = unittest.TestLoader().discover(str(ROOT / "tests"), pattern="test_*.py")
                with skip_local_process_tests(suite):
                    suite.run(results)
            except Exception as exc:
                discovery_error = type(exc).__name__
    records = sorted(results.records.values(), key=lambda record: record["name"])
    statuses = ("passed", "failed", "error", "skipped", "expected_failure", "unexpected_success", "running")
    counts = {status: sum(record["status"] == status for record in records) for status in statuses}
    passed = (results.testsRun > 0 and results.wasSuccessful() and not discovery_error
              and not any(violations.values()) and counts["running"] == 0)
    source_paths = [ROOT / "orchestrator.py", Path(__file__).resolve()]
    for directory in ("lib", "tests"):
        source_paths.extend(sorted((ROOT / directory).glob("*.py")))
    report = {
        "schema_version": "jev-library-tests/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "python_version": sys.version.split()[0],
        "discovery": "tests/test_*.py",
        "offline_guards": {"network_connections": "blocked", "subprocess_popen": "blocked",
                           "platform_machine": "mocked_read_only_sdk_runtime_metadata",
                           "real_credential_files": "blocked", "synthetic_credentials": "private_temp_fixtures_only",
                           "blocked_attempts": violations},
        "tests_run": results.testsRun, "counts": counts, "tests": records,
        "tested_sources_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in source_paths if path.is_file()},
        "limitations": ["Synthetic measurements and mocked external services only.",
                        "No real NPU, compiler, CANN-Bench, SSH, Jev or Kerminal execution was tested.",
                        LOCAL_PROCESS_SKIP_REASON + " Run separately: " + LOCAL_PROCESS_TEST_COMMAND,
                        "Each test method counts once; parameterized subtests are not separate test methods.",
                        "Stage1.5, fusion evidence, semantic exit, best snapshots and existing knowledge routes are covered offline."],
    }
    if discovery_error:
        report["discovery_error_type"] = discovery_error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Offline tests: {report['status']}; methods={results.testsRun}; counts={json.dumps(counts)}")
    print("Detailed outcomes are in the requested JSON report; failure payloads are omitted.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
