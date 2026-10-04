"""Ensure strict offline exclusions remain visible without weakening guards."""

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.run_jev_offline_tests import (
    LOCAL_PROCESS_SKIP_REASON,
    LOCAL_PROCESS_TEST_COMMAND,
    SafeResults,
    offline_guards,
    skip_local_process_tests,
)


class OfflineTestRunnerTests(unittest.TestCase):
    def test_local_process_tests_are_reported_as_skipped_and_class_flags_are_restored(self):
        suite = unittest.TestLoader().discover(
            str(Path(__file__).parent), pattern="test_agent_output_streams.py"
        )
        expected_count = suite.countTestCases()
        self.assertGreater(expected_count, 0)
        first_test = suite
        while isinstance(first_test, unittest.TestSuite):
            first_test = next(iter(first_test))
        test_class = first_test.__class__
        original_flags = {key: test_class.__dict__.get(key) for key in
                          ("__unittest_skip__", "__unittest_skip_why__")}
        violations = {"network": 0, "subprocess": 0, "real_secret_access": 0}
        results = SafeResults()
        with tempfile.TemporaryDirectory() as fixture_root:
            with offline_guards(fixture_root, violations), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with skip_local_process_tests(suite):
                    suite.run(results)
        self.assertTrue(results.wasSuccessful())
        self.assertEqual(results.testsRun, expected_count)
        self.assertEqual(len(results.skipped), expected_count)
        self.assertEqual(violations, {"network": 0, "subprocess": 0, "real_secret_access": 0})
        for record in results.records.values():
            self.assertEqual(record["status"], "skipped")
            self.assertEqual(record["skip_reason"], LOCAL_PROCESS_SKIP_REASON)
            self.assertEqual(record["run_separately"], LOCAL_PROCESS_TEST_COMMAND)
        self.assertEqual(original_flags, {key: test_class.__dict__.get(key) for key in original_flags})

    def test_unlisted_process_test_is_still_blocked(self):
        class UnlistedProcessTest(unittest.TestCase):
            def runTest(self):
                subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])

        suite = unittest.TestSuite([UnlistedProcessTest()])
        violations = {"network": 0, "subprocess": 0, "real_secret_access": 0}
        results = SafeResults()
        with tempfile.TemporaryDirectory() as fixture_root:
            with offline_guards(fixture_root, violations), skip_local_process_tests(suite):
                suite.run(results)
        self.assertFalse(results.wasSuccessful())
        self.assertEqual(len(results.failures), 1)
        self.assertEqual(len(results.skipped), 0)
        self.assertEqual(violations["subprocess"], 1)

    def test_arbitrary_skip_payload_is_not_published(self):
        class ArbitrarySkipTest(unittest.TestCase):
            def runTest(self):
                self.skipTest("private fixture details must not be published")

        results = SafeResults()
        ArbitrarySkipTest().run(results)
        record = next(iter(results.records.values()))
        self.assertEqual(record["status"], "skipped")
        self.assertNotIn("skip_reason", record)
        self.assertNotIn("private fixture details", repr(results.skipped))


if __name__ == "__main__":
    unittest.main()
