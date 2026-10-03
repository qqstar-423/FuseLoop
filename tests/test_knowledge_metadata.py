"""Knowledge records must describe their evidence's environment, not today's guess."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from lib.knowledge_metadata import (build_knowledge_environment, environment_summary,
                                    log_knowledge_write)


class KnowledgeMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.device = {"chip_model": "LiveChip", "soc_version": "LiveSoC",
                       "programming_model": "Triton block program (SPMD)"}
        (self.work / "device_info.json").write_text(json.dumps(self.device), encoding="utf-8")

    def test_performance_environment_comes_from_frozen_report_not_live_device(self):
        report = self.work / "perf_result.json"
        context = {"framework": "Triton", "backend": "triton-ascend", "hardware": {"chip_model": "MeasuredChip", "soc_version": "MeasuredSoC",
                                "programming_model": "Triton block program (SPMD)", "npu_arch": "MeasuredArch"},
                   "evaluation_device_id": 3, "extra_config": "DO_NOT_COPY_CONFIG"}
        report.write_text(json.dumps({"comparison_context": context}), encoding="utf-8")
        environment = build_knowledge_environment(self.work, performance_report=report,
                                                   comparison_context={"hardware": self.device})
        self.assertEqual(environment["framework"], "Triton")
        self.assertEqual(environment["framework_source"], "performance_report")
        self.assertEqual(environment["chip_model"], "MeasuredChip")
        self.assertEqual(environment["soc_version"], "MeasuredSoC")
        self.assertEqual(environment["device_id"], 3)
        self.assertEqual(environment["source_path"], str(report.resolve()))
        self.assertEqual(environment["source_kind"], "performance_report")
        self.assertNotIn("DO_NOT_COPY_CONFIG", json.dumps(environment))

    def test_missing_legacy_or_invalid_report_does_not_borrow_live_hardware(self):
        report = self.work / "perf_result.json"
        for content in (None, "{}", "[]", "{broken", '{"comparison_context":{"hardware":null}}'):
            if content is not None:
                report.write_text(content, encoding="utf-8")
            with self.subTest(content=content):
                environment = build_knowledge_environment(self.work, performance_report=report)
                self.assertEqual(environment["chip_model"], "unknown")
                self.assertEqual(environment["soc_version"], "unknown")
                self.assertEqual(environment["source_kind"], "performance_report")
                self.assertEqual(environment["framework"], "unknown")

    def test_fault_review_uses_workflow_context_then_device_file(self):
        environment = build_knowledge_environment(self.work, comparison_context={
            "hardware": {"chip_model": "ReviewChip", "soc_version": "ReviewSoC"},
            "evaluation_device_id": 0})
        self.assertEqual(environment["chip_model"], "ReviewChip")
        self.assertEqual(environment["source_kind"], "workflow_context")
        fallback = build_knowledge_environment(self.work)
        self.assertEqual(fallback["chip_model"], "LiveChip")
        self.assertEqual(fallback["source_kind"], "device_info")
        (self.work / "device_info.json").unlink()
        missing = build_knowledge_environment(self.work)
        self.assertEqual(missing["chip_model"], "unknown")
        self.assertIsNone(missing["source_path"])

    def test_summary_exposes_legacy_unknowns_and_does_not_infer_a_framework(self):
        for value in (None, {}, "malformed"):
            self.assertIn("未记录", environment_summary(value))
            self.assertNotIn("PyPTO", environment_summary(value))
        self.assertIn("芯片=未记录", environment_summary({"framework": "Triton", "chip_model": "unknown"}))

    def test_both_logs_include_environment_output_and_provenance_once(self):
        log, state_log = Mock(), Mock()
        environment = build_knowledge_environment(self.work)
        output = self.work / "knowledge/proven_patterns.md"
        decision = self.work / "knowledge/stage9/iter2/req/decision.json"
        kwargs = dict(label="成功经验", path=output, iteration=2, environment=environment,
                      decision_path=decision, detail="avg_speedup=1.0→1.1；变化=10%")
        log_knowledge_write(log, state_log, **kwargs)
        for logger in (log, state_log):
            logger.info.assert_called_once()
            args = logger.info.call_args.args
            rendered = args[0] % args[1:]
            for value in ("Triton", "LiveChip", "LiveSoC", str(output.resolve()),
                          str(output.parent.resolve()), str(decision), "变化=10%"):
                self.assertIn(value, rendered)
        log.reset_mock()
        log_knowledge_write(log, log, **kwargs)
        log.info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
