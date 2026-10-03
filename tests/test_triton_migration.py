"""Migration contracts: active prompts, stable fusion IDs and resumed hardware."""

import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import orchestrator
from lib.bench_parser import analyze_kernel_csv_for_anticheat
import test_fusion_routing as routing


ROOT = Path(__file__).resolve().parents[1]


class TritonMigrationTests(unittest.TestCase):
    def test_kernel_names_cannot_become_an_ownership_or_cheating_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel_details.csv"
            path.write_text("Name,Type\nCannBenchCacheClean,AI_VECTOR_CORE\n"
                            "aclnn_like_user_name,AI_VECTOR_CORE\n"
                            "some_generated_name,AI_CORE\n", encoding="utf-8-sig")
            summary = analyze_kernel_csv_for_anticheat({
                "worst_6_cases": [{"case_id": "case1", "kernel_csv": str(path)}]})
            self.assertIn("共 3 条 kernel 事件", summary)
            self.assertIn("aclnn_like_user_name", summary)
            self.assertIn("不能据此认定来源", summary)
            self.assertNotIn("全是 aclnn", summary)
            self.assertNotIn("其中自定义", summary)

    def test_active_instructions_do_not_require_the_previous_dsl(self):
        paths = [ROOT / "README.md"]
        for directory in ("roles", "knowledge", "skills", "examples/triton_ascend_example"):
            paths.extend(path for path in (ROOT / directory).rglob("*")
                         if path.suffix in {".md", ".py", ".json"})
        pattern = re.compile(r"@pl\.|\bpypto_pro\b|\bTILE_FWK_DEVICE_ID\b|\bMEMORY_BASE\b|\bREGISTER_BASE\b")
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(pattern.search(text))
        self.assertTrue((ROOT / "skills/triton-profiling-analysis/SKILL.md").is_file())

    def test_fusion_catalog_keeps_method_and_variant_identity(self):
        current = json.loads((ROOT / "knowledge/fusion_options.json").read_text(encoding="utf-8"))
        # Public IDs are a compatibility contract; no historical source is needed.
        expected = {
            "F1": {"F1.compute_epilogue_split", "F1.partial_fusion_partitions", "F1.primitive_orchestration"},
            "F2": {"F2.compute_epilogue", "F2.loop_chain", "F2.tiled_pipeline"},
            "F3": {"F3.complementary_regions", "F3.small_branches"},
            "F4": {"F4.shared_input", "F4.producer_multi_consumer", "F4.multi_output"},
            "F5": {"F5.fusion_boundary_routing", "F5.schedule_routing", "F5.specialized_with_fallback"},
            "F6": {"F6.cooperative_shared_tiles", "F6.cross_kernel_sharing"},
            "F7": {"F7.reuse_guided_partition", "F7.cost_guided_search"},
            "F8": {"F8.compute_memory_pipeline", "F8.back_to_back_compute"},
            "F9": {"F9.long_chain", "F9.branched_region"},
            "F10": {"F10.single_node", "F10.multi_node"},
        }
        def identities(catalog):
            return {method["id"]: {variant["id"] for variant in method["variants"]}
                    for method in catalog["methods"]}
        self.assertEqual(identities(current), expected)
        self.assertEqual(set(identities(current)), {f"F{i}" for i in range(1, 11)})
        self.assertEqual(sum(map(len, identities(current).values())), 24)

    def test_resume_refreshes_chip_before_downstream_generation(self):
        fixture = routing.FusionRoutingTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.checkpoint()
        previous = json.loads((fixture.work / "device_info.json").read_text(encoding="utf-8"))
        actual = dict(previous, chip_model="NewChip", soc_version="NewSoC", ai_core_num=8)
        with fixture.patches(), patch.object(orchestrator, "detect_npu_device", return_value=actual) as probe:
            orchestrator.main()
        probe.assert_called_once_with(device_id=0, config_path=str(fixture.work / "synthetic_config.yaml"))
        current = json.loads((fixture.work / "device_info.json").read_text(encoding="utf-8"))
        self.assertEqual(current["chip_model"], "NewChip")
        self.assertEqual(current["ai_core_num"], 8)
        self.assertIn("NewChip", fixture.prompts["stage2"])
        self.assertNotIn("offline_test (", fixture.prompts["stage2"])


if __name__ == "__main__":
    unittest.main()
