"""GPU placement and readiness tests use inventories only; no CUDA execution."""
import unittest

from weight_space_lab import gpu_runtime as gpu


def inventory():
    return [{"index": i, "uuid": f"GPU-fixture-{i}", "name": "Fixture GPU", "memory_total_mib": 24000,
             "memory_free_mib": 22000 if i != 1 else 6000, "driver_version": "fixture"} for i in range(4)]


class GpuLayoutTests(unittest.TestCase):
    def test_auto_uses_available_gpus_and_preserves_busy_device(self):
        plan = gpu.resolve({}, inventory(), environ={})
        self.assertTrue(plan["ready"])
        self.assertEqual([w["devices"][0]["index"] for w in plan["workers"]], [0, 2, 3])
        self.assertEqual(plan["skipped"][0]["index"], 1)

    def test_explicit_groups_resolve_uuids_and_report_memory_shortfall(self):
        plan = gpu.resolve({"device_groups": [["0"], ["GPU-fixture-1"]]}, inventory(), environ={})
        self.assertFalse(plan["ready"])
        self.assertEqual(plan["workers"][1]["visible_devices"], "GPU-fixture-1")
        self.assertIn("memory", plan["workers"][1]["reasons"][0])

    def test_tensor_and_pipeline_groups_validate_size_and_never_overlap(self):
        plan = gpu.resolve({"device_groups": [["0", "2"]], "tensor_parallel_size": 2}, inventory(), environ={})
        self.assertTrue(plan["ready"])
        self.assertEqual(plan["workers"][0]["visible_devices"], "GPU-fixture-0,GPU-fixture-2")
        for config in ({"device_groups": [["0"]], "tensor_parallel_size": 2},
                       {"device_groups": [["0"], ["0"]], "max_workers": 1},
                       {"device_groups": [["0"], ["GPU-fixture-0"]]}, {"memory_fraction": float('nan')},
                       {"tensor_parallel_size": True}, {"device_groups": [["99"]]}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                gpu.resolve(config, inventory(), environ={})

    def test_parent_visibility_mask_is_honored(self):
        plan = gpu.resolve({}, inventory(), environ={"CUDA_VISIBLE_DEVICES": "2,3"})
        self.assertEqual([w["visible_devices"] for w in plan["workers"]], ["GPU-fixture-2", "GPU-fixture-3"])
        with self.assertRaises(ValueError):
            gpu.resolve({"device_groups": [["0"]]}, inventory(), environ={"CUDA_VISIBLE_DEVICES": "2"})
        self.assertFalse(gpu.resolve({}, inventory(), environ={"CUDA_VISIBLE_DEVICES": ""})["ready"])
        override = gpu.resolve({'respect_visible_devices':False}, inventory(), environ={'CUDA_VISIBLE_DEVICES':''})
        self.assertTrue(override['ready'])

    def test_no_gpu_is_a_readiness_result_and_max_workers_limits_pool(self):
        self.assertFalse(gpu.resolve({}, [], environ={})["ready"])
        self.assertEqual(len(gpu.resolve({"max_workers": 1}, inventory(), environ={})["workers"]), 1)

    def test_inventory_parser_handles_csv_and_rejects_inconsistent_rows(self):
        rows = gpu.parse_inventory('0, GPU-a, "GPU, named", 24000, 22000, 610.0\n')
        self.assertEqual(rows[0]["name"], "GPU, named")
        for text in ('0, GPU-a, GPU, 10, 11, x\n', '0, GPU-a, GPU, 10, 8, x\n0, GPU-b, GPU, 10, 8, x\n'):
            with self.assertRaises(ValueError):
                gpu.parse_inventory(text)
