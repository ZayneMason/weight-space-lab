import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

SCRIPT = Path(__file__).resolve().parents[1] / "skills/weight-space-lab/scripts/weight_lab.py"
spec = importlib.util.spec_from_file_location("weight_lab", SCRIPT)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class WeightLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original.npz"
        self.weights = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        np.savez(self.source, weight=self.weights, bias=np.zeros(2, dtype=np.float32))
        self.sha = lab.digest(self.source)

    def patch(self, operation=None, **changes):
        spec = {"base_sha256": self.sha, "reason": "test repair",
                "operations": [operation or {"op": "set", "tensor": "weight", "indices": [0], "values": [9]}]}
        spec.update(changes)
        path = self.root / "edit.json"
        path.write_text(json.dumps(spec))
        return path

    def test_npz_preserves_source_unedited_values_and_dtype(self):
        receipt = lab.patch_checkpoint(self.source, self.patch(), self.root / "candidate", max_fraction=1)
        self.assertEqual(lab.digest(self.source), self.sha)
        result, _ = lab.load_checkpoint(self.root / "candidate/model.npz")
        np.testing.assert_array_equal(result["weight"], [[9, 2], [3, 4]])
        np.testing.assert_array_equal(result["bias"], np.zeros(2))
        self.assertEqual(result["weight"].dtype, self.weights.dtype)
        self.assertEqual(receipt["changed"], 1)
        self.assertEqual(receipt["fraction"], 1 / 6)

    def test_safetensors_retains_metadata(self):
        self.source = self.root / "original.safetensors"
        save_file({"weight": self.weights}, self.source, metadata={"format": "pt"})
        self.sha = lab.digest(self.source)
        lab.patch_checkpoint(self.source, self.patch(), self.root / "candidate", max_fraction=1)
        with safe_open(self.root / "candidate/model.safetensors", framework="numpy") as f:
            self.assertEqual(f.metadata(), {"format": "pt"})
        self.assertEqual(load_file(self.root / "candidate/model.safetensors")["weight"][0, 0], 9)
        self.assertEqual(lab.digest(self.source), self.sha)

    def test_wrong_hash_rejected_before_output(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            lab.patch_checkpoint(self.source, self.patch(base_sha256="0" * 64), self.root / "candidate", max_fraction=1)
        self.assertFalse((self.root / "candidate").exists())

    def test_budget_counts_actual_changes(self):
        with self.assertRaisesRegex(ValueError, "fraction"):
            lab.patch_checkpoint(self.source, self.patch(), self.root / "candidate")
        self.assertFalse((self.root / "candidate").exists())
        self.assertEqual(lab.digest(self.source), self.sha)

    def test_existing_output_is_untouched(self):
        target = self.root / "candidate"
        target.mkdir()
        (target / "keep").write_text("keep")
        with self.assertRaises(FileExistsError):
            lab.patch_checkpoint(self.source, self.patch(), target, max_fraction=1)
        self.assertEqual((target / "keep").read_text(), "keep")

    def test_nonfinite_and_overflow_rejected(self):
        for value in (float("nan"), float("inf"), 1e100):
            with self.subTest(value=value), self.assertRaises(ValueError):
                lab.patch_checkpoint(self.source, self.patch({"op": "set", "tensor": "weight", "indices": [0], "values": [value]}), self.root / "candidate", max_fraction=1)
        self.assertFalse((self.root / "candidate").exists())

    def test_invalid_indices_and_fields_rejected(self):
        for ix in ([-1], [4], [0, 0], [True], [0.5], []):
            with self.subTest(ix=ix), self.assertRaises(ValueError):
                lab.edit_tensor(self.weights.copy(), {"op": "set", "tensor": "weight", "indices": ix, "values": [1] * len(ix)})
        with self.assertRaises(ValueError):
            lab.edit_tensor(self.weights.copy(), {"op": "set", "tensor": "weight", "indices": [0], "values": [5], "typo": True})

    def test_projection_removes_only_selected_component(self):
        weights = self.weights.copy()
        lab.edit_tensor(weights, {"op": "remove_direction", "tensor": "weight", "columns": [1], "direction": [1, 0], "strength": 1})
        np.testing.assert_array_equal(weights, [[1, 0], [3, 4]])

    def test_add_scale_and_noop(self):
        weights = self.weights.copy()
        lab.edit_tensor(weights, {"op": "add", "tensor": "weight", "indices": [1], "values": [2]})
        lab.edit_tensor(weights, {"op": "scale", "tensor": "weight", "indices": [1], "values": [-2]})
        self.assertEqual(weights[0, 1], -8)
        with self.assertRaisesRegex(ValueError, "no stored weights"):
            lab.patch_checkpoint(self.source, self.patch({"op": "set", "tensor": "weight", "indices": [0], "values": [1]}), self.root / "candidate", max_fraction=1)

    def test_inspection_slice_and_memory_limits(self):
        report = lab.describe(self.source)
        self.assertEqual(report["parameters"], 6)
        self.assertEqual(lab.tensor_slice(self.source, "weight", 1, 2)["values"], [2, 3])
        with self.assertRaises(ValueError):
            lab.tensor_slice(self.source, "weight", count=4097)
        compressed = self.root / "compressed.npz"
        np.savez_compressed(compressed, weight=np.zeros(100_000))
        with self.assertRaisesRegex(ValueError, "Expanded"):
            lab.load_checkpoint(compressed, max_mib=0.1)

    def test_pickle_and_integer_edits_rejected(self):
        obj = self.root / "object.npz"
        np.savez(obj, weight=np.array([object()], dtype=object))
        with self.assertRaises(ValueError):
            lab.load_checkpoint(obj)
        with self.assertRaises(ValueError):
            lab.edit_tensor(np.array([1, 2]), {"op": "set", "tensor": "weight", "indices": [0], "values": [3]})

    def test_demo_scores_and_comparison_regressions(self):
        folder = self.root / "demo"
        report = lab.demo(folder)
        self.assertEqual(report["comparison"]["baseline_accuracy"], 0)
        self.assertEqual(report["comparison"]["candidate_accuracy"], 1)
        reverse = lab.compare_reports(folder / "candidate.json", folder / "baseline.json")
        self.assertEqual(len(reverse["regressed_ids"]), 6)
        self.assertEqual(reverse["accuracy_delta"], -1)

    def test_comparison_rejects_mismatch_and_recomputes_accuracy(self):
        folder = self.root / "demo"
        lab.demo(folder)
        path = folder / "candidate.json"
        report = lab.read_json(path)
        report["accuracy"] = 0.123
        path.write_text(json.dumps(report))
        self.assertEqual(lab.compare_reports(folder / "baseline.json", path)["candidate_accuracy"], 1)
        report["dataset_sha256"] = "different"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            lab.compare_reports(folder / "baseline.json", path)

    def test_evaluator_rejects_bad_shapes_and_labels(self):
        arch, dataset = self.root / "arch.json", self.root / "data.json"
        arch.write_text(json.dumps({"layers": [{"weight": "weight", "bias": "bias", "activation": "linear"}]}))
        for row in ({"id": "bad", "x": [1], "y": 0}, {"id": "bad", "x": [1, 2], "y": 3}):
            dataset.write_text(json.dumps({"split": "heldout", "examples": [row]}))
            with self.assertRaises(ValueError):
                lab.evaluate_mlp(self.source, arch, dataset)


if __name__ == "__main__":
    unittest.main()
