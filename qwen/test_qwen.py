import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("qwen_runner", Path(__file__).with_name("reproduce_qwen.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class QwenTests(unittest.TestCase):
    def fixture(self):
        original = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
        edited = original.clone()
        edited[0, 1] = 9
        manifest = {"changed_entries_total": 1, "tensors": [{"name": "weight", "shape": [2, 2],
                    "form": "sparse", "base_sha256": q.tensor_sha(original),
                    "edited_sha256": q.tensor_sha(edited), "changed_entries": 1}]}
        edits = {"t0_idx": np.array([1]), "t0_val": edited.view(torch.uint16).numpy().reshape(-1)[[1]]}
        return original, edited, manifest, edits

    def test_exact_bf16_reconstruction_preserves_input(self):
        original, edited, manifest, edits = self.fixture()
        tensors = {"weight": original}
        evidence = q.reconstruct(tensors, manifest, edits)
        self.assertTrue(torch.equal(tensors["weight"], edited))
        self.assertEqual(float(original[0, 1]), 2)
        self.assertEqual(evidence[0]["changed_entries"], 1)

    def test_wrong_base_hash_and_wrong_edited_hash_rejected(self):
        for key in ("base_sha256", "edited_sha256"):
            original, _, manifest, edits = self.fixture()
            manifest["tensors"][0][key] = "0" * 64
            with self.assertRaises(ValueError):
                q.reconstruct({"weight": original}, manifest, edits)

    def test_out_of_bounds_and_duplicate_indices_rejected(self):
        for indices in ([-1], [4], [1, 1]):
            original, _, manifest, edits = self.fixture()
            edits["t0_idx"] = np.array(indices)
            edits["t0_val"] = np.array([0] * len(indices), dtype=np.uint16)
            with self.assertRaises(ValueError):
                q.reconstruct({"weight": original}, manifest, edits)

    def test_changed_count_rejected(self):
        original, _, manifest, edits = self.fixture()
        manifest["changed_entries_total"] = 2
        with self.assertRaises(ValueError):
            q.reconstruct({"weight": original}, manifest, edits)

    def test_extractor_matches_released_implementation(self):
        source = Path(__file__).parent / "fixtures/mathqa_extraction.py"
        tree = ast.parse(source.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                     n.name in ("answer_text", "_box_at", "last_boxed", "extract")]
        namespace = {"_BOX_FORMS": re.compile(r"\\boxed|\\fbox")}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        for text in [r"\boxed{1} then \boxed{\frac{1}{2}}", r"\boxed{1} \boxed{2",
                     r"\fbox{3}", r"\boxed 5$", r"\boxed { 7 }", r"\boxed{}", "nothing",
                     r"\boxed{1} \fbox{2}", r"\boxed{1}</think>\boxed{4}"]:
            self.assertEqual(q.last_boxed(text), namespace["extract"](text)[0], text)

    def test_symbolic_grader_and_missing_answer(self):
        self.assertTrue(q.grade(r"\boxed{\frac{1}{2}}", "0.5")["correct"])
        self.assertFalse(q.grade(r"\boxed{2}", "3")["correct"])
        self.assertFalse(q.grade("the answer is 4", "4")["correct"])
        self.assertFalse(q.grade(r"\boxed{1} \boxed{2", "1")["correct"])

    def test_paired_comparison_rejects_mismatched_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / x for x in ("baseline", "p50c")]
            for i, folder in enumerate(dirs):
                folder.mkdir()
                q.write_json(folder / "job.json", {"arm": folder.name, "pilot": bool(i)})
            with self.assertRaises(ValueError):
                q.compare(*dirs, Path(tmp) / "comparison.json")

    def test_paired_bootstrap_keeps_four_samples_clustered(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / x for x in ("baseline", "p50c")]
            for i, folder in enumerate(dirs):
                folder.mkdir()
                job = dict(arm=folder.name, pilot=True, problem_ids=["p"], sampling={}, protocol={},
                           prepared_sha256="x", script_sha256="y", versions={}, engine={}, environment={}, batch_problems=4)
                q.write_json(folder / "job.json", job)
                filename = hashlib.sha256(b"p").hexdigest()[:20] + ".json"
                q.write_json(folder / filename, {"id": "p", "generations": [
                    {"correct": bool(i), "truncated": not bool(i)} for _ in range(4)]})
            out = Path(tmp) / "comparison.json"
            q.compare(*dirs, out)
            result = json.loads(out.read_text())
            self.assertEqual(result["metrics"]["accuracy_delta"]["mean"], 1)
            self.assertEqual(result["bootstrap_unit"], "problem (all four samples together)")


if __name__ == "__main__":
    unittest.main()
