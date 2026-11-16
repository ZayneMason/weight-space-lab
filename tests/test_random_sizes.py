import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("random_sizes", ROOT / "experiments/locate-repair/random_sizes.py")
rs = importlib.util.module_from_spec(spec)
sys.modules["random_sizes"] = rs
spec.loader.exec_module(rs)


class RandomSizesTests(unittest.TestCase):
    def setUp(self):
        self.base = rs.tf.corrupt(8, 0)

    def test_resize_keeps_entries_and_signs(self):
        case = rs.resize(self.base)
        teacher, scale = self.base["teacher"], self.base["scale"]
        for (name, index), size in zip(self.base["faults"], case["sizes"]):
            self.assertTrue(rs.LOW <= size <= rs.HIGH)
            old = float(self.base["arrays"][name].flat[index]) - float(teacher[name].flat[index])
            new = float(case["arrays"][name].flat[index]) - float(teacher[name].flat[index])
            self.assertEqual(np.sign(old), np.sign(new))
            self.assertAlmostEqual(abs(new) / scale[name], size, places=4)
        changed = sum(int((case["arrays"][n] != teacher[n]).sum()) for n in teacher)
        self.assertEqual(changed, len(self.base["faults"]))

    def test_fixed_size_reproduces_the_original_case(self):
        case = rs.resize(self.base, 1.6, 1.6)
        for name in self.base["arrays"]:
            np.testing.assert_allclose(case["arrays"][name], self.base["arrays"][name], rtol=0, atol=1e-6)

    def test_summary_counts_per_query_round(self):
        step = lambda found, visible, ok: {"queries": 0, "faults_found": found, "found_visible": visible, "success": ok}
        rows = [{"joint_rank": 0, "silent_faults": [], "candidates_have_true_pair": True, "targeted": [step(2, True, True)]},
                {"joint_rank": 60, "silent_faults": [1], "candidates_have_true_pair": False, "targeted": [step(1, True, False)]}]
        out = rs.summarize_pairs(rows)
        self.assertEqual(out["joint_rank_in_top_40"], 1)
        self.assertEqual(out["silent_cases"], 1)
        self.assertEqual(out["targeted"][0], {"queries": 0, "found_pair": 1, "found_visible": 2, "success": 1})


if __name__ == "__main__":
    unittest.main()
