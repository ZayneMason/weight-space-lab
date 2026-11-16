import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("locate_repair", ROOT / "experiments/locate-repair/analyze.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)


class LocateRepairTests(unittest.TestCase):
    def test_generation_is_deterministic_and_visible(self):
        a, b = lr.generate(8, 1), lr.generate(8, 1)
        self.assertEqual(a["corrupted"], b["corrupted"])
        for name in lr.TENSORS:
            np.testing.assert_array_equal(a["arrays"][name], b["arrays"][name])
        _, _, repair = lr.split(a["practice"])
        self.assertEqual(int(repair.sum()), 16)
        name, index = a["corrupted"]
        self.assertNotEqual(a["arrays"][name].flat[index], a["teacher"][name].flat[index])

    def test_fast_evaluator_matches_full_forward_for_every_tensor(self):
        case = lr.generate(4, 2)
        x, _, _ = lr.split(case["practice"])
        evaluator = lr.Evaluator(case["arrays"], x)
        for name, index in lr.entries(case["arrays"]):
            values = [-2.1, 0.0, 1.7]
            fast = evaluator.predict(name, index, values)
            for row, value in zip(fast, values):
                arrays = {n: v.copy() for n, v in case["arrays"].items()}
                arrays[name].flat[index] = np.float32(value)
                np.testing.assert_array_equal(row, lr.tasks._predict(arrays, lr.ARCHITECTURE, x))

    def test_teacher_value_meets_the_success_rule(self):
        for width in (2, 16):
            case = lr.generate(width, 0)
            fx, fy, repair = lr.split(case["heldout"])
            name, index = case["corrupted"]
            self.assertTrue(lr.success(lr.Evaluator(case["arrays"], fx), fy, repair, name, index,
                                       float(case["teacher"][name].flat[index])))

    def test_analysis_row_and_summary(self):
        lr.RANDOM_REPLICATES = 5
        rows = [lr.analyse(lr.generate(2, s), np.random.default_rng(0)) for s in range(2)]
        self.assertTrue(all(r["parameters"] == 12 for r in rows))
        summary = lr.summarize(rows)["2"]
        self.assertEqual(summary["cases"], 2)
        self.assertTrue(0 <= summary["given"] <= 1)


if __name__ == "__main__":
    unittest.main()
