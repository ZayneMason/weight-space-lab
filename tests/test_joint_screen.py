import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("joint_screen", ROOT / "experiments/locate-repair/joint_screen.py")
js = importlib.util.module_from_spec(spec)
sys.modules["joint_screen"] = js
spec.loader.exec_module(js)


class JointScreenTests(unittest.TestCase):
    def test_joint_order_ranks_by_accuracy_and_keeps_coverage_order_for_ties(self):
        self.assertEqual(js.joint_order([0.9, 1.0, 0.95, 1.0]).tolist(), [1, 3, 2, 0])

    def test_first_true_is_the_rank_of_the_first_matching_pair(self):
        self.assertEqual(js.first_true(np.array([4, 2, 7, 1]), [7, 1]), 2)

    def test_joint_score_is_the_coarse_joint_fit(self):
        case = js.tf.corrupt(4, 1)
        px, py, _ = js.lr.split(case["practice"])
        pair = js.sc.ranked_pairs(case["arrays"], px, py)[0]
        score = js.joint_scores(case, px, py, [pair])[0]
        self.assertEqual(score, js.tf.joint(case, px, py, list(pair), points=js.POINTS)[1])
        self.assertGreaterEqual(score, (js.tf.predict(case["arrays"], px) == py).mean())


if __name__ == "__main__":
    unittest.main()
