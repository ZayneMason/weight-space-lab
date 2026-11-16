import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pair_screen", ROOT / "experiments/locate-repair/pair_screen.py")
sc = importlib.util.module_from_spec(spec)
sys.modules["pair_screen"] = sc
spec.loader.exec_module(sc)


class PairScreenTests(unittest.TestCase):
    def setUp(self):
        self.case = sc.tf.corrupt(4, 1)
        self.x = np.random.default_rng(0).uniform(-2, 2, size=(40, 2))

    def test_margin_jacobian_matches_finite_differences(self):
        arrays = self.case["arrays"]
        jac, margin = sc.margin_jacobian(arrays, self.x)

        def m(a):
            a = {n: v.astype(np.float64) for n, v in a.items()}
            logits = np.maximum(self.x @ a["hidden_weight"].T + a["hidden_bias"], 0) @ a["weight"].T + a["bias"]
            return logits[:, 1] - logits[:, 0]

        np.testing.assert_allclose(margin, m(arrays), atol=1e-9)
        for k, (name, index) in enumerate(sc.lr.entries(arrays)):
            moved = {n: v.astype(np.float64).copy() for n, v in arrays.items()}
            moved[name].flat[index] += 1e-6
            np.testing.assert_allclose((m(moved) - m(arrays)) / 1e-6, jac[:, k], atol=1e-4)

    def test_a_mirror_pair_scores_like_one_entry_in_the_newton_screen(self):
        y = sc.tf.predict(self.case["teacher"], self.x)
        score, single = sc.pair_scores(self.case["arrays"], self.x, y)
        entries = sc.lr.entries(self.case["arrays"])
        b0, b1 = entries.index(("bias", 0)), entries.index(("bias", 1))
        self.assertAlmostEqual(score[b0, b1] / single[b0], 1.0, places=3)

    def test_coverage_of_the_true_pair_is_high_and_bounded(self):
        px, py, _ = sc.lr.split(self.case["practice"])
        score, cover = sc.coverage_scores(self.case["arrays"], px, py)
        self.assertTrue((cover <= 1 + 1e-9).all() and (cover >= 0).all())
        entries = sc.lr.entries(self.case["arrays"])
        f1, f2 = (entries.index(tuple(f)) for f in self.case["faults"])
        self.assertGreater(cover[f1, f2], 0.5)
        pairs = sc.order_pairs(self.case["arrays"], score)
        self.assertEqual(len(pairs), len(entries) * (len(entries) - 1) // 2)


if __name__ == "__main__":
    unittest.main()
