import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("active_queries", ROOT / "experiments/locate-repair/active_queries.py")
aq = importlib.util.module_from_spec(spec)
sys.modules["active_queries"] = aq
spec.loader.exec_module(aq)


class ActiveQueryTests(unittest.TestCase):
    def test_committee_split_is_the_minority_share(self):
        votes = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]])  # 4 candidates, 3 inputs
        np.testing.assert_allclose(aq.committee_split(votes), [0.0, 0.25, 0.5])

    def test_targeted_queries_come_from_where_near_best_candidates_disagree(self):
        case = aq.tf.corrupt(4, 1)
        name, index = "bias", 1
        current = float(case["arrays"][name].flat[index])
        same = [(name, index, current)]
        moved = [(name, index, current + 0.5)]
        fits = {"a": ((1.0, 0.0), same), "b": ((0.999, 0.0), moved), "c": ((0.5, 0.0), moved)}
        pool = np.random.default_rng(0).uniform(-2, 2, size=(2000, 2))
        disagree = (aq.tf.predict(aq.tf.apply(case["arrays"], same), pool)
                    != aq.tf.predict(aq.tf.apply(case["arrays"], moved), pool)).sum()
        self.assertGreaterEqual(disagree, aq.PER_ROUND)
        picked, split = aq.choose_queries(case, fits, pool, np.random.default_rng(1), "targeted")
        self.assertEqual(len(picked), aq.PER_ROUND)
        a = aq.tf.predict(aq.tf.apply(case["arrays"], same), picked)
        b = aq.tf.predict(aq.tf.apply(case["arrays"], moved), picked)
        self.assertTrue((a != b).all())  # candidate c is outside the margin and does not vote
        self.assertEqual(split, 0.5)


if __name__ == "__main__":
    unittest.main()
