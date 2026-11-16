import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_hedge", ROOT / "experiments/breakout-anatomy/hedge.py")
hedge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hedge)
anatomy = hedge.anatomy


class HedgeTests(unittest.TestCase):
    def test_descent_splits_the_flight(self):
        empty = np.zeros((10, 10), dtype=bool)
        info = hedge.descent(4, 6, 1, empty, False)
        self.assertEqual(info["steps"], anatomy.ball_future(4, 6, 1, empty, False)["steps"])
        self.assertEqual(info["descending_steps"], 9)  # from the ceiling (row 0) down to row 9
        already_down = hedge.descent(3, 5, 2, empty, False)
        self.assertEqual(already_down["steps"], already_down["descending_steps"])

    def test_full_wall_bounces_at_row_three(self):
        catches = hedge.full_wall_catches()
        # Up-right from (row 5, col 0): hits the brick at row 3 after two moves and comes back down.
        self.assertEqual(catches[(5, 0, 1)], [6, 7])
        self.assertNotIn((9, 0, 1), catches)

    def test_deadline_fit_prefers_reachable_targets(self):
        # Two observed outcomes: an early landing at column 0 with little time, a late one at 9 with lots.
        seen = {(5, 4, 1): [([0], 3), ([9], 12)]}
        table = hedge.fit_deadline_aware(seen)
        self.assertLessEqual(table[(5, 4, 1)], 1.0)

    def test_most_likely_prefers_the_common_landing(self):
        # Landings at 2, 3 and 9: mean distance is smallest at 3 (the median), but 2.5 is
        # within half a column of two of the three.
        seen = {(5, 4, 1): [([2], 6), ([3], 6), ([9], 6)]}
        self.assertEqual(hedge.fit_most_likely(seen)[(5, 4, 1)], 2.5)
        fit = hedge.observed_fit({(5, 4, 1): 2.5}, seen)
        self.assertAlmostEqual(fit["exact_hit"], 2 / 3)
        self.assertAlmostEqual(fit["late_by_margin"]["1"], 1 / 3)  # 6.5 columns away with 6 steps left

    def test_on_catch_columns(self):
        catches = {(1, 1, 1): [3, 4], (2, 2, 0): [5]}
        result = hedge.on_catch_columns({(1, 1, 1): 4.0, (2, 2, 0): 7.0}, catches, sorted(catches))
        self.assertEqual(result, {"cells": 2, "on_a_catch_column": 0.5})


if __name__ == "__main__":
    unittest.main()
