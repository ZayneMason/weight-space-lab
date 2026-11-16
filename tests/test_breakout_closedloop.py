import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_closedloop", ROOT / "experiments/breakout-anatomy/closedloop.py")
closedloop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closedloop)


class ClosedLoopTests(unittest.TestCase):
    def test_refit_moves_visited_cells_and_keeps_the_rest(self):
        previous = {(5, 4, 1): 7.0, (6, 2, 0): 3.0}
        seen = {(5, 4, 1): [([2], 6), ([3], 6), ([9], 6)]}
        table = closedloop.fit_mean_distance(seen, previous)
        self.assertEqual(table[(5, 4, 1)], 3.0)  # the median landing
        self.assertEqual(table[(6, 2, 0)], 3.0)
        self.assertEqual(previous[(5, 4, 1)], 7.0)


if __name__ == "__main__":
    unittest.main()
