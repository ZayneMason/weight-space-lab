import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_underfit", ROOT / "experiments/breakout-anatomy/underfit.py")
underfit = importlib.util.module_from_spec(spec)
sys.modules["breakout_underfit"] = underfit
spec.loader.exec_module(underfit)


class UnderFitTests(unittest.TestCase):
    def test_target_columns(self):
        future = {"catch": [4, 5], "under": 5, "corner": 4}
        self.assertEqual(underfit.target_columns(future, "nearest"), [4, 5])
        self.assertEqual(underfit.target_columns(future, "under"), [5])
        self.assertEqual(underfit.target_columns(future, "corner"), [4])

    def test_fit_picks_the_chosen_column(self):
        cell = (3, 2, 1)
        observations = [(cell, [6, 7], 6, 7)] * 3
        self.assertEqual(underfit.fit(observations, "under")[cell], 6.0)
        self.assertEqual(underfit.fit(observations, "corner")[cell], 7.0)
        # the nearest catch column is indifferent between the two; the grid's first minimum wins
        self.assertIn(underfit.fit(observations, "nearest")[cell], (6.0, 6.5, 7.0))


if __name__ == "__main__":
    unittest.main()
