import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_returns", ROOT / "experiments/breakout-anatomy/returns.py")
returns = importlib.util.module_from_spec(spec)
sys.modules["breakout_returns"] = returns
spec.loader.exec_module(returns)


class ReturnsTests(unittest.TestCase):
    def test_catch_type_follows_minatar_order(self):
        self.assertEqual(returns.catch_type(4, 4), "under")
        self.assertEqual(returns.catch_type(4, 5), "corner")

    def test_under_from_corner_flips_only_the_horizontal_direction(self):
        # MinAtar: 0 = up-left, 1 = up-right
        self.assertEqual(returns.UNDER_FROM_CORNER, {0: 1, 1: 0})

    def test_since_corner_counts_returns_after_the_last_corner(self):
        records = [{"type": t} for t in ("under", "corner", "under", "under", "corner", "under")]
        self.assertEqual(returns.since_corner(records), [None, 0, 1, 2, 0, 1])

    def test_summary_bins_by_returns_since_corner(self):
        records = [{"type": "under", "bricks": 1, "top": 3, "died": False}, {"type": "corner", "bricks": 1, "top": 3, "died": False},
                   {"type": "under", "bricks": 0, "top": 0, "died": True}]
        summary = returns.summarize([{"score": 2, "steps": 100, "died": False, "records": records}])
        self.assertAlmostEqual(summary["no_brick_share"], 1 / 3)
        self.assertEqual(summary["by_returns_since_corner"]["before first"]["returns"], 1)
        self.assertEqual(summary["by_returns_since_corner"]["1-5"]["bricks_per_return"], 0.0)
        self.assertEqual(summary["by_turn_row"]["0"]["deaths_per_1000"], 1000.0)
        self.assertEqual(summary["deaths"], 0)


if __name__ == "__main__":
    unittest.main()
