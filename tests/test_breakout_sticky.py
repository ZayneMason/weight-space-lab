import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_sticky", ROOT / "experiments/breakout-anatomy/sticky_sweep.py")
sweep = importlib.util.module_from_spec(spec)
sys.modules["breakout_sticky"] = sweep
spec.loader.exec_module(sweep)


class StickySweepTests(unittest.TestCase):
    def test_hazard_counts_only_episodes_that_ended_in_a_miss(self):
        rows = [{"truncated": True, "steps": 10000}, {"truncated": False, "steps": 4000},
                {"truncated": False, "steps": 6000}]
        out = sweep.hazard(rows)
        self.assertEqual(out["deaths"], 2)
        self.assertAlmostEqual(out["per_10k_steps"], 1.0)
        self.assertAlmostEqual(out["per_10k_steps_se"], 2 ** 0.5 / 2)


if __name__ == "__main__":
    unittest.main()
