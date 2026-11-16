import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_deaths", ROOT / "experiments/breakout-anatomy/deaths.py")
deaths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deaths)


class DeathsTests(unittest.TestCase):
    def test_summary_rates_and_slack(self):
        records = [{"row": 4, "steps_left": 5, "distance": 1, "died": False},
                   {"row": 4, "steps_left": 4, "distance": 4, "died": True},
                   {"row": 0, "steps_left": 9, "distance": 2, "died": False}]
        summary = deaths.summarize(records)
        self.assertEqual(summary["4"]["flights"], 2)
        self.assertEqual(summary["4"]["deaths_per_1000"], 500.0)
        self.assertEqual(summary["4"]["mean_slack"], 2.0)
        self.assertEqual(summary["4"]["no_slack_share"], 0.5)
        self.assertEqual(summary["0"]["deaths"], 0)


if __name__ == "__main__":
    unittest.main()
