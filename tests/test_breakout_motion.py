import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_motion", ROOT / "experiments/breakout-anatomy/motion.py")
motion = importlib.util.module_from_spec(spec)
sys.modules["breakout_motion"] = motion
spec.loader.exec_module(motion)


class MotionTests(unittest.TestCase):
    def test_turn_motion_is_relative_to_the_nearest_catch(self):
        self.assertEqual(motion.turn_motion(3, 4, [6, 7]), "toward")
        self.assertEqual(motion.turn_motion(5, 4, [6, 7]), "away")
        self.assertEqual(motion.turn_motion(4, 4, [6, 7]), "still")
        self.assertEqual(motion.turn_motion(6, 7, [6, 7]), "still")  # between two catch columns

    def test_finish_marks_flights_turned_back_by_a_brick(self):
        steps = [{"pos": 5, "chosen_moves": False, "override_changed": False}] * 3
        turned = motion.finish({"row": 0, "steps_left": 8, "catch": [5], "steps": list(steps), "died": False})
        self.assertFalse(turned["reached"])
        reached = motion.finish({"row": 4, "steps_left": 3, "catch": [5], "steps": list(steps), "died": False})
        self.assertTrue(reached["reached"] and reached["settled"])
        self.assertFalse(reached["unsettled_final"])

    def test_summary_splits_by_motion(self):
        base = {"reached": True, "settled": True, "unsettled_final": False, "late_override": False, "steps_left": 5, "distance": 1}
        records = [dict(base, row=3, turn_motion="toward", died=False),
                   dict(base, row=3, turn_motion="away", died=True, settled=False),
                   dict(base, row=3, turn_motion="away", died=False)]
        summary = motion.summarize(records)["3"]
        self.assertEqual(summary["flights"], 3)
        self.assertEqual(summary["by_turn_motion"]["away"]["deaths_per_1000"], 500.0)
        self.assertEqual(summary["deaths_settled_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
