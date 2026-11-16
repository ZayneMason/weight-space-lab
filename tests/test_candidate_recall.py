import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candidate_recall", ROOT / "experiments/locate-repair/candidate_recall.py")
cr = importlib.util.module_from_spec(spec)
sys.modules["candidate_recall"] = cr
spec.loader.exec_module(cr)


class CandidateRecallTests(unittest.TestCase):
    def test_min_change_orders_by_accuracy_then_cost(self):
        order = cr.order([1.0, 1.0, 0.99], [3.0, 1.0, 0.1], [0, 1, 2], "min-change")
        self.assertEqual(order.tolist(), [1, 0, 2])

    def test_tolerance_lets_a_near_miss_tie_and_screen_rank_break_it(self):
        acc, cost = [0.998, 1.0, 1.0], [3.2, 0.5, 0.7]
        self.assertEqual(cr.order(acc, cost, [0, 1, 2], "screen").tolist(), [1, 2, 0])
        self.assertEqual(cr.order(acc, cost, [0, 1, 2], "screen", tolerance=0.005).tolist(), [0, 1, 2])
        self.assertEqual(cr.order(acc, cost, [0, 1, 2], "min-change", tolerance=0.005).tolist(), [1, 2, 0])

    def test_recall_checks_the_kept_prefix(self):
        row = {"accuracy": [1.0, 0.99, 1.0], "cost": [2.0, 0.1, 1.0], "true_indices": [0]}
        self.assertFalse(cr.recall(row, 3, 1, "min-change", 0.0))
        self.assertTrue(cr.recall(row, 3, 1, "screen", 0.0))
        self.assertTrue(cr.recall(row, 3, 2, "min-change", 0.0))

    def test_a_fault_that_changes_no_practice_label_is_silent(self):
        case = cr.tf.corrupt(32, 14)  # measured: the hidden-weight fault alone flips no practice label
        self.assertEqual(cr.silent_faults(case), [1])

    def test_found_visible_ignores_silent_faults_and_counts_mirrors(self):
        faults = [["weight", 27], ["hidden_weight", 62]]
        self.assertTrue(cr.found_visible([["weight", 59], ["bias", 0]], faults, 32, silent=[1]))
        self.assertFalse(cr.found_visible([["weight", 59], ["bias", 0]], faults, 32, silent=[]))
        self.assertFalse(cr.found_visible([["weight", 26], ["bias", 0]], faults, 32, silent=[1]))


if __name__ == "__main__":
    unittest.main()
