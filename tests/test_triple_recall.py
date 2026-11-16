import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("triple_recall", ROOT / "experiments/locate-repair/triple_recall.py")
tr = importlib.util.module_from_spec(spec)
sys.modules["triple_recall"] = tr
spec.loader.exec_module(tr)


class TripleRecallTests(unittest.TestCase):
    def test_order_breaks_ties_by_change_or_earlier_order(self):
        accuracy = np.array([0.9, 1.0, 1.0, 1.0])
        cost = np.array([0.1, 3.0, 1.0, 2.0])
        self.assertEqual(list(tr.order(accuracy, cost, None)), [2, 3, 1, 0])
        self.assertEqual(list(tr.order(accuracy, cost, np.array([3, 0, 2, 1]))), [1, 3, 2, 0])

    def test_position_finds_first_true(self):
        self.assertEqual(tr.position(np.array([2, 0, 1]), np.array([False, True, False])), 2)
        self.assertIsNone(tr.position(np.array([0, 1]), np.array([False, False])))

    def test_summary_counts_top_k(self):
        rows = [{"true_in_triples": True, "positions": {"a": 5}},
                {"true_in_triples": True, "positions": {"a": 150}},
                {"true_in_triples": False, "positions": {"a": None}}]
        out = tr.summarize(rows)
        self.assertEqual(out["a"], {"top40": 1, "top100": 1, "top200": 2, "median_position": 78.5})
        self.assertEqual(out["true_in_triples"], 2)


if __name__ == "__main__":
    unittest.main()
