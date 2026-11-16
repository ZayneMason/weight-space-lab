import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pair_search", ROOT / "experiments/locate-repair/pair_search.py")
ps = importlib.util.module_from_spec(spec)
sys.modules["pair_search"] = ps
spec.loader.exec_module(ps)


class PairSearchTests(unittest.TestCase):
    def test_found_counts_each_fault_once_including_mirrors(self):
        width, faults = 4, [["weight", 1], ["hidden_bias", 3]]
        self.assertEqual(ps.found([("weight", 5), ("hidden_bias", 3)], faults, width), 2)  # mirror + exact
        self.assertEqual(ps.found([("weight", 1), ("weight", 5)], faults, width), 1)  # both stand for one fault
        self.assertEqual(ps.found([("hidden_bias", 0), ("bias", 0)], faults, width), 0)

    def test_search_on_the_true_pair_repairs_a_small_case(self):
        case = ps.tf.corrupt(4, 1)
        px, py, _ = ps.lr.split(case["practice"])
        true = [tuple(f) for f in case["faults"]]
        winner, fits, fine = ps.search(case, px, py, true, refine=1)
        self.assertEqual(len(fits), 1)
        self.assertEqual(set(winner), set(true))
        self.assertGreater(fine[winner][0][0], 0.95)


if __name__ == "__main__":
    unittest.main()
