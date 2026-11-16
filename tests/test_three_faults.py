import importlib.util
from itertools import combinations
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("three_faults", ROOT / "experiments/locate-repair/three_faults.py")
t3 = importlib.util.module_from_spec(spec)
sys.modules["three_faults"] = t3
spec.loader.exec_module(t3)


class ThreeFaultTests(unittest.TestCase):
    def test_three_faults_sit_in_different_units(self):
        case = t3.corrupt(4, 1)
        self.assertEqual(len(case["faults"]), 3)
        for a, b in combinations(case["faults"], 2):
            self.assertNotIn(t3.sub.relation(tuple(a), tuple(b), 4), ("mirror", "same unit"))

    def test_triples_extend_each_pair_once(self):
        entries = [("w", 0), ("w", 1), ("w", 2), ("w", 3)]
        triples = t3.triples_from([(entries[0], entries[1]), (entries[1], entries[0])], entries, limit=2)
        self.assertEqual(len(triples), 2)
        self.assertEqual(len(set(triples)), 2)

    def test_told_fine_fit_restores_practice(self):
        case = t3.corrupt(4, 1)
        px, py, _ = t3.lr.split(case["practice"])
        (accuracy, _), edits = t3.fine_fit(case, px, py, [tuple(f) for f in case["faults"]])
        before = (t3.tf.predict(case["arrays"], px) == py).mean()
        self.assertGreater(accuracy, before)
        self.assertEqual(len(edits), 3)


if __name__ == "__main__":
    unittest.main()
