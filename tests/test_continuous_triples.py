import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("continuous_triples", ROOT / "experiments/locate-repair/continuous_triples.py")
ct = importlib.util.module_from_spec(spec)
sys.modules["continuous_triples"] = ct
spec.loader.exec_module(ct)


class ContinuousTriplesTests(unittest.TestCase):
    def test_best_of_prefers_accuracy_then_smaller_change(self):
        a, b, c = ((0.99, -1.0), "a"), ((1.0, -3.0), "b"), ((1.0, -2.0), "c")
        self.assertEqual(ct.best_of(a, b)[1], "b")
        self.assertEqual(ct.best_of(b, c)[1], "c")

    def test_fit_from_current_values_reaches_the_told_triple(self):
        case = ct.rs.resize(ct.t3.corrupt(8, 0))
        self.assertIsNotNone(case)
        px, py, _ = ct.lr.split(case["practice"])
        triple = tuple(tuple(f[:2]) for f in case["faults"])
        here = ct.wt.score(case, px, py, triple, ct.current_values(case, triple))
        fit = ct.wt.refine(case, px, py, triple, here)
        self.assertGreaterEqual(fit[0][0], here[0][0])


if __name__ == "__main__":
    unittest.main()
