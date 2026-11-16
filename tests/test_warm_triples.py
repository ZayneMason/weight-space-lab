import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("warm_triples", ROOT / "experiments/locate-repair/warm_triples.py")
wt = importlib.util.module_from_spec(spec)
sys.modules["warm_triples"] = wt
spec.loader.exec_module(wt)


class WarmTripleTests(unittest.TestCase):
    def setUp(self):
        self.case = wt.t3.corrupt(4, 1)
        self.px, self.py, _ = wt.lr.split(self.case["practice"])
        self.triple = tuple(tuple(f[:2]) for f in self.case["faults"])

    def test_teacher_values_have_positive_margins(self):
        values = [float(self.case["teacher"][n].flat[i]) for n, i in self.triple]
        (accuracy, _), _ = wt.score(self.case, self.px, self.py, self.triple, values)
        self.assertEqual(accuracy, 1.0)
        margins = wt.signed_margins(self.case, self.triple, values, self.px, self.py)
        self.assertGreaterEqual((margins >= 0).mean(), 0.999)

    def test_fit_is_never_worse_than_the_grid_fit(self):
        grid = wt.t3.fine_fit(self.case, self.px, self.py, self.triple)
        best = wt.fit(self.case, self.px, self.py, self.triple, previous=grid)
        self.assertGreaterEqual(best[0][0], grid[0][0])
        self.assertEqual(len(best[1]), 3)

    def test_refine_keeps_the_entries(self):
        grid = wt.t3.fine_fit(self.case, self.px, self.py, self.triple)
        refined = wt.refine(self.case, self.px, self.py, self.triple, grid)
        self.assertEqual([e[:2] for e in refined[1]], [e[:2] for e in grid[1]])
        self.assertTrue(np.isfinite(refined[0][1]))


if __name__ == "__main__":
    unittest.main()
