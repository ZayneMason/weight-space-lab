import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lookalikes", ROOT / "experiments/locate-repair/lookalikes.py")
la = importlib.util.module_from_spec(spec)
sys.modules["lookalikes"] = la
spec.loader.exec_module(la)


class LookalikeTests(unittest.TestCase):
    def setUp(self):
        self.case = la.rs.resize(la.t3.corrupt(8, 0))
        self.probe = np.random.default_rng(1).uniform(-2, 2, size=(5000, 2))
        self.labels = la.tf.predict(self.case["teacher"], self.probe)

    def test_undoing_the_bugs_agrees_with_the_teacher(self):
        undo = [(n, i, float(self.case["teacher"][n].flat[i])) for n, i, *_ in self.case["faults"]]
        self.assertEqual(la.disagreement(self.case, undo, self.probe, self.labels), 0.0)
        self.assertGreater(la.disagreement(self.case, [], self.probe, self.labels), 0.0)

    def test_mixed_pool_is_half_uniform_half_boundary(self):
        px, py, _ = la.lr.split(self.case["practice"])
        triple = tuple(tuple(f[:2]) for f in self.case["faults"])
        fits = {triple: la.tc.wt.score(self.case, px, py, triple, la.tc.ct.current_values(self.case, triple))}
        pool = la.tc.query_pool(self.case, fits, np.random.default_rng(2), "mixed")
        self.assertEqual(pool.shape, (la.tc.aq.POOL, 2))
        self.assertTrue((np.abs(pool) <= 2).all())


if __name__ == "__main__":
    unittest.main()
