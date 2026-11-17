import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("half_line", ROOT / "experiments/locate-repair/half_line.py")
hl = importlib.util.module_from_spec(spec)
sys.modules["half_line"] = hl
spec.loader.exec_module(hl)


class HalfLineTests(unittest.TestCase):
    def setUp(self):
        self.case = hl.rs.resize(hl.t3.corrupt(8, 0))
        px, py, _ = hl.lr.split(self.case["practice"])
        self.triple = tuple(tuple(f[:2]) for f in self.case["faults"])
        self.fit = hl.st.told_start(self.case, px, py, self.triple)
        self.pool = np.random.default_rng(0).uniform(-2, 2, size=(2000, 2))

    def test_below_the_line_half_the_labels_are_value_queries(self):
        picked, info = hl.choose(self.case, {self.triple: self.fit}, self.fit[1], self.pool, np.random.default_rng(1))
        self.assertEqual(len(set(picked)), hl.aq.PER_ROUND)
        self.assertEqual(info, {"value_queries": hl.aq.PER_ROUND // 2, "splitting": 0})

    def test_above_the_line_it_matches_the_switch_line_rule(self):
        (accuracy, cost), edits = self.fit
        shifted = [(n, i, v + 0.5 * self.case["scale"][n]) for n, i, v in edits]
        fits = {self.triple: self.fit, "shifted": ((accuracy, cost), shifted)}
        picked, info = hl.choose(self.case, fits, edits, self.pool, np.random.default_rng(1), line=0)
        reference = hl.sl.choose(self.case, fits, edits, self.pool, np.random.default_rng(1), line=0)
        self.assertEqual(info["value_queries"], 0)
        self.assertEqual((picked, info), reference)


if __name__ == "__main__":
    unittest.main()
