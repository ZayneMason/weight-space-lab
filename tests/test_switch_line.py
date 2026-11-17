import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("switch_line", ROOT / "experiments/locate-repair/switch_line.py")
sl = importlib.util.module_from_spec(spec)
sys.modules["switch_line"] = sl
spec.loader.exec_module(sl)


class SwitchLineTests(unittest.TestCase):
    def setUp(self):
        self.case = sl.rs.resize(sl.t3.corrupt(8, 0))
        px, py, _ = sl.lr.split(self.case["practice"])
        self.triple = tuple(tuple(f[:2]) for f in self.case["faults"])
        self.fit = sl.st.told_start(self.case, px, py, self.triple)
        self.pool = np.random.default_rng(0).uniform(-2, 2, size=(2000, 2))

    def test_committee_of_one_switches_to_value_queries(self):
        picked, info = sl.choose(self.case, {self.triple: self.fit}, self.fit[1], self.pool, np.random.default_rng(1))
        self.assertEqual(len(set(picked)), sl.aq.PER_ROUND)
        self.assertEqual(info, {"value_queries": sl.aq.PER_ROUND, "splitting": 0})

    def test_line_zero_never_switches_and_matches_the_switch_rule_while_splits_remain(self):
        (accuracy, cost), edits = self.fit
        shifted = [(n, i, v + 0.5 * self.case["scale"][n]) for n, i, v in edits]
        fits = {self.triple: self.fit, "shifted": ((accuracy, cost), shifted)}  # two committee members that disagree
        picked, info = sl.choose(self.case, fits, edits, self.pool, np.random.default_rng(1), line=0)
        reference, value = sl.ss.choose(self.case, fits, edits, self.pool, np.random.default_rng(1), "switch")
        self.assertGreater(info["splitting"], 0)
        self.assertEqual(info["value_queries"], 0)
        self.assertEqual((picked, 0), (reference, value))

if __name__ == "__main__":
    unittest.main()
