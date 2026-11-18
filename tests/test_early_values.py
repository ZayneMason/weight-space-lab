import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("early_values", ROOT / "experiments/locate-repair/early_values.py")
ev = importlib.util.module_from_spec(spec)
sys.modules["early_values"] = ev
spec.loader.exec_module(ev)


class EarlyValuesTests(unittest.TestCase):
    def setUp(self):
        self.case = ev.rs.resize(ev.t3.corrupt(8, 0))
        px, py, _ = ev.lr.split(self.case["practice"])
        self.triple = tuple(tuple(f[:2]) for f in self.case["faults"])
        self.fit = ev.st.told_start(self.case, px, py, self.triple)
        (accuracy, cost), edits = self.fit
        shifted = [(n, i, v + 0.5 * self.case["scale"][n]) for n, i, v in edits]
        self.fits = {self.triple: self.fit, "shifted": ((accuracy, cost), shifted)}
        self.pool = np.random.default_rng(0).uniform(-2, 2, size=(2000, 2))

    def test_mixes_for_the_early_rounds_then_only_splits(self):
        choose = ev.make_chooser(early=2)
        counts = [choose(self.case, self.fits, self.fit[1], self.pool, np.random.default_rng(r))[1]["value_queries"]
                  for r in range(4)]
        self.assertEqual(counts, [ev.aq.PER_ROUND // 2] * 2 + [0, 0])

    def test_switch_point_is_set_by_early(self):
        for early in (0, 3):
            choose = ev.make_chooser(early=early)
            counts = [choose(self.case, self.fits, self.fit[1], self.pool, np.random.default_rng(r))[1]["value_queries"]
                      for r in range(5)]
            self.assertEqual(counts, [ev.aq.PER_ROUND // 2] * early + [0] * (5 - early))

    def test_each_case_gets_its_own_round_count(self):
        first, second = ev.make_chooser(early=1), ev.make_chooser(early=1)
        first(self.case, self.fits, self.fit[1], self.pool, np.random.default_rng(0))
        _, info = second(self.case, self.fits, self.fit[1], self.pool, np.random.default_rng(0))
        self.assertEqual(info["value_queries"], ev.aq.PER_ROUND // 2)


    def test_fresh_switch_evidence_matches_its_arms(self):
        evidence = ROOT / "experiments/locate-repair/evidence/fresh-switch"
        import json
        for name, early, start in (("early-1", 1, 0), ("fresh-early-0", 0, 25), ("fresh-early-2", 2, 25),
                                   ("fresh-early-4", 4, 25)):
            report = json.loads((evidence / f"{name}.json").read_text())
            self.assertEqual(report["early_rounds"], early)
            self.assertEqual(sorted(c["seed"] for c in report["cases"]), list(range(start, start + 25)))
            for case in report["cases"]:
                asked = [h.get("value_queries", 0) for h in case["switch"][:-1]]
                self.assertEqual(sum(1 for a in asked if a), early)


if __name__ == "__main__":
    unittest.main()
