import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("switch_triples", ROOT / "experiments/locate-repair/switch_triples.py")
st = importlib.util.module_from_spec(spec)
sys.modules["switch_triples"] = st
spec.loader.exec_module(st)


class SwitchTripleTests(unittest.TestCase):
    def test_told_start_edits_the_true_triple_and_fits_at_least_as_well_as_the_corrupted_values(self):
        case = st.rs.resize(st.t3.corrupt(8, 0))
        px, py, _ = st.lr.split(case["practice"])
        triple = tuple(tuple(f[:2]) for f in case["faults"])
        (accuracy, _), edits = st.told_start(case, px, py, triple)
        self.assertEqual([tuple(e[:2]) for e in edits], list(triple))
        corrupted = (st.tf.predict(case["arrays"], px) == py).mean()
        self.assertGreaterEqual(accuracy, corrupted)

    def test_switch_rule_on_triples_returns_one_round_of_distinct_inputs(self):
        case = st.rs.resize(st.t3.corrupt(8, 0))
        px, py, _ = st.lr.split(case["practice"])
        triple = tuple(tuple(f[:2]) for f in case["faults"])
        fit = st.told_start(case, px, py, triple)
        pool = np.random.default_rng(0).uniform(-2, 2, size=(2000, 2))
        picked, value = st.ss.choose(case, {triple: fit}, fit[1], pool, np.random.default_rng(1), "switch")
        self.assertEqual(len(picked), st.aq.PER_ROUND)
        self.assertEqual(len(set(picked)), len(picked))
        self.assertEqual(value, st.aq.PER_ROUND)  # a committee of one never splits, so every label is a value query


if __name__ == "__main__":
    unittest.main()
