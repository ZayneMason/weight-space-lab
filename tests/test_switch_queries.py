import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("switch_queries", ROOT / "experiments/locate-repair/switch_queries.py")
sq = importlib.util.module_from_spec(spec)
sys.modules["switch_queries"] = sq
spec.loader.exec_module(sq)


def setup():
    case = sq.rs.resize(sq.tf.corrupt(8, 0))
    edits = [(n, i, float(case["teacher"][n].flat[i])) for n, i, *_ in case["faults"]]
    pair = tuple((n, i) for n, i, _ in edits)
    pool = np.random.default_rng(0).uniform(-2, 2, size=(3000, 2))
    return case, edits, pair, pool


class SwitchQueryTests(unittest.TestCase):
    def test_a_lone_committee_gets_only_value_queries(self):
        case, edits, pair, pool = setup()
        fits = {pair: ((1.0, 0.0), edits)}
        for rule in sq.ARMS:
            chosen, value = sq.choose(case, fits, edits, pool, np.random.default_rng(1), rule)
            self.assertEqual(len(chosen), sq.aq.PER_ROUND)
            self.assertEqual(len(set(chosen)), sq.aq.PER_ROUND)
            self.assertEqual(value, sq.aq.PER_ROUND)

    def test_a_split_committee_gets_targeted_labels_first(self):
        case, edits, pair, pool = setup()
        other = [(edits[0][0], edits[0][1], edits[0][2] + 2 * case["scale"][edits[0][0]])] + edits[1:]
        fits = {pair: ((1.0, 0.0), edits), ("other",): ((1.0, 0.0), other)}
        split = sq.committee_split(case, fits, pool)
        self.assertGreater((split > 0).sum(), 0)
        chosen, value = sq.choose(case, fits, edits, pool, np.random.default_rng(1), "switch")
        self.assertEqual(value, 0)
        self.assertTrue((split[chosen[:min(8, int((split > 0).sum()))]] > 0).all())
        chosen, value = sq.choose(case, fits, edits, pool, np.random.default_rng(1), "fill")
        self.assertEqual(len(set(chosen)), sq.aq.PER_ROUND)
        self.assertEqual(value + min(8, int((split > 0).sum())), sq.aq.PER_ROUND)


if __name__ == "__main__":
    unittest.main()
