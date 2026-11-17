import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("triple_values", ROOT / "experiments/locate-repair/triple_values.py")
tv = importlib.util.module_from_spec(spec)
sys.modules["triple_values"] = tv
spec.loader.exec_module(tv)


class TripleValueTests(unittest.TestCase):
    def test_half_the_round_is_value_queries_around_a_triple(self):
        case = None
        for seed in range(10):
            base = tv.t3.corrupt(8, seed)
            if base is not None:
                case = tv.rs.resize(base)
                break
        self.assertIsNotNone(case)
        edits = [(n, i, float(case["teacher"][n].flat[i])) for n, i, *_ in case["faults"]]
        self.assertEqual(len(edits), 3)
        triple = tuple((n, i) for n, i, _ in edits)
        other = [(edits[0][0], edits[0][1], edits[0][2] + 2 * case["scale"][edits[0][0]])] + edits[1:]
        fits = {triple: ((1.0, 0.0), edits), ("other",): ((1.0, 0.0), other)}
        pool = np.random.default_rng(0).uniform(-2, 2, size=(3000, 2))
        picked, value = tv.choose(case, fits, edits, pool, np.random.default_rng(1))
        self.assertEqual(len(picked), tv.aq.PER_ROUND)
        self.assertEqual(value, tv.aq.PER_ROUND // 2)
        self.assertEqual(len(np.unique(picked, axis=0)), tv.aq.PER_ROUND)
        self.assertEqual(len(tv.vq.neighbours(case, edits)), 6)


if __name__ == "__main__":
    unittest.main()
