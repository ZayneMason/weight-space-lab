import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("value_queries", ROOT / "experiments/locate-repair/value_queries.py")
vq = importlib.util.module_from_spec(spec)
sys.modules["value_queries"] = vq
spec.loader.exec_module(vq)


class ValueQueryTests(unittest.TestCase):
    def test_middle_of_tie_picks_the_tied_point_nearest_the_centre(self):
        accuracy = np.zeros((5, 5))
        accuracy[1:4, 0:3] = 1.0
        self.assertEqual(tuple(int(v) for v in vq.middle_of_tie(accuracy)), (2, 1))

    def test_neighbours_move_one_value_each_way(self):
        case = vq.rs.resize(vq.tf.corrupt(8, 0))
        edits = [(n, i, float(case["arrays"][n].flat[i])) for n, i, *_ in case["faults"]]
        moved = vq.neighbours(case, edits)
        self.assertEqual(len(moved), 2 * len(edits))
        for k, n in enumerate(moved):
            changed = [a[2] != b[2] for a, b in zip(n, edits)]
            self.assertEqual(sum(changed), 1)
            self.assertTrue(changed[k // 2])

    def test_value_queries_are_inputs_where_a_nudged_copy_disagrees(self):
        case = vq.rs.resize(vq.tf.corrupt(8, 0))
        edits = [(n, i, float(case["teacher"][n].flat[i])) for n, i, *_ in case["faults"]]
        pool = np.random.default_rng(0).uniform(-2, 2, size=(5000, 2))
        picked = vq.value_queries(case, edits, pool, np.random.default_rng(1), 4)
        self.assertEqual(len(picked), 4)
        self.assertEqual(len(set(picked)), 4)
        winner = vq.tf.predict(vq.tf.apply(case["arrays"], edits), pool[picked])
        flips = [vq.tf.predict(vq.tf.apply(case["arrays"], n), pool[picked]) != winner for n in vq.neighbours(case, edits)]
        self.assertTrue(np.any(flips, axis=0).all())

    def test_refined_fit_fits_at_least_as_well_as_the_grid(self):
        case = vq.rs.resize(vq.tf.corrupt(8, 0))
        px, py, _ = vq.lr.split(case["practice"])
        pair = tuple(tuple(f) for f in case["faults"])
        if len(pair) != 2:
            self.skipTest("needs a two-fault case")
        grid, _ = vq.fit(case, px, py, pair, "edge")
        refined, edits = vq.fit(case, px, py, pair, "refine")
        self.assertGreaterEqual(refined[0], grid[0])
        self.assertEqual([e[:2] for e in edits], [tuple(p) for p in pair])


if __name__ == "__main__":
    unittest.main()
