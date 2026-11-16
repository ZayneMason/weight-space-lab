import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("active_screen", ROOT / "experiments/locate-repair/active_screen.py")
asc = importlib.util.module_from_spec(spec)
sys.modules["active_screen"] = asc
spec.loader.exec_module(asc)


class ActiveScreenTests(unittest.TestCase):
    def test_uses_a_larger_query_pool_than_roadmap_22(self):
        self.assertEqual(asc.aq.POOL, 20_000)
        self.assertEqual((asc.TOP, asc.CANDIDATES), (1000, 40))

    def test_candidates_keep_the_best_coarse_fits_from_the_screen(self):
        case = asc.tf.corrupt(4, 1)
        px, py, _ = asc.lr.split(case["practice"])
        screened = asc.sc.ranked_pairs(case["arrays"], px, py)[:6]
        pool, rank = asc.candidates(case, top=6, keep=3)
        self.assertEqual(len(pool), 3)
        self.assertTrue(set(pool) <= set(screened))
        fits = {pair: asc.ps.pair_fit(case, px, py, pair) for pair in screened}
        best = sorted(fits.values(), reverse=True)[:3]
        self.assertEqual(sorted((fits[p] for p in pool), reverse=True), best)
        self.assertEqual(rank, asc.sc.true_rank(asc.sc.ranked_pairs(case["arrays"], px, py), case["faults"], 4))


if __name__ == "__main__":
    unittest.main()
