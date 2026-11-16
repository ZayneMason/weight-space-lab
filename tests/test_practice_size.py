import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("practice_size", ROOT / "experiments/locate-repair/practice_size.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
lr = ps.lr


class PracticeSizeTests(unittest.TestCase):
    def test_practice_sets_grow_and_keep_the_strata(self):
        case = lr.generate(4, 3)
        for size in (16, 64):
            x, y, repair = lr.split(ps.practice_set(case, size))
            self.assertEqual(len(y), 2 * size)
            self.assertEqual(int(repair.sum()), size)

    def test_analysis_row(self):
        case = lr.generate(2, 0)
        row = ps.analyse(case, 16)
        self.assertEqual(row["per_group"], 16)
        self.assertGreaterEqual(row["candidates"], 1)
        # The true entry's teacher value is practice-perfect, so the given strategy has a candidate.
        self.assertTrue(row["given_practice_perfect"])
        summary = ps.summarize([row], [2])
        self.assertEqual(summary["2/16"]["cases"], 1)


if __name__ == "__main__":
    unittest.main()
