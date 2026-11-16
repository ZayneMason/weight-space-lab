import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_combined", ROOT / "experiments/breakout-anatomy/combine.py")
combine = importlib.util.module_from_spec(spec)
sys.modules["breakout_combined"] = combine
spec.loader.exec_module(combine)


class CombineTests(unittest.TestCase):
    def test_variants_swap_one_row_at_a_time(self):
        cells = [(r, c, d) for r in range(9) for c in range(2) for d in (0, 1)]
        author = {k: 1.0 for k in cells}
        under = {k: 2.0 for k in cells}
        out = combine.variants(author, under)
        self.assertEqual(out["author, row 4 from under"][(4, 0, 0)], 2.0)
        self.assertEqual(out["author, row 4 from under"][(3, 0, 0)], 1.0)
        self.assertEqual(out["under, row 4 from author"][(4, 1, 1)], 1.0)
        self.assertEqual(out["under, row 4 from author"][(5, 1, 1)], 2.0)
        self.assertEqual(out["under, rows 3-5 from author"][(5, 0, 1)], 1.0)
        self.assertEqual(out["author, rows 3,8 from under"][(8, 0, 1)], 2.0)

    def test_paired_difference(self):
        report = {"results": {"author": {"sticky_0.1": {"scores": [10, 20, 30]}},
                              "mix": {"sticky_0.1": {"scores": [12, 22, 32]}}}}
        out = combine.paired(report)["mix"]
        self.assertEqual(out["difference"], 2.0)
        self.assertEqual(out["se"], 0.0)


if __name__ == "__main__":
    unittest.main()
