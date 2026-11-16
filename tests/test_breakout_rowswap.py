import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_rowswap", ROOT / "experiments/breakout-anatomy/rowswap.py")
rowswap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rowswap)


class RowSwapTests(unittest.TestCase):
    def test_mix_takes_only_the_chosen_rows(self):
        base = {(3, 0, 0): 1.0, (4, 0, 0): 2.0, (9, 0, 0): 3.0}
        donor = {(3, 0, 0): 7.0, (4, 0, 0): 8.0}
        self.assertEqual(rowswap.mix(base, donor, {4, 9}), {(3, 0, 0): 1.0, (4, 0, 0): 8.0, (9, 0, 0): 3.0})

    def test_accurate_table_reads_the_committed_evidence(self):
        table = rowswap.accurate_table()
        self.assertEqual(len(table), 180)
        self.assertTrue(all(0 <= v <= 9 for v in table.values()))


if __name__ == "__main__":
    unittest.main()
