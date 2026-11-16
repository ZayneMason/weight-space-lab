import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/locate-repair/dense_screen.py"
spec = importlib.util.spec_from_file_location("dense_screen", SCRIPT)
ds = importlib.util.module_from_spec(spec)
sys.modules["dense_screen"] = ds
spec.loader.exec_module(ds)


def worst_gap(points, low=ds.rs.LOW, high=ds.rs.HIGH):
    grid = np.linspace(-2.5, 2.5, points)
    sizes = np.linspace(low, high, 2001)
    return float(np.abs(sizes[:, None] - grid[None, :]).min(axis=1).max())


class DenseScreenTests(unittest.TestCase):
    def test_denser_grids_leave_every_size_nearer_a_point(self):
        self.assertGreater(worst_gap(7), 0.4)
        self.assertLess(worst_gap(ds.DENSE), 0.26)
        self.assertGreater(worst_gap(13), 0.2)
        self.assertLess(worst_gap(ds.PAIR_POINTS), 0.11)

    def test_merge_sorts_and_summarizes(self):
        rows = [{"seed": 3, "true_in_triples": True, "positions": {"coarse7": 50, "coarse11": 2}},
                {"seed": 1, "true_in_triples": False, "positions": {"coarse7": None, "coarse11": None}}]
        with tempfile.TemporaryDirectory() as tmp:
            part = Path(tmp) / "part.json"
            part.write_text(json.dumps({"cases": rows}))
            subprocess.run([sys.executable, str(SCRIPT), "--merge", str(part), "--output", tmp], check=True,
                           capture_output=True)
            report = json.loads((Path(tmp) / "dense-screen-triples.json").read_text())
        self.assertEqual([r["seed"] for r in report["cases"]], [1, 3])
        self.assertEqual(report["summary"]["coarse11"]["top40"], 1)
        self.assertEqual(report["summary"]["coarse7"]["top40"], 0)


if __name__ == "__main__":
    unittest.main()
