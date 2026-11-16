import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
targets_spec = importlib.util.spec_from_file_location("breakout_targets", ROOT / "experiments/breakout-anatomy/targets.py")
targets = importlib.util.module_from_spec(targets_spec)
targets_spec.loader.exec_module(targets)
anatomy = targets.anatomy
HAVE_MINATAR = importlib.util.find_spec("minatar") is not None


def random_network(seed):
    rng = np.random.default_rng(seed)
    shapes = [(600, 32), (32,), (32, 32), (32,), (32, 6), (6,)]
    return {n: (rng.standard_normal(s) * 0.3).astype(np.float32) for n, s in zip(anatomy.NAMES, shapes)}


class TargetTableTests(unittest.TestCase):
    def test_tables_write_only_ascending_ball_weights(self):
        arrays = random_network(8)
        edited = targets.set_ascending_targets(arrays, targets.table_follow_ball())
        changed = np.argwhere(edited[targets.W1] != arrays[targets.W1])
        self.assertTrue(len(changed))
        self.assertTrue(set(changed[:, 1]) <= {2, 3})
        self.assertTrue(all(row % 6 == anatomy.BALL for row in changed[:, 0]))
        w = edited[targets.W1]
        self.assertAlmostEqual(float(w[anatomy.feature(4, 7, anatomy.BALL), 2]) / anatomy.TARGET_SCALE, 7.0, places=5)

    def test_no_brick_table_is_reflection_geometry(self):
        table = targets.table_no_brick()
        # Up-right from the bottom-left corner region: bounces off the ceiling and the right wall.
        self.assertIn((8, 0, 1), table)
        self.assertNotIn((9, 0, 1), table)
        self.assertTrue(all(0 <= t <= 9 for t in table.values()))

    @unittest.skipUnless(HAVE_MINATAR, "minatar not installed")
    def test_oracle_never_misses_without_sticky_actions(self):
        arrays = random_network(9)
        for index in range(3):
            row = targets.run_episode(index, 0, 600, arrays, "oracle", sticky=0.0)
            self.assertTrue(row["truncated"])

    @unittest.skipUnless(HAVE_MINATAR, "minatar not installed")
    def test_catch_profile_counts_every_catch_event(self):
        arrays = random_network(10)
        profile = targets.catch_profile(arrays, "oracle", episodes=range(2))
        self.assertGreater(profile["catch_events"], 0)
        total = sum(profile[k]["events"] for k in ("settle_0", "settle_1", "settle_2", "settle_3+"))
        self.assertEqual(total, profile["catch_events"])


if __name__ == "__main__":
    unittest.main()
