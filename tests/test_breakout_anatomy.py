import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("breakout_anatomy", ROOT / "experiments/breakout-anatomy/anatomy.py")
anatomy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(anatomy)

try:
    import minatar  # noqa: F401
    HAVE_MINATAR = True
except ImportError:
    HAVE_MINATAR = False


def random_network(seed):
    rng = np.random.default_rng(seed)
    shapes = [(600, 32), (32,), (32, 32), (32,), (32, 6), (6,)]
    return {n: (rng.standard_normal(s) * 0.3).astype(np.float32) for n, s in zip(anatomy.NAMES, shapes)}


class BallPhysicsTests(unittest.TestCase):
    def test_wall_and_ceiling_reflections(self):
        empty = np.zeros((10, 10), dtype=bool)
        # Down-right from (8, 5): the next move reaches row 9 at column 9.
        self.assertEqual(anatomy.ball_future(8, 8, 2, empty, False), {"steps": 1, "catch": [8, 9], "under": 8, "corner": 9})
        # Down-right from the right wall reflects to column 9 again (clamped).
        self.assertEqual(anatomy.ball_future(9, 8, 2, empty, False), {"steps": 1, "catch": [9], "under": 9, "corner": 9})
        # Straight up-left from the corner bounces off the ceiling and comes back.
        result = anatomy.ball_future(0, 1, 0, empty, False)
        self.assertGreater(result["steps"], 8)

    @unittest.skipUnless(HAVE_MINATAR, "minatar not installed")
    def test_replica_matches_environment(self):
        from minatar import Environment
        checked = 0
        for episode in range(6):
            env = Environment("breakout", sticky_action_prob=0.1)
            env.seed(episode)
            env.reset()
            rng = np.random.default_rng(episode)
            for _ in range(400):
                predicted = anatomy.env_future(env.env)
                probe = copy.deepcopy(env.env)
                probe.pos = -5  # never under the ball, so the paddle-row move ends the game
                steps = 0
                while not probe.terminal:
                    probe.act(0)
                    steps += 1
                self.assertEqual(predicted["steps"], steps)
                self.assertEqual(predicted["catch"], sorted({probe.last_x, probe.ball_x}))
                checked += 1
                _, terminal = env.act(int(rng.choice([0, 1, 3])))
                if terminal:
                    break
        self.assertGreater(checked, 50)


@unittest.skipUnless(HAVE_MINATAR, "minatar not installed")
class EpisodeLoopTests(unittest.TestCase):
    def test_default_loop_reproduces_protocol(self):
        arrays = random_network(3)
        for index in range(4):
            ours, _ = anatomy.run_episode(index, 0, 500, arrays)
            theirs, _ = anatomy.breakout.run_episode((index, 0, 500), anatomy.as_list(arrays))
            self.assertEqual((ours["score"], ours["steps"], ours["truncated"]),
                             (theirs["score"], theirs["steps"], theirs["truncated"]))

    def test_sticky_zero_keeps_ball_starts(self):
        arrays = random_network(4)
        for index in range(3):
            a, rows_a = anatomy.run_episode(index, 0, 1, arrays, sticky=0.1, trace=True)
            b, rows_b = anatomy.run_episode(index, 0, 1, arrays, sticky=0.0, trace=True)
            self.assertEqual(rows_a[0]["ball"], rows_b[0]["ball"])
            self.assertEqual(rows_a[0]["dir"], rows_b[0]["dir"])

    def test_trace_and_summary(self):
        arrays = random_network(5)
        _, rows = anatomy.run_episode(0, 0, 60, arrays, trace=True)
        self.assertTrue(rows)
        summary = anatomy.summarize_semantics(rows)
        self.assertEqual(summary["states"], len(rows))
        self.assertIn("descending", summary["target_vs_truth"])


class VariantTests(unittest.TestCase):
    def test_silenced_layer2_unit_outputs_minus_one(self):
        arrays = random_network(6)
        arrays["actor/dense2/kernel"][:, 8:] = 0
        arrays["actor/dense2/bias"][8:] = 0
        made = anatomy.variants(arrays, noise_draws=1)
        silenced = made["silence-layer2-unit-2"][0]
        _, _, h2, _ = anatomy.forward(np.ones(600, dtype=np.float32), silenced)
        self.assertAlmostEqual(float(h2[2]), -1.0, places=6)
        self.assertEqual(made["sticky-0.0"][1], {"sticky": 0.0})

    def test_target_grid_rounds_only_target_units(self):
        arrays = random_network(7)
        a = anatomy.variants(arrays, noise_draws=1)["target-grid-0.1"][0]
        w1 = a["actor/dense1/kernel"]
        np.testing.assert_allclose(w1[:, :4], np.round(w1[:, :4] / 0.1) * 0.1, atol=1e-6)
        np.testing.assert_array_equal(w1[:, 4:], arrays["actor/dense1/kernel"][:, 4:])


if __name__ == "__main__":
    unittest.main()
