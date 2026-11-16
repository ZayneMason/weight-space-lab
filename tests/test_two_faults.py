import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("two_faults", ROOT / "experiments/locate-repair/two_faults.py")
tf = importlib.util.module_from_spec(spec)
sys.modules["two_faults"] = tf
spec.loader.exec_module(tf)


class TwoFaultTests(unittest.TestCase):
    def test_second_fault_is_in_another_unit_and_restoring_both_repairs(self):
        case = tf.corrupt(4, 0)
        first, second = (tuple(f) for f in case["faults"])
        self.assertNotIn(tf.sub.relation(second, first, 4), ("mirror", "same unit"))
        changed = [(n, i) for n in tf.lr.TENSORS for i in range(case["teacher"][n].size)
                   if case["arrays"][n].flat[i] != case["teacher"][n].flat[i]]
        self.assertEqual(sorted(changed), sorted([first, second]))
        fx, fy, frepair = tf.lr.split(case["heldout"])
        inputs = np.random.default_rng(0).uniform(-2, 2, size=(500, 2))
        restore = [(n, i, float(case["teacher"][n].flat[i])) for n, i in (first, second)]
        result = tf.outcome(case, restore, fx, fy, frepair, inputs, tf.predict(case["teacher"], inputs))
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["damage_undone"], 1.0)

    def test_matches_counts_the_exact_mirror(self):
        width = 4
        faults = [["weight", 1], ["hidden_bias", 3]]
        self.assertEqual(tf.matches(("weight", 1 + width), faults, width), [0])  # same unit, other logit
        self.assertEqual(tf.matches(("hidden_bias", 3), faults, width), [1])
        self.assertEqual(tf.matches(("hidden_bias", 2), faults, width), [])


if __name__ == "__main__":
    unittest.main()
