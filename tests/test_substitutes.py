import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("substitutes", ROOT / "experiments/locate-repair/substitutes.py")
subs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subs)


class SubstituteTests(unittest.TestCase):
    def test_relations_follow_the_tensor_layout(self):
        width = 4  # hidden_weight (4, 2), hidden_bias (4,), weight (2, 4), bias (2,)
        fault = ("hidden_weight", 5)  # unit 2, second input
        self.assertEqual(subs.relation(("hidden_weight", 4), fault, width), "same unit")
        self.assertEqual(subs.relation(("hidden_bias", 2), fault, width), "same unit")
        self.assertEqual(subs.relation(("weight", 6), fault, width), "same unit")  # output 1, unit 2
        self.assertEqual(subs.relation(("weight", 1), fault, width), "elsewhere")
        output_fault = ("bias", 1)
        self.assertEqual(subs.relation(("weight", 5), output_fault, width), "same output")
        self.assertEqual(subs.relation(("weight", 1), output_fault, width), "elsewhere")

    def test_mirror_is_the_other_logits_weight_from_the_same_unit(self):
        width = 4
        self.assertEqual(subs.relation(("weight", 6), ("weight", 2), width), "mirror")  # unit 2, outputs 1 and 0
        self.assertEqual(subs.relation(("bias", 0), ("bias", 1), width), "mirror")
        self.assertEqual(subs.relation(("weight", 7), ("weight", 2), width), "elsewhere")
        self.assertEqual(subs.relation(("weight", 6), ("hidden_bias", 2), width), "same unit")

    def test_mirror_edit_restores_the_decision_exactly(self):
        import numpy as np
        rng = np.random.default_rng(0)
        width = 8
        teacher = {"hidden_weight": rng.standard_normal((width, 2)), "hidden_bias": rng.uniform(-.5, .5, width),
                   "weight": rng.standard_normal((2, width)), "bias": np.zeros(2)}
        broken = {k: v.copy() for k, v in teacher.items()}
        broken["weight"][0, 3] += 1.7
        broken["weight"][1, 3] += 1.7  # the mirror edit: same shift on the other logit
        x = rng.uniform(-2, 2, size=(5000, 2))

        def margin(arrays):
            h = np.maximum(x @ arrays["hidden_weight"].T + arrays["hidden_bias"], 0)
            logits = h @ arrays["weight"].T + arrays["bias"]
            return logits[:, 1] - logits[:, 0]
        np.testing.assert_allclose(margin(broken), margin(teacher), atol=1e-12)

    def test_unit_cosine_is_none_within_a_unit(self):
        import numpy as np
        teacher = {"hidden_weight": np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])}
        self.assertIsNone(subs.unit_cosine(teacher, ("hidden_bias", 0), ("weight", 0), 3))
        self.assertAlmostEqual(subs.unit_cosine(teacher, ("hidden_bias", 1), ("hidden_weight", 0), 3), 2 ** -0.5)
        self.assertEqual(len(subs.chance_cosines(teacher, ("hidden_weight", 4), 3)), 2)

    def test_analysis_row_counts_every_other_entry(self):
        case = subs.lr.generate(4, 0)
        row = subs.analyse(case)
        if row is None:
            self.skipTest("case cannot supply 1,024 practice examples")
        self.assertEqual(sum(row["chance_counts"].values()), 5 * 4 + 2 - 1)


if __name__ == "__main__":
    unittest.main()
