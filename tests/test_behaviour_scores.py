import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("behaviour_scores", ROOT / "experiments/locate-repair/behaviour_scores.py")
bs = importlib.util.module_from_spec(spec)
sys.modules["behaviour_scores"] = bs
spec.loader.exec_module(bs)


def step(found, visible, ok, disagreement):
    return {"queries": 0, "faults_found": found, "found_visible": visible, "success": ok, "disagreement": disagreement}


class BehaviourScoreTests(unittest.TestCase):
    def test_probe_scores_undoing_the_bugs_as_zero(self):
        case = bs.rs.resize(bs.tf.corrupt(8, 0))
        probe = bs.make_probe(case, 5000)
        undo = [(n, i, float(case["teacher"][n].flat[i])) for n, i, *_ in case["faults"]]
        self.assertEqual(bs.aq.disagreement(case, undo, probe), 0.0)
        self.assertGreater(bs.aq.disagreement(case, [], probe), 0.0)

    def test_summary_counts_each_line(self):
        rows = [{"corrupted_disagreement": 0.1, "targeted": [step(2, True, True, 5e-5)]},
                {"corrupted_disagreement": 0.3, "targeted": [step(1, False, True, 5e-3)]}]
        out = bs.summarize(rows, "pairs")["targeted"][0]
        self.assertEqual((out["all_entries"], out["found_visible"], out["success"]), (1, 1, 2))
        self.assertEqual((out["under_0.01"], out["under_0.001"], out["under_0.0001"]), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
