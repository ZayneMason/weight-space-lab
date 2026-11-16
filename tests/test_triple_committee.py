import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("triple_committee", ROOT / "experiments/locate-repair/triple_committee.py")
tc = importlib.util.module_from_spec(spec)
sys.modules["triple_committee"] = tc
spec.loader.exec_module(tc)


def step(found, visible, ok, accuracy, true, told, tied):
    return {"queries": 0, "faults_found": found, "found_visible": visible, "success": ok, "practice_accuracy": accuracy,
            "true_triple_accuracy": true, "told_accuracy": told, "tied_best": tied}


class TripleCommitteeTests(unittest.TestCase):
    def test_summary_counts_each_round(self):
        rows = [{"silent_faults": [], "committee_has_true_triple": True,
                 "targeted": [step(3, True, True, 1.0, 1.0, 1.0, 2)]},
                {"silent_faults": [1], "committee_has_true_triple": False,
                 "targeted": [step(2, True, False, 0.99, None, 1.0, 6)]}]
        out = tc.summarize(rows)
        self.assertEqual(out["committee_has_true_triple"], 1)
        self.assertEqual(out["silent_cases"], 1)
        self.assertEqual(out["targeted"][0], {"queries": 0, "all_three": 1, "found_visible": 2, "success": 1,
                                              "winner_fits_all": 1, "told_fits_all": 2, "true_triple_fits_all": 1,
                                              "tied_best_median": 4.0})

    def test_refit_never_loses_to_its_starting_values(self):
        case = tc.rs.resize(tc.t3.corrupt(8, 0))
        px, py, _ = tc.lr.split(case["practice"])
        triple = tuple(tuple(f[:2]) for f in case["faults"])
        start = tc.wt.score(case, px, py, triple, tc.ct.current_values(case, triple))
        self.assertGreaterEqual(tc.refit(case, px, py, triple, start)[0][0], start[0][0])


if __name__ == "__main__":
    unittest.main()
