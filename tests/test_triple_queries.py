import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("triple_queries", ROOT / "experiments/locate-repair/triple_queries.py")
tq = importlib.util.module_from_spec(spec)
sys.modules["triple_queries"] = tq
spec.loader.exec_module(tq)


class TripleQueryTests(unittest.TestCase):
    def test_true_triple_committee_records_every_round(self):
        case = tq.t3.corrupt(4, 1)
        true = tuple(tuple(f) for f in case["faults"])
        history = tq.run(case, [true], tq.cr.silent_faults(case))
        self.assertEqual(len(history), tq.aq.ROUNDS + 1)
        self.assertEqual([h["queries"] for h in history], [tq.aq.PER_ROUND * k for k in range(tq.aq.ROUNDS + 1)])
        self.assertTrue(all(h["faults_found"] == 3 and h["found_visible"] for h in history))

    def test_summary_counts_the_last_round(self):
        step = {"queries": 0, "faults_found": 3, "found_visible": True, "success": True, "damage_undone": 0.9}
        rows = [{"silent_faults": [], "committee_has_true_triple": True, "targeted": [step]},
                {"silent_faults": [2], "committee_has_true_triple": False,
                 "targeted": [dict(step, faults_found=2, success=False, damage_undone=0.5)]}]
        out = tq.summarize(rows)
        self.assertEqual(out["silent_cases"], 1)
        self.assertEqual(out["targeted"][0]["found_all_three"], 1)
        self.assertEqual(out["targeted"][0]["found_visible"], 2)
        self.assertEqual(out["targeted"][0]["success"], 1)


if __name__ == "__main__":
    unittest.main()
