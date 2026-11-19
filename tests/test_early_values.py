import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("early_values", ROOT / "experiments/locate-repair/early_values.py")
ev = importlib.util.module_from_spec(spec)
sys.modules["early_values"] = ev
spec.loader.exec_module(ev)


class EarlyValuesTests(unittest.TestCase):
    real_run = staticmethod(ev.st.run)

    def setUp(self):
        self.case = ev.rs.resize(ev.t3.corrupt(8, 0))
        px, py, _ = ev.lr.split(self.case["practice"])
        self.triple = tuple(tuple(f[:2]) for f in self.case["faults"])
        self.fit = ev.st.told_start(self.case, px, py, self.triple)
        (accuracy, cost), edits = self.fit
        shifted = [(n, i, v + 0.5 * self.case["scale"][n]) for n, i, v in edits]
        self.fits = {self.triple: self.fit, "shifted": ((accuracy, cost), shifted)}
        self.pool = np.random.default_rng(0).uniform(-2, 2, size=(2000, 2))

    def test_mixes_for_the_early_rounds_then_only_splits(self):
        choose = ev.make_chooser(early=2)
        counts = [choose(self.case, self.fits, self.fit[1], self.pool, np.random.default_rng(r))[1]["value_queries"]
                  for r in range(4)]
        self.assertEqual(counts, [ev.aq.PER_ROUND // 2] * 2 + [0, 0])

    def test_switch_point_is_set_by_early(self):
        for early in (0, 3):
            choose = ev.make_chooser(early=early)
            counts = [choose(self.case, self.fits, self.fit[1], self.pool, np.random.default_rng(r))[1]["value_queries"]
                      for r in range(5)]
            self.assertEqual(counts, [ev.aq.PER_ROUND // 2] * early + [0] * (5 - early))

    def test_each_case_gets_its_own_round_count(self):
        first, second = ev.make_chooser(early=1), ev.make_chooser(early=1)
        first(self.case, self.fits, self.fit[1], self.pool, np.random.default_rng(0))
        _, info = second(self.case, self.fits, self.fit[1], self.pool, np.random.default_rng(0))
        self.assertEqual(info["value_queries"], ev.aq.PER_ROUND // 2)


    def test_fresh_switch_evidence_matches_its_arms(self):
        evidence = ROOT / "experiments/locate-repair/evidence/fresh-switch"
        import json
        for name, early, start in (("early-1", 1, 0), ("fresh-early-0", 0, 25), ("fresh-early-2", 2, 25),
                                   ("fresh-early-4", 4, 25)):
            report = json.loads((evidence / f"{name}.json").read_text())
            self.assertEqual(report["early_rounds"], early)
            self.assertEqual(sorted(c["seed"] for c in report["cases"]), list(range(start, start + 25)))
            for case in report["cases"]:
                asked = [h.get("value_queries", 0) for h in case["switch"][:-1]]
                self.assertEqual(sum(1 for a in asked if a), early)


    def test_more_pairs_kept_means_more_triples(self):
        px, py, _ = ev.lr.split(self.case["practice"])
        _, _, few = ev.tc.committee(self.case, px, py, size=5, pairs_kept=2)
        members, fits, more = ev.tc.committee(self.case, px, py, size=5, pairs_kept=6)
        self.assertGreater(more, few)
        self.assertEqual(len(members), 5)
        self.assertEqual(set(fits), set(members))

    def test_wide_screen_evidence_covers_all_fifty_cases(self):
        import json
        evidence = ROOT / "experiments/locate-repair/evidence/wide-screen"
        report = json.loads((evidence / "wide-screen.json").read_text())
        self.assertEqual(report["pairs_kept"], 1000)
        self.assertEqual(report["early_rounds"], 2)
        self.assertEqual(sorted(c["seed"] for c in report["cases"]), list(range(50)))
        comparison = json.loads((evidence / "comparison.json").read_text())
        self.assertEqual(comparison["pairs_1000"]["under_0.1%"], sum(
            c["switch"][-1]["disagreement"] < 1e-3 for c in report["cases"]))
        self.assertEqual(comparison["committee_lost"], [])

    def test_adaptive_widen_evidence_and_both_flags(self):
        import json
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        import adaptive_check as ac
        evidence = ROOT / "experiments/locate-repair/evidence/adaptive-widen"
        report = json.loads((evidence / "adaptive-widen.json").read_text())
        self.assertEqual(report["widen_to"], 1000)
        self.assertEqual(sorted(c["seed"] for c in report["cases"]), list(range(50, 100)))
        check = json.loads((evidence / "check.json").read_text())
        fails = [c["seed"] for c in report["cases"] if c["switch"][-1]["disagreement"] >= 1e-3]
        self.assertEqual([f["seed"] for f in check["fresh"]["failures"]], fails)
        self.assertEqual(check["end_flag"]["seeds_50_99"]["failures_missed"], [])
        self.assertEqual(check["end_flag"]["seeds_0_49"]["failures_missed"], [])
        row = {"switch": [{"practice_accuracy": 1.0}, {"practice_accuracy": 0.99}], "first_best_practice_accuracy": 1.0}
        self.assertFalse(ac.start_flag(row))
        self.assertTrue(ac.end_flag(row))

    def test_end_flag_rerun_evidence(self):
        import json
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        import end_flag_check as ef
        evidence = ROOT / "experiments/locate-repair/evidence"
        report = json.loads((evidence / "end-flag/end-flag.json").read_text())
        self.assertEqual(report["pairs_kept"], 1000)
        check = json.loads((evidence / "end-flag/check.json").read_text())
        self.assertEqual(sorted(c["seed"] for c in report["cases"]), check["seeds_50_99"]["flagged"])
        self.assertEqual(check["seeds_50_99"]["broken"] + check["seeds_0_49"]["broken"], [])
        self.assertEqual(check["all_100"]["under_0.1%"], 100 - len(check["failures_left"]))
        narrow = {1: {"switch": [{"practice_accuracy": 0.9, "disagreement": 0.01, "faults_found": 1, "found_visible": False}], "seconds": 10,
                      "committee_has_true_triple": False}}
        wide = {1: {"switch": [{"practice_accuracy": 1.0, "disagreement": 0.0, "faults_found": 3, "found_visible": True}], "seconds": 30,
                    "committee_has_true_triple": True}}
        arm, summary, changes, _ = ef.apply_rule({**narrow, 2: {**narrow[1], "switch": [{"practice_accuracy": 1.0, "disagreement": 0.0, "faults_found": 3, "found_visible": True}]}}, wide)
        self.assertEqual(changes["fixed"], [1])
        self.assertAlmostEqual(summary["cpu_hours"], 50 / 3600)
        with self.assertRaises(ValueError):
            ef.apply_rule(narrow, {})

    def test_rank_misses_evidence(self):
        import json
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        import rank_misses as rm
        evidence = ROOT / "experiments/locate-repair/evidence"
        summary = json.loads((evidence / "rank-misses/summary.json").read_text())
        end = json.loads((evidence / "end-flag/end-flag.json").read_text())
        lacking = sorted(c["seed"] for c in end["cases"] if not c["committee_has_true_triple"])
        self.assertEqual(sorted(r["seed"] for r in summary["cases"]), lacking)
        self.assertEqual(summary["stages"]["cut"], lacking)
        self.assertEqual(rm.stage({"in_triples": False}), "never_extended")
        self.assertEqual(rm.stage({"in_triples": True, "rank_continuous": 199}), "kept")
        self.assertEqual(rm.stage({"in_triples": True, "rank_continuous": 200}), "cut")

    def test_committee_size_is_passed_through(self):
        sizes = []
        real = ev.tc.committee

        def committee(case, px, py, pairs_kept=None, **kw):
            sizes.append(kw.get("size"))
            return real(case, px, py, size=4, pairs_kept=3)

        ev.tc.committee, ev.st.run = committee, lambda *a, **k: [{"disagreement": 0.0}]
        try:
            row = ev.analyse(self.case, committee_size=9)
            self.assertEqual(sizes, [9])
            self.assertEqual(row["committee_size"], 4)
            sizes.clear()
            ev.analyse(self.case)
            self.assertEqual(sizes, [None])
        finally:
            ev.tc.committee = real
            ev.st.run = type(self).real_run

    def test_big_committee_evidence(self):
        import json
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        import big_committee_check as bc
        evidence = ROOT / "experiments/locate-repair/evidence"
        report = json.loads((evidence / "big-committee/big-committee.json").read_text())
        self.assertEqual((report["committee"], report["pairs_kept"], report["early_rounds"]), (1000, 200, 2))
        check = json.loads((evidence / "big-committee/check.json").read_text())
        self.assertEqual(sorted(c["seed"] for c in report["cases"]), check["larger_committee"]["flagged"])
        self.assertEqual(check["larger_committee"]["broken"], [])
        self.assertEqual(check["both_in_turn"]["under_0.1%"], 50 - len(check["both_in_turn"]["failures_left"]))
        row = lambda acc, d: {"switch": [{"practice_accuracy": acc, "disagreement": d, "faults_found": 3,
                                          "found_visible": True}], "seconds": 1, "committee_has_true_triple": True}
        arm, _, steps = bc.two_step({1: row(0.9, 0.01), 2: row(0.9, 0.01)}, {1: row(1.0, 0.0), 2: row(0.9, 0.01)},
                                    {2: row(1.0, 0.0)})
        self.assertEqual((steps["flagged_again"], steps["failures_left"]), ([2], []))

    def test_widens_only_when_no_member_fits_every_practice_label(self):
        calls = []
        real = ev.tc.committee

        def committee(case, px, py, pairs_kept=None, **kw):
            calls.append(pairs_kept)
            members, fits, count = real(case, px, py, size=4, pairs_kept=3)
            accuracy = 0.99 if pairs_kept is None else 1.0
            fits = {t: ((accuracy, f[0][1]), f[1]) for t, f in fits.items()}
            return members, fits, count

        ev.tc.committee, ev.st.run = committee, lambda *a, **k: [{"disagreement": 0.0}]
        try:
            row = ev.analyse(self.case, widen_to=7)
            self.assertEqual(calls, [None, 7])
            self.assertTrue(row["widened"])
            self.assertEqual(row["first_best_practice_accuracy"], 0.99)
            calls.clear()
            ev.analyse(self.case)
            self.assertEqual(calls, [None])
        finally:
            ev.tc.committee = real
            ev.st.run = type(self).real_run


if __name__ == "__main__":
    unittest.main()
