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

    def test_two_step_evidence(self):
        import json
        evidence = ROOT / "experiments/locate-repair/evidence"
        old = json.loads((evidence / "two-step/old-big.json").read_text())
        self.assertEqual((old["committee"], old["pairs_kept"], old["early_rounds"]), (1000, 200, 2))
        check = json.loads((evidence / "two-step/check.json").read_text())
        self.assertEqual(sorted(c["seed"] for c in old["cases"]), check["seeds_0_49"]["both_in_turn"]["flagged"])
        self.assertEqual(check["all_100"]["under_0.1%"], 100 - len(check["all_100"]["failures_left"]))
        self.assertEqual(check["all_100"]["under_0.1%"], sum(check[k]["both_in_turn"]["under_0.1%"]
                                                            for k in ("seeds_0_49", "seeds_50_99")))

    def test_two_step_unseen_evidence(self):
        import json
        evidence = ROOT / "experiments/locate-repair/evidence/two-step-unseen"
        report = json.loads((evidence / "two-step-unseen.json").read_text())
        self.assertEqual((report["committee"], report["early_rounds"], report["two_step"]), (200, 2, True))
        seeds = sorted(c["seed"] for c in report["cases"])
        self.assertEqual(seeds, list(range(100, 150)))
        check = json.loads((evidence / "check.json").read_text())
        self.assertEqual(check["seeds"], 50)
        self.assertEqual(sum(check["stages_run"].values()), 50)
        self.assertEqual(check["two_step"]["under_0.1%"], 50 - len(check["failures_left"]))
        reran = sorted(c["seed"] for c in report["cases"] if c["stages_run"] > 1)
        self.assertEqual(reran, check["reran_with_bigger_committee"])

    def test_unseen_misses_evidence(self):
        import json
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from unseen_misses import where
        self.assertEqual(where({"in_triples": False}, 200), "never_extended")
        self.assertEqual(where({"in_triples": True, "rank_continuous": 297}, 200), "cut")
        self.assertEqual(where({"in_triples": True, "rank_continuous": 297}, 1000), "kept")
        evidence = ROOT / "experiments/locate-repair/evidence/unseen-misses"
        summary = json.loads((evidence / "summary.json").read_text())
        self.assertEqual([c["seed"] for c in summary["cases"]], [124, 126, 149])
        check = json.loads((evidence.parent / "two-step-unseen/check.json").read_text())
        self.assertEqual([c["seed"] for c in summary["cases"]], check["still_misses_a_label"])
        for c in summary["cases"]:
            wide = json.loads((evidence / f"rank-{c['seed']}-1000.json").read_text())
            self.assertEqual(wide["limit"], 1000)
            self.assertTrue(wide["in_triples"])

    def test_three_step_adds_both_settings_last(self):
        calls = []
        accuracies = iter([0.9, 0.9, 0.9, 1.0])
        real = ev.analyse

        def analyse(case, early, **options):
            calls.append(options)
            return {"switch": [{"practice_accuracy": next(accuracies), "disagreement": 0.0, "faults_found": 3}],
                    "committee_has_true_triple": True, "seconds": 1.0}

        ev.analyse = analyse
        try:
            row = ev.two_step(self.case, stages=ev.THREE_STEP)
            self.assertEqual(calls[-1], {"pairs_kept": 1000, "committee_size": 1000})
            self.assertEqual(row["stages_run"], 4)
        finally:
            ev.analyse = real

    def test_both_wide_evidence(self):
        import json
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from both_wide_check import score
        evidence = ROOT / "experiments/locate-repair/evidence"
        both = json.loads((evidence / "both-wide/both-wide.json").read_text())
        self.assertEqual((both["pairs_kept"], both["committee"], both["early_rounds"]), (1000, 1000, 2))
        check = json.loads((evidence / "both-wide/check.json").read_text())
        unseen = json.loads((evidence / "two-step-unseen/check.json").read_text())
        self.assertEqual(check["flagged"], unseen["still_misses_a_label"])
        self.assertEqual(sorted(c["seed"] for c in both["cases"]), check["flagged"])
        self.assertEqual(check["three_step"]["under_0.1%"], 50)
        with self.assertRaises(ValueError):
            score({1: {"switch": [{"practice_accuracy": 0.9}]}}, {})

    def test_largest_edit_is_the_biggest_scaled_change(self):
        import numpy as np
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from early_values import tc
        case = {"arrays": {"w": np.array([[1.0, 2.0]]), "b": np.array([0.5])}, "scale": {"w": 2.0, "b": 0.25}}
        fit = ((1.0, -1.0), [("w", 0, 2.0), ("w", 1, 1.0), ("b", 0, 0.6)])
        self.assertAlmostEqual(tc.largest_edit(case, fit), 0.5)  # |2-1|/2 = |1-2|/2 = 0.5 > 0.1/0.25 = 0.4
        with self.assertRaises(ValueError):
            tc.committee({"arrays": {}}, None, None, tie_break="nope")

    def test_tie_break_rank_orders_by_accuracy_then_key(self):
        import numpy as np
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from tie_rank import rank
        f = {"acc": np.array([1.0, 1.0, 0.9, 1.0]), "cost": np.array([3.0, 1.0, 0.0, 2.0]),
             "margin_min": np.array([0.5, np.nan, 9.0, 0.1]), "is_true": np.array([False, False, False, True])}
        self.assertEqual(rank(f, "cost", 1), 2)  # perfect fits by cost: 1.0, 2.0 (true), 3.0
        self.assertEqual(rank(f, "margin_min", -1), 2)  # larger margin first, NaN last: 0.5, 0.1 (true), NaN
        self.assertIsNone(rank({**f, "is_true": np.zeros(4, bool)}, "cost", 1))

    def test_tie_breaks_evidence(self):
        import json
        evidence = ROOT / "experiments/locate-repair/evidence/tie-breaks"
        out = json.loads((evidence / "summary.json").read_text())
        self.assertEqual((out["lost_cases"], out["controls"], out["committee"]), (12, 8, 200))
        self.assertEqual(len(out["cases"]), 20)  # the .npz feature files are git-ignored; summary.json is the record
        current = out["by_tie_break"]["coarse_rank (current)"]
        self.assertEqual((current["lost_kept"], current["controls_kept"]), (0, 8))

    def test_largest_edit_check_compares_per_seed(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from largest_edit_check import compare
        row = lambda seed, d, found, true=True, k=1, sec=10.0: {"seed": seed, "stages_run": k, "seconds": sec,
            "committee_has_true_triple": true,
            "switch": [{"disagreement": d, "faults_found": found, "found_visible": found == 3, "practice_accuracy": 1.0}]}
        coarse = {1: row(1, 5e-5, 2), 2: row(2, 2e-4, 3, sec=20.0)}
        largest = {1: row(1, 2e-3, 0, true=False, k=3, sec=40.0), 2: row(2, 5e-5, 3, sec=20.0)}
        out = compare(coarse, largest)
        self.assertEqual(out["line_0.001"], {"gained": [], "lost": [1]})
        self.assertEqual(out["line_0.0001"], {"gained": [2], "lost": [1]})
        self.assertEqual((out["fewer_true_weights"], out["true_triple_lost"]), ([1], [1]))
        self.assertEqual((out["closer"], out["further"], out["failures_left"]), (1, 1, [1]))
        self.assertAlmostEqual(out["cost_ratio"], 2.0)
        with self.assertRaises(ValueError):
            compare({}, largest)

    def test_largest_edit_evidence(self):
        import json
        evidence = ROOT / "experiments/locate-repair/evidence/largest-edit"
        report = json.loads((evidence / "largest-edit.json").read_text())
        self.assertEqual(report["tie_break"], "largest-edit")
        self.assertEqual(sorted(c["seed"] for c in report["cases"]), list(range(100, 150)))
        check = json.loads((evidence / "check.json").read_text())
        self.assertEqual((check["largest_edit"]["under_0.1%"], check["coarse"]["under_0.1%"]), (49, 50))
        self.assertEqual(check["line_0.001"]["lost"], [104])

    def test_late_edit_check_compares_per_seed(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from late_edit_check import compare
        row = lambda seed, d, found, k=3, sec=10.0: {"seed": seed, "stages_run": k, "seconds": sec,
            "committee_has_true_triple": found == 3, "two_step": [{"seconds": sec}],
            "switch": [{"disagreement": d, "faults_found": found}]}
        coarse = {1: row(1, 2e-4, 3), 2: row(2, 1e-4, 1, sec=20.0), 3: row(3, 5e-5, 3, k=2)}
        late = {1: row(1, 5e-5, 3), 2: row(2, 6e-4, 0, sec=30.0), 3: row(3, 5e-5, 3, k=2)}
        out = compare(coarse, late)
        self.assertEqual((out["reached_last_stage"], out["stopped_earlier"]), ([1, 2], [3]))
        self.assertTrue(out["unchanged_before_last_stage"])
        self.assertEqual(out["line_0.0001"], {"gained": [1], "lost": []})
        self.assertEqual((out["closer"], out["further"], out["fewer_true_weights"]), ([1], [2], [2]))
        self.assertAlmostEqual(out["last_stage_cpu_hours"]["late"], 40.0 / 3600)
        with self.assertRaises(ValueError):
            compare({}, late)

    def test_late_edit_evidence(self):
        import json
        evidence = ROOT / "experiments/locate-repair/evidence/late-edit"
        report = json.loads((evidence / "late-edit.json").read_text())
        self.assertEqual((report["tie_break"], report["tie_break_from"]), ("largest-edit", 2))
        self.assertEqual(len(report["cases"]), 15)
        check = json.loads((evidence / "check.json").read_text())
        self.assertEqual(check["reached_last_stage"], [102, 109, 124, 126, 129, 149])
        self.assertTrue(check["unchanged_before_last_stage"])
        self.assertEqual(check["line_0.001"], {"gained": [], "lost": []})
        self.assertEqual(check["fewer_true_weights"], [124])

    def test_width32_check_scores_shares(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from width32_check import compare
        row = lambda seed, d, found, k=1, sec=3600.0, triples=100: {"seed": seed, "stages_run": k, "seconds": sec,
            "triples": triples, "committee_has_true_triple": found == 3,
            "switch": [{"disagreement": d, "faults_found": found, "found_visible": found == 3}]}
        base = {1: row(1, 5e-5, 3, sec=1800.0), 2: row(2, 2e-4, 2, sec=1800.0)}
        wide = {10: row(10, 5e-5, 3, triples=300), 11: row(11, 2e-4, 1, k=3, sec=10800.0, triples=200)}
        stage = lambda d, sec: {"disagreement": d, "faults_found": 1, "practice_accuracy": 0.99,
                                "committee_has_true_triple": False, "seconds": sec}
        wide[11]["two_step"] = [stage(3e-4, 1800.0), stage(8e-5, 1800.0), stage(2e-4, 7200.0)]
        out = compare(base, wide)
        third = out["third_stage"]["width_32"]
        self.assertEqual((third["reached"], third["worse"], third["better"]), ([11], [11], []))
        self.assertAlmostEqual(third["cpu_hours"], 2.0)
        self.assertEqual(third["without"], {"under_0.1%": 2, "under_0.01%": 2, "exact": 1})
        self.assertEqual(out["width_32"]["share_under_0.01%"], 0.5)
        self.assertEqual(out["width_32"]["stages_run"], {"1": 1, "2": 0, "3": 1})
        self.assertAlmostEqual(out["cost_ratio_per_case"], 4.0)
        self.assertAlmostEqual(out["triples_ratio"], 2.5)
        self.assertEqual((out["width_32_not_under_0.01%"], out["width_32_not_exact"]), ([11], [11]))

    def test_width32_evidence(self):
        import json
        evidence = ROOT / "experiments/locate-repair/evidence/width-32"
        report = json.loads((evidence / "width-32.json").read_text())
        self.assertEqual((report["width"], len(report["cases"])), (32, 12))
        check = json.loads((evidence / "check.json").read_text())
        self.assertEqual(check["width_32"]["under_0.1%"], 12)
        self.assertEqual((check["width_32"]["under_0.01%"], check["width_32"]["exact"]), (8, 5))
        self.assertEqual(check["width_32"]["stages_run"], {"1": 7, "2": 1, "3": 4})
        self.assertEqual(check["third_stage"]["width_32"]["worse"], [202, 205, 210, 211])
        self.assertEqual(check["third_stage"]["width_16"]["better"], [102, 109, 129])

    def test_final_stage_choice_rules(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from final_stage_choice import best_practice_earlier, compare, third_only_if_perfect
        st = lambda d, acc, found=1, sec=3600.0: {"disagreement": d, "practice_accuracy": acc, "faults_found": found,
                                                  "seconds": sec}
        rows = {1: {"two_step": [st(5e-4, 0.98), st(8e-5, 0.999), st(3e-4, 0.995)]},
                2: {"two_step": [st(5e-4, 0.98), st(4e-4, 0.99), st(2e-5, 1.0, found=3)]},
                3: {"two_step": [st(3e-5, 1.0, found=3)]}}
        self.assertEqual([third_only_if_perfect(rows[s]["two_step"]) for s in (1, 2, 3)], [1, 2, 0])
        self.assertEqual(best_practice_earlier([st(1e-4, 1.0), st(2e-4, 1.0)]), 0)
        out = compare({"arm": rows})["arm"]
        self.assertEqual(out["reached_third_stage"], [1, 2])
        rule = out["rules"]["third_only_if_perfect"]
        self.assertEqual((rule["under_0.01%"], rule["exact"], rule["better"], rule["worse"]), (3, 2, [1], []))
        skip = out["rules"]["no_third_stage"]
        self.assertEqual((skip["worse"], skip["cpu_hours"]), ([2], 5.0))

    def test_final_stage_choice_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/final-stage-choice/check.json").read_text())
        w16, w32 = check["width_16"]["rules"], check["width_32"]["rules"]
        self.assertEqual((w16["current"]["under_0.01%"], w32["current"]["under_0.01%"]), (35, 8))
        self.assertEqual((w16["third_only_if_perfect"]["worse"], w32["third_only_if_perfect"]["worse"]), ([], []))
        self.assertEqual(w32["third_only_if_perfect"]["under_0.01%"], 10)
        self.assertEqual(check["width_16_largest_edit"]["rules"]["third_only_if_perfect"]["worse"], [126])

    def test_third_rule_check_counts_moves_and_exact_misses(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from third_rule_check import misses, third_stage_moves
        st = lambda d, acc, found=1: {"disagreement": d, "practice_accuracy": acc, "faults_found": found, "seconds": 1.0}
        rows = {1: {"two_step": [st(5e-4, 0.98), st(4e-4, 0.99), st(3e-4, 0.995)]},
                2: {"two_step": [st(5e-4, 0.98), st(2e-4, 0.99), st(6e-4, 0.995)]},
                3: {"two_step": [st(5e-4, 0.98), st(4e-4, 0.99), st(2e-5, 1.0, found=3)]},
                4: {"two_step": [st(2e-4, 1.0, found=3)]}}
        self.assertEqual(third_stage_moves(rows), {"1": "closer", "2": "further"})
        self.assertEqual(misses(rows), {"over_0.01%": 3, "exact_over_0.01%": [4]})

    def test_third_rule_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/third-stage-rule/check.json").read_text())
        self.assertTrue(check["kept_stage_matches_rescore"])
        self.assertEqual(check["stages_run"], {"1": 36, "2": 7, "3": 7})
        fresh = check["fresh"]["rules"]
        self.assertEqual([fresh["current"][k] for k in ("under_0.1%", "under_0.01%", "exact")], [49, 32, 31])
        self.assertEqual(fresh["third_only_if_perfect"]["worse"], [160, 170, 186, 197])
        self.assertEqual(check["pooled"]["rules"]["third_only_if_perfect"]["under_0.1%"], 98)
        self.assertEqual(len(check["pooled_misses"]["exact_over_0.01%"]), 17)

    def test_refine_values_summaries(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from refine_values import by_chord, same_direction, summarize, WALKS
        walk = lambda d, chord: {"disagreement": d, "accuracy": 1.0, "mean_chord": chord, "ratios": []}
        row = lambda s, rep, d, chord: {"seed": s, "repair": rep, **{w: walk(d, chord) for w in WALKS}}
        rows = [row(1, 2e-4, 5e-5, 0.01), row(2, 5e-5, 2e-4, 0.001), row(3, 3e-4, 2e-4, 0.003)]
        out = summarize(rows, "long")
        self.assertEqual((out["repair_under_0.01%"], out["centre_under_0.01%"]), (1, 1))
        self.assertEqual((out["closer"], out["further"]), (2, 1))
        self.assertEqual((out["crossed_under_0.01%"], out["crossed_over_0.01%"]), ([1], [2]))
        self.assertEqual([b["cases"] for b in by_chord(rows)], [1, 1, 1])
        self.assertEqual(by_chord(rows)[2]["closer_in_every_walk"], 1)
        self.assertEqual(same_direction(rows), 3)

    def test_refine_values_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/refine-values/check.json").read_text())
        self.assertTrue(check["replays_match_saved"])
        self.assertEqual(len(check["cases"]), 58)
        self.assertEqual({w: v["centre_under_0.01%"] for w, v in check["summary"].items()},
                         {"short": 42, "short_again": 41, "long": 43})
        self.assertEqual(check["summary"]["long"]["repair_under_0.01%"], 41)
        self.assertEqual([b["closer_in_every_walk"] for b in check["by_chord"]], [7, 6, 11])

    def test_middle_step_uses_the_middle_only_when_the_region_is_wide(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from switch_triples import told_start
        case = ev.rs.resize(ev.t3.corrupt(16, 164))
        px, py, _ = ev.lr.split(case["practice"])
        triple = tuple(tuple(f[:2]) for f in case["faults"])
        keep = {"x": px, "y": py, "winner": triple, "edits": told_start(case, px, py, triple)[1]}
        probe = ev.bs.make_probe(case, 20_000)
        wide = ev.middle_step(case, keep, probe, min_chord=0.0)
        self.assertTrue(wide["used"])
        self.assertEqual(wide["accuracy"], 1.0)
        self.assertFalse(ev.middle_step(case, keep, probe, min_chord=10.0)["used"])

    def test_middle_check_arms(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from middle_check import check
        row = lambda d, acc, mid, used, macc, chord: {
            "switch": [{"disagreement": d, "practice_accuracy": acc, "faults_found": 3}], "stages_run": 1,
            "seconds": 3600.0, "middle": {"disagreement": mid, "used": used, "accuracy": macc, "mean_chord": chord}}
        rows = {1: row(2e-4, 1.0, 5e-5, True, 1.0, 0.01), 2: row(2e-4, 1.0, 9e-5, False, 1.0, 0.001),
                3: row(5e-5, 1.0, 2e-4, False, 0.99, 0.003)}
        out = check(rows)
        self.assertEqual(out["arms"]["rule"]["crossed_under_0.01%"], [1])
        self.assertEqual(out["arms"]["always"]["crossed_under_0.01%"], [1, 2])
        self.assertEqual(out["arms"]["always"]["crossed_over_0.01%"], [])
        self.assertEqual(out["middle_fits_fewer"], [3])
        self.assertEqual([b["cases"] for b in out["by_chord"]], [1, 1, 1])

    def test_middle_step_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/middle-step/check.json").read_text())
        self.assertEqual((check["seeds"], check["exact"]), (50, 31))
        arms = check["arms"]
        self.assertEqual([arms[k]["under_0.01%"] for k in ("repair", "rule", "always")], [36, 37, 39])
        self.assertEqual(arms["always"]["crossed_under_0.01%"], [208, 224, 235])
        self.assertEqual(arms["always"]["crossed_over_0.01%"], [])
        self.assertEqual(check["rule_used"], [200, 209, 223, 224, 227, 230, 237, 238])

    def test_extra_round_scores_both_arms_from_one_pool(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from switch_triples import told_start
        case = ev.rs.resize(ev.t3.corrupt(16, 164))
        px, py, _ = ev.lr.split(case["practice"])
        true = tuple(tuple(f[:2]) for f in case["faults"])
        other = (true[0], true[1], ("bias", 0))
        fits = {t: told_start(case, px, py, t) for t in (true, other)}
        keep = {"x": px, "y": py, "winner": true, "edits": fits[true][1], "members": [true, other], "fits": fits}
        probe, samples = ev.bs.make_probe(case, 20_000), []
        ev.middle_step(case, keep, probe, samples=samples)
        self.assertEqual(len(samples), 1600)
        out, added = ev.extra_round(case, keep, samples, probe)
        self.assertEqual(set(out), {"splits", "region"})
        self.assertEqual(added["region"].shape, (8, 2))
        self.assertGreater(out["region"]["splitting"], 0)
        self.assertEqual(out["region"]["faults_found"], 3)

    def test_region_check_arms(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from region_check import check
        fit = lambda d, acc, mid, macc, w: {"disagreement": d, "practice_accuracy": acc, "faults_found": 3, "winner": w,
                                            "middle": {"disagreement": mid, "accuracy": macc}}
        w, v = [["bias", 0], ["bias", 1], ["bias", 2]], [["bias", 0], ["bias", 1], ["bias", 3]]

        def row(repair, splits, region):
            end = {k: repair[k] for k in ("disagreement", "practice_accuracy", "faults_found", "winner")}
            return {"switch": [end], "middle": repair["middle"], "stages_run": 1, "seconds": 3600.0,
                    "extra": {"splits": {**splits, "splitting": 1}, "region": {**region, "splitting": 3}}}
        rows = {1: row(fit(2e-4, 1.0, 1.5e-4, 1.0, w), fit(2e-4, 1.0, 9e-5, 0.99, w), fit(8e-5, 1.0, 5e-5, 1.0, v)),
                2: row(fit(5e-5, 1.0, 4e-5, 1.0, w),
                       fit(2e-4, 1.0, 2e-4, 1.0, w),
                       fit(3e-5, 1.0, 2e-5, 1.0, w))}
        out = check(rows)
        arms = out["arms"]
        self.assertEqual(arms["splits+middle"]["crossed_under_0.01%"], [])  # its middle fits fewer labels
        self.assertEqual(arms["splits+middle"]["crossed_over_0.01%"], [2])
        self.assertEqual(arms["region+middle"]["crossed_under_0.01%"], [1])
        self.assertEqual(arms["region+middle"]["under_0.01%"], 2)
        self.assertEqual(out["winner_changed"], {"splits": [], "region": [1]})
        self.assertEqual(out["region_vs_splits"], {"closer": [1, 2], "further": []})

    def test_region_labels_evidence(self):
        import json
        import numpy as np
        here = ROOT / "experiments/locate-repair/evidence/region-labels"
        check = json.loads((here / "check.json").read_text())
        arms = check["arms"]
        self.assertEqual(check["seeds"], 50)
        self.assertEqual([arms[k]["under_0.01%"] for k in ("repair", "middle", "splits+middle", "region",
                                                            "region+middle")], [36, 37, 41, 41, 41])
        self.assertEqual({arms[k]["exact"] for k in arms}, {30})
        self.assertEqual(len(arms["region+middle"]["closer"]), 42)
        self.assertEqual(arms["region+middle"]["crossed_over_0.01%"], [])
        self.assertEqual(len(check["region_vs_splits"]["closer"]), 32)
        labels = np.load(here / "labels/case-250.npz")
        self.assertEqual(labels["x"].shape, (1088, 2))
        self.assertEqual(labels["added_region"].shape, (8, 2))

    def test_rebuild_check_compares_with_the_region_arm(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from rebuild_committee import check, first
        fit = lambda d, faults, mid, macc: {"disagreement": d, "practice_accuracy": 1.0, "faults_found": faults,
                                            "middle": {"disagreement": mid, "accuracy": macc}}
        before = {1: {"extra": {"region": fit(2e-4, 2, 1.5e-4, 1.0)}, "committee_has_true_triple": False},
                  2: {"extra": {"region": fit(5e-5, 3, 4e-5, 1.0)}, "committee_has_true_triple": True}}
        head = {"same_as_winner": False, **fit(3e-5, 3, 2e-5, 1.0)}
        after = {1: {**fit(8e-5, 3, 6e-5, 0.99), "first_member": {"same_as_winner": True},
                     "committee_has_true_triple": True, "true_rank": 0, "seconds": 1800.0},
                 2: {**fit(3e-4, 1, 2e-4, 1.0), "first_member": head,
                     "committee_has_true_triple": True, "true_rank": 4, "seconds": 1800.0}}
        self.assertIs(first(after[1]), after[1])
        self.assertIs(first(after[2]), head)
        out = check(before, after)
        self.assertEqual(out["cpu_hours"], 1.0)
        self.assertEqual(out["true_entered"], [1])
        self.assertEqual(out["became_exact"], [1])
        self.assertEqual(out["stopped_exact"], [2])
        self.assertEqual(out["crossed_under_0.01%"], [1])  # the middle fits fewer labels, so the fit counts
        self.assertEqual(out["crossed_over_0.01%"], [2])
        self.assertEqual(out["first_member"]["exact"], 2)
        self.assertEqual(out["first_member"]["differs_from_winner"], [2])
        self.assertEqual(out["first_member"]["closer_than_winner"], [2])
        self.assertEqual(out["after"]["true_in_committee"], 2)

    def test_rebuild_committee_evidence(self):
        import json
        here = ROOT / "experiments/locate-repair/evidence/rebuild-committee"
        check = json.loads((here / "check.json").read_text())
        rows = json.loads((here / "rebuild-committee.json").read_text())
        self.assertEqual(check["seeds"], 50)
        self.assertEqual({r["labels"] for r in rows}, {1096})
        self.assertEqual((check["before"]["true_in_committee"], check["after"]["true_in_committee"]), (35, 46))
        self.assertEqual((check["before"]["exact"], check["after"]["exact"], check["first_member"]["exact"]),
                         (30, 30, 34))
        self.assertEqual((check["before"]["under_0.01%"], check["after"]["under_0.01%"]), (41, 42))
        self.assertEqual(check["true_left"], [258])
        tied = [r["seed"] for r in rows if r["committee_has_true_triple"] and r["faults_found"] != 3
                and r["true_accuracy"] == r["practice_accuracy"]]
        self.assertEqual(len(tied), 12)

    def test_tie_break_pick_and_check(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from tie_break_final import check, pick
        fits = {"a": ((0.99, -1.0), None), "b": ((1.0, -3.0), None), "c": ((1.0, -2.0), None)}
        self.assertEqual(pick(["a", "b", "c"], fits, "cost", None), "c")  # most labels first, then cheapest
        self.assertEqual(pick(["a", "b", "c"], fits, "coarse", None), "b")  # committee order among the tied
        with self.assertRaises(ValueError):
            pick(["a"], fits, "nope", None)
        fit = lambda d, faults, mid, macc=1.0: {"disagreement": d, "practice_accuracy": 1.0, "faults_found": faults,
                                                "middle": {"disagreement": mid, "accuracy": macc}}
        row = lambda cost, coarse, has, tied, tied_refit: {
            "seconds": 1800.0, "committee_has_true_triple": has, "true_tied_top": tied, "true_tied_top_refit": tied_refit,
            "tied_top": 2, "tied_top_refit": 3, "true_from_clean": 1.0, "rules": {"cost": cost, "coarse": coarse}}
        rows = {1: row(fit(2e-4, 2, 1.5e-4), fit(5e-5, 3, 4e-5), True, True, True),
                2: row(fit(3e-5, 3, 2e-5), fit(3e-4, 1, 2e-4, 0.99), True, False, True)}
        out = check(rows)
        self.assertEqual(out["cpu_hours"], 1.0)
        self.assertEqual(out["rules"]["cost"]["exact"], 1)
        self.assertEqual(out["rules"]["coarse"]["closer"], [1])
        self.assertEqual(out["rules"]["coarse"]["crossed_under_0.01%"], [1])
        self.assertEqual(out["rules"]["coarse"]["crossed_over_0.01%"], [2])  # the middle fits fewer labels
        self.assertEqual((out["true_tied_top"], out["true_tied_top_refit"]), (1, 2))
        self.assertEqual(out["true_short"], [2])
        self.assertEqual(out["true_short_after_refit"], [])

    def test_tie_break_final_evidence(self):
        import json
        here = ROOT / "experiments/locate-repair/evidence/tie-break-final"
        check = json.loads((here / "check.json").read_text())
        rows = json.loads((here / "tie-break-final.json").read_text())
        self.assertEqual(check["seeds"], 50)
        self.assertEqual({r["labels"] for r in rows}, {1096})
        self.assertEqual((check["true_in_committee"], check["true_tied_top"], check["true_tied_top_refit"]),
                         (46, 42, 44))
        self.assertEqual(check["true_short_after_refit"], [259, 277])
        exact = {k: v["exact"] for k, v in check["rules"].items()}
        self.assertEqual(exact, {"cost": 30, "coarse": 34, "largest-edit": 34,
                                 "refit-cost": 31, "refit-coarse": 35, "refit-largest-edit": 35})
        self.assertEqual(check["rules"]["cost"]["under_0.01%"], 42)
        self.assertEqual(len(check["true_fits_every_label_from_clean"]), 50)

    def test_tied_labels_top_and_check(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from tied_labels import check, tied_top
        fits = {"a": ((0.99, -1.0), None), "b": ((1.0, -3.0), None), "c": ((1.0, -2.0), None)}
        self.assertEqual(tied_top(["a", "b", "c"], fits), ["b", "c"])  # committee order kept
        step = lambda k, d, faults, tied, mid=None: {"round": k, "disagreement": d, "practice_accuracy": 1.0,
                                                     "faults_found": faults, "tied": tied, "winner": [[k]],
                                                     "middle": {"disagreement": mid or d, "accuracy": 1.0}}
        rows = {1: {"seconds": 1800.0, "rounds": [step(0, 6e-4, 1, 3), step(1, 2e-5, 3, 1)]},
                2: {"seconds": 1800.0, "rounds": [step(0, 5e-5, 3, 1)]}}
        before = {1: {"rules": {"refit-coarse": {"winner": [[0]], "disagreement": 6e-4, "practice_accuracy": 1.0,
                                                 "faults_found": 1, "middle": {"disagreement": 6e-4, "accuracy": 1.0}}}},
                  2: {"rules": {"refit-coarse": {"winner": [[9]], "disagreement": 5e-5, "practice_accuracy": 1.0,
                                                 "faults_found": 3, "middle": {"disagreement": 5e-5, "accuracy": 1.0}}}}}
        out = check(rows, before)
        self.assertEqual(out["cpu_hours"], 1.0)
        self.assertEqual(out["reproduces_67"], [1])
        self.assertEqual(out["asked"], [1])
        self.assertEqual(out["rounds_run"], {"0": 1, "1": 1, "2": 0, "3": 0})
        self.assertEqual((out["by_round"]["0"]["exact"], out["by_round"]["1"]["exact"]), (1, 2))
        self.assertEqual(out["by_round"]["3"]["still_tied"], 0)  # a case that stopped keeps its last round
        self.assertEqual(out["became_exact"], [1])
        self.assertEqual(out["crossed_under_0.01%"], [1])
        self.assertAlmostEqual(out["geometric_mean_change"], (2e-5 / 6e-4) ** 0.5)

    def test_tied_labels_evidence(self):
        import json
        here = ROOT / "experiments/locate-repair/evidence/tied-labels"
        check = json.loads((here / "check.json").read_text())
        rows = json.loads((here / "tied-labels.json").read_text())
        self.assertEqual(check["seeds"], 50)
        self.assertEqual(len(check["reproduces_67"]), 50)
        self.assertEqual(len(check["asked"]), 44)
        self.assertEqual(check["rounds_run"], {"0": 6, "1": 2, "2": 1, "3": 41})
        lines = {k: (v["under_0.01%"], v["exact"], v["still_tied"]) for k, v in check["by_round"].items()}
        self.assertEqual(lines, {"0": (42, 35, 44), "1": (44, 36, 42), "2": (44, 37, 41), "3": (44, 37, 41)})
        self.assertEqual(check["became_exact"], [250, 255])
        self.assertEqual(check["crossed_under_0.01%"], [255, 287])
        self.assertEqual(check["crossed_over_0.01%"], [])
        silent = [r["seed"] for r in rows if len(r["rounds"]) > 1 and all(v["splitting"] == 0 for v in r["rounds"][1:])]
        self.assertEqual(len(silent), 19)

    def test_tied_agree_fold_and_check(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from tied_agree import check, fold, mirrors_expected
        self.assertEqual(fold(("weight", 21), 16), fold(("weight", 5), 16))  # logit 1's weight folds onto logit 0's
        self.assertEqual(fold(("bias", 1), 16), ("bias", 0))
        self.assertEqual(fold(("hidden_weight", 21), 16), ("hidden_weight", 21))
        self.assertEqual(mirrors_expected([["weight", 3], ["bias", 1], ["hidden_bias", 2]], 16), 4)
        member = lambda mirror, d, extra=(): {"mirror": mirror, "vs_winner": d, "extra": list(extra)}
        extra = {"layer": "hidden_bias", "state": "mixed", "size": 0.05}
        rows = {1: {"seconds": 3600.0, "winner": [["weight", 3], ["hidden_bias", 0], ["hidden_bias", 1]],
                    "rounds": [{"winner": [[1]]}], "members": [member(True, 0.0), member(True, 2e-5)]},
                2: {"seconds": 3600.0, "winner": [["hidden_bias", 0], ["hidden_bias", 1], ["hidden_bias", 2]],
                    "rounds": [{"winner": [[2]]}], "members": [member(True, 0.0), member(False, 0.0, [extra])]}}
        before = {1: {"rounds": [{"winner": [[1]]}]}, 2: {"rounds": [{"winner": [[3]]}]}}
        out = check(rows, before, {1: [0.1, 0.2, 0.3], 2: [0.1, 0.0, 0.3]})
        self.assertEqual(out["cpu_hours"], 2.0)
        self.assertEqual(out["matches_68"], 1)
        self.assertEqual((out["tied_members"], out["mirror_members"], out["same_on_probe_members"]), (4, 3, 3))
        self.assertEqual(out["all_mirrors"], [1])
        self.assertEqual(out["all_same_on_probe"], [2])
        self.assertEqual(out["more_than_mirrors"], [2])
        self.assertEqual(out["with_silent_fault"], [2])
        self.assertEqual(out["extra_entry_kinds"], {"hidden_bias:mixed": 1})
        self.assertEqual(out["per_case"]["1"]["mirrors_expected"], 2)
        self.assertIsNone(out["per_case"]["1"]["extra_size_median"])

    def test_tied_agree_evidence(self):
        import json
        here = ROOT / "experiments/locate-repair/evidence/tied-agree"
        check = json.loads((here / "check.json").read_text())
        self.assertEqual(check["seeds"], 41)
        self.assertEqual(check["matches_68"], 41)
        self.assertEqual((check["tied_members"], check["mirror_members"], check["same_on_probe_members"]),
                         (672, 134, 242))
        self.assertEqual(len(check["all_mirrors"]), 28)
        self.assertEqual(check["over_0.01%_from_winner"], [])
        self.assertEqual(check["with_silent_fault"], [253, 254, 277, 285])
        self.assertEqual(len(check["more_than_mirrors"]), 13)
        self.assertTrue(all(v["mirrors"] == v["mirrors_expected"] for v in check["per_case"].values()))

    def test_fix_six_margin_and_check(self):
        import numpy as np
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from fix_six import check, margin
        arrays = {"hidden_weight": np.array([[1.0, 0.0]]), "hidden_bias": np.array([0.0]),
                  "weight": np.array([[0.0], [2.0]]), "bias": np.array([0.5, 0.0])}
        x = np.array([[1.0, 0.0], [-1.0, 0.0]])
        self.assertTrue(np.allclose(margin(arrays, x), [1.5, -0.5]))
        fit = lambda d, faults=2: {"disagreement": d, "faults_found": faults, "practice_accuracy": 1.0,
                                   "middle": {"accuracy": 0.5, "disagreement": 0.0}}
        row = lambda arms, acc=1.0, w=1: {"seconds": 1800.0, "arms": arms, "rounds": [{"winner": [[w]]}],
                                          "true_check": {"accuracy": acc, "disagreement": 1e-4}}
        before = {1: {"rounds": [{"winner": [[1]], **fit(2e-4)}]}, 2: {"rounds": [{"winner": [[3]], **fit(5e-4, 3)}]}}
        rows = {1: row({"restarts": fit(2e-5, 3), "boundary": fit(3e-4)}),
                2: row({"restarts": fit(5e-4, 3), "boundary": fit(4e-4, 3)}, acc=0.9, w=2)}
        out = check(rows, before)
        self.assertEqual(out["cpu_hours"], 1.0)
        self.assertEqual(out["reproduces_68"], [1])
        self.assertEqual(out["arms"]["restarts"]["under_0.01%"], [1])
        self.assertEqual(out["arms"]["restarts"]["closer"], [1])
        self.assertEqual(out["arms"]["boundary"]["further"], [1])
        self.assertEqual(out["arms"]["boundary"]["exact"], [2])
        self.assertEqual(out["true_fits_every_label"], [1])

    def test_fix_six_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/fix-six/check.json").read_text())
        self.assertEqual(check["reproduces_68"], [258, 259, 260, 261, 269, 296])
        arms = check["arms"]
        self.assertEqual((arms["bigger"]["under_0.01%"], arms["restarts"]["under_0.01%"],
                          arms["boundary"]["under_0.01%"]), ([258], [259], [269]))
        self.assertTrue(all(a["further"] == [] for a in arms.values()))
        self.assertEqual(arms["boundary"]["closer"], [258, 259, 260, 261, 269, 296])
        self.assertEqual(check["true_fits_every_label"], [258, 259, 260, 269, 296])

    def test_pick_fixes_choose_and_check(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from pick_fixes import check, choose
        self.assertEqual(choose({"pick68": 3, "boundary": 3, "restarts": 1}), "pick68")
        self.assertEqual(choose({"pick68": 1, "boundary": 3, "restarts": 3}), "boundary")
        self.assertEqual(choose({"pick68": 0, "restarts": 2, "bigger": 2}), "restarts")
        cand = lambda d, faults=3: {"disagreement": d, "faults_found": faults, "winner": [[1]]}
        row = lambda cands, du, dd: {"seconds": 1800.0, "bigger_ran": "bigger" in cands, "candidates": cands,
                                     "choosers": {"uniform": {"pick": du}, "disagree": {"pick": dd, "split_inputs": 1}}}
        rows = {1: row({"pick68": cand(2e-4, 2), "restarts": cand(2e-4, 2), "boundary": cand(5e-5)}, "pick68", "boundary"),
                2: row({"pick68": cand(5e-5), "restarts": cand(5e-5), "boundary": cand(8e-5)}, "pick68", "boundary")}
        out = check(rows, {1: {"rounds": [{"winner": [[1]]}]}, 2: {"rounds": [{"winner": [[2]]}]}})
        self.assertEqual(out["cpu_hours"], 1.0)
        self.assertEqual(out["reproduces_68"], 1)
        self.assertEqual(out["picks"]["disagree"]["gained"], [1])
        self.assertEqual(out["picks"]["disagree"]["worse"], [2])
        self.assertEqual(out["picks"]["disagree"]["under_0.01%"], 2)
        self.assertEqual(out["picks"]["bigger"]["switched"], [])
        self.assertEqual(out["picks"]["probe_best"]["worse"], [])

    def test_pick_fixes_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/pick-fixes/check.json").read_text())
        self.assertEqual((check["seeds"], check["reproduces_68"]), (50, 50))
        self.assertEqual(check["bigger_ran"], [251, 258, 259, 261, 296])
        picks = check["picks"]
        self.assertEqual(picks["pick68"]["under_0.01%"], 44)
        self.assertEqual(picks["uniform"]["switched"], [])
        for name in ("disagree", "probe_best"):
            self.assertEqual((picks[name]["under_0.01%"], picks[name]["exact"]), (47, 39))
            self.assertEqual((picks[name]["gained"], picks[name]["lost"]), ([258, 259, 269], []))
        self.assertEqual(picks["disagree"]["worse"], [272])

    def test_fresh_repair_collect_and_score(self):
        import json
        import tempfile
        from pathlib import Path
        import numpy as np
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from fresh_repair import collect, score
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            for folder, seed, stages in (("p0", 300, 1), ("p1", 301, 3)):
                (src / folder).mkdir()
                (src / folder / f"case-{seed}.json").write_text(json.dumps({"seed": seed, "stages_run": stages}))
                (src / folder / f"case-{seed}.stage0.json").write_text("{}")
                np.savez(src / folder / f"case-{seed}.stage{stages - 1}.npz", x=np.zeros((1, 2)) + seed)
            rows = collect(src)
            self.assertEqual(sorted(rows), [300, 301])
            self.assertEqual(float(np.load(src / "labels" / "case-301.npz")["x"][0, 0]), 301.0)
            self.assertEqual([c["seed"] for c in json.loads((src / "cases.json").read_text())["cases"]], [300, 301])
        region = lambda d, faults: {"region": {"middle": None, "faults_found": faults, "final_d": d}}
        before = {1: {"seconds": 3600.0, "stages_run": 1, "extra": region(2e-4, 2)},
                  2: {"seconds": 3600.0, "stages_run": 3, "extra": region(5e-5, 3)}}
        cand = lambda d, faults=3: {"disagreement": d, "faults_found": faults, "winner": [[1]]}
        row = lambda cands, dd, asked: {"seconds": 1800.0, "bigger_ran": False, "candidates": cands,
                                        "choosers": {"uniform": {"pick": "pick68"},
                                                     "disagree": {"pick": dd, "split_inputs": asked, "asked": asked}}}
        rows = {1: row({"pick68": cand(2e-4), "restarts": cand(2e-4), "boundary": cand(5e-5)}, "boundary", 4),
                2: row({"pick68": cand(5e-5), "restarts": cand(5e-5), "boundary": cand(3e-4)}, "boundary", 2)}
        import fresh_repair
        real = fresh_repair.final
        fresh_repair.final = lambda e, middle: e["final_d"]
        try:
            out = score(before, rows)
        finally:
            fresh_repair.final = real
        self.assertEqual(out["repair_65"], {"under_0.01%": 1, "under_0.1%": 2, "exact": 1})
        self.assertEqual(out["disagree_vs_65"], {"gained": [1], "lost": [2]})
        self.assertEqual(out["over_line"], {"2": 3e-4})
        self.assertEqual((out["labels_asked"], out["cpu_hours_65"]), (6, 2.0))
        self.assertEqual(out["stages_run"], {"1": 1, "2": 0, "3": 1})

    def test_fresh_repair_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/fresh-repair/check.json").read_text())
        self.assertEqual(check["seeds"], 50)
        self.assertEqual(check["stages_run"], {"1": 38, "2": 9, "3": 3})
        self.assertEqual(check["repair_65"]["under_0.01%"], 41)
        picks = check["picks"]
        self.assertEqual(picks["pick68"]["under_0.01%"], 46)
        for name in ("disagree", "probe_best"):
            self.assertEqual((picks[name]["under_0.01%"], picks[name]["under_0.1%"], picks[name]["exact"]), (49, 50, 39))
            self.assertEqual((picks[name]["gained"], picks[name]["lost"]), ([308, 348, 349], []))
        self.assertEqual(check["disagree_vs_65"]["lost"], [])
        self.assertEqual(list(check["over_line"]), ["331"])
        self.assertEqual(picks["disagree"]["worse"], [341])
        self.assertEqual(len(list((ROOT / "experiments/locate-repair/evidence/fresh-repair/labels").glob("*.npz"))), 50)

    def test_time_repair_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/time-repair/check.json").read_text())
        self.assertEqual(check["seeds"], 35)
        self.assertEqual(check["reproduces_disagree_pick"], 35)
        self.assertEqual(next(iter(check["step_core_hours"])), "margin refine")
        stage = sum(check["core_hours"][k] for k in ("committee", "restarts", "boundary", "bigger", "finals", "choose"))
        self.assertGreater(check["step_core_hours"]["margin refine"] / stage, 0.85)
        for name in ("full", "cheap"):
            self.assertEqual((check[name]["under_0.01%"], check[name]["exact"], check[name]["over_line"]), (34, 28, [296]))
        self.assertEqual((check["cheap_vs_full"]["gained"], check["cheap_vs_full"]["lost"]), ([], []))
        self.assertLess(check["core_hours_full"] - check["core_hours_cheap"], 0.1)

    def test_cut_refine_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/cut-refine/check.json").read_text())
        for kind in ("columns", "loose"):
            c = check[kind]
            self.assertEqual(c["seeds"], 35)
            self.assertEqual((c["under_0.01%"], c["exact"]), (34, 28))
            self.assertEqual((c["crossed"]["gained"], c["crossed"]["lost"]), ([], []))
            self.assertLess(c["core_hours_stage"], c["core_hours_stage_73"])
        self.assertEqual((check["columns"]["same_pick"], check["columns"]["changed"]), (35, {}))
        self.assertGreater(check["columns"]["speedup_stage"], 1.3)
        self.assertGreater(check["loose"]["speedup_stage"], 1.7)
        self.assertEqual(check["loose"]["same_pick"], 32)
        self.assertTrue(all(max(v) < 1e-4 for s, v in check["loose"]["changed"].items() if s != "296"))

    def test_batched_nelder_mead_is_scipys(self):
        import numpy as np
        from scipy.optimize import minimize
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        import batched_refine as br
        rng = np.random.default_rng(3)
        A, c = rng.normal(size=(12, 3, 3)), rng.normal(size=(12, 3))
        g = lambda k, z: float(np.abs(A[k] @ (z - c[k])).sum() + 0.1 * max(z[0], 0))
        best, calls = br.nelder_mead(lambda rows, z: np.array([g(k, v) for k, v in zip(rows, z)]), 12, **br.OPTIONS)
        for k in range(12):
            r = minimize(lambda z: g(k, z), np.zeros(3), method="Nelder-Mead",
                         options={"initial_simplex": np.vstack([np.zeros(3), 0.02 * np.eye(3)]), **br.OPTIONS})
            self.assertTrue(np.array_equal(r.x, best[k]))
            self.assertEqual(r.nfev, calls[k])

    def test_batched_losses_match_the_columns_loss(self):
        import numpy as np
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        import early_values as ev
        import cut_refine as cr
        import batched_refine as br
        case = ev.rs.resize(ev.t3.corrupt(16, 264))
        px, py, _ = ev.tc.lr.split(case["practice"])
        x, y = px[:300], py[:300]
        entries = ev.tc.lr.entries(case["arrays"])
        rng = np.random.default_rng(7)
        triples = [tuple(tuple(entries[j]) for j in rng.choice(len(entries), 3, replace=False)) for _ in range(20)]
        jobs = [(t, ev.tc.wt.score(case, x, y, t, ev.tc.ct.current_values(case, t))) for t in triples]
        problem = br.setup(case, x, y, jobs)
        z = rng.normal(0, 0.05, (len(jobs), 3))
        got = br.losses(problem, np.arange(len(jobs)), z, x, (2 * y - 1).astype(np.float64))
        for k, (t, _) in enumerate(jobs):
            self.assertAlmostEqual(got[k], cr.column_loss(case, t, problem[3][k], problem[4][k], x, y)(z[k]), places=12)

    def test_time_65_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/time-65/check.json").read_text())
        o, c, l = check["original"], check["columns"], check["loose"]
        self.assertEqual(o["seeds"], [250, 264, 291, 297, 307, 320, 328, 338])
        self.assertEqual((o["same_labels"], o["same_repair"], o["changed"]), (8, 8, {}))
        self.assertGreater(o["refine_share"], 0.8)
        self.assertEqual((c["same_labels"], c["same_repair"]), (4, 7))
        self.assertEqual(list(c["changed"]), ["250"])
        self.assertEqual(c["changed"]["250"][0]["disagreement"], c["changed"]["250"][1]["disagreement"])
        self.assertGreater(c["speedup_vs_original"], 1.6)
        self.assertGreater(l["speedup_vs_original"], 2.0)
        for k in (o, c, l):
            self.assertEqual((k["under_0.01%"], k["under_0.01%_65"]), (7, 7))
        for before, after in l["changed"].values():
            self.assertEqual((before["faults_found"], before["stages_run"]), (after["faults_found"], after["stages_run"]))
            self.assertLess(after["disagreement"], 1e-4)

    def test_end_to_end_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/end-to-end/check.json").read_text())
        o, c, l = check["original"], check["columns"], check["loose"]
        self.assertEqual((o["under_0.01%"], o["exact"]), (8, 7))
        self.assertEqual(sorted(c["rerun"] + c["from_74"]), [250, 264, 291, 297, 307, 320, 328, 338])
        self.assertEqual((c["same_pick_as_73"], c["same_disagreement_as_73"], c["under_0.01%"], c["exact"]), (8, 7, 8, 7))
        self.assertEqual((l["same_pick_as_73"], l["under_0.01%"], l["exact"]), (5, 8, 6))
        for k in (c, l):
            self.assertEqual(k["crossed"], {"gained": [], "lost": []})
        self.assertGreater(c["speedup_vs_original"], 1.5)
        self.assertGreater(l["speedup_vs_original"], 1.9)

    def test_batched_refine_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/batched-refine/check.json").read_text())
        self.assertEqual(check["seeds"], 8)
        self.assertEqual((check["same_pick"], check["same_disagreement"], check["changed"]), (8, 8, {}))
        self.assertEqual((check["under_0.01%"], check["exact"]), (8, 7))
        v = check["vs_columns"]
        self.assertEqual((v["same_pick"], v["same_disagreement"], v["changed"]), (8, 8, {}))
        self.assertGreater(v["speedup_stage"], 1.3)
        self.assertGreater(check["batched_share_of_refine_calls"], 0.9)

    def test_fresh_cheap_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/fresh-cheap/check.json").read_text())
        o, c = check["original"], check["cheap"]
        self.assertEqual(o["seeds"], list(range(350, 362)))
        self.assertEqual(c["seeds"], o["seeds"])
        self.assertEqual((o["under_0.01%"], o["exact"], o["repair_65_under_0.01%"]), (12, 9, 9))
        self.assertEqual((c["under_0.01%"], c["exact"], c["repair_65_under_0.01%"]), (12, 8, 9))
        v = c["vs_original"]
        self.assertEqual((v["seeds"], v["same_pick"], v["same_disagreement"]), (12, 11, 9))
        self.assertEqual(v["crossed"], {"gained": [], "lost": []})
        self.assertEqual(sorted(v["changed"]), ["355", "356", "360"])
        self.assertGreater(v["speedup"], 2.1)
        self.assertGreater(o["refine_share"], 0.8)

    def test_fresh_50_evidence(self):
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/fresh-50/check.json").read_text())
        c, h = check["cheap"], check["held_out_72"]
        self.assertEqual(check["seeds"], list(range(400, 450)))
        self.assertEqual((c["under_0.01%"], c["under_0.1%"], c["exact"]), (48, 50, 39))
        self.assertEqual((c["repair_65_under_0.01%"], c["repair_65_exact"]), (41, 28))
        self.assertEqual(sorted(c["over_line"]), ["400", "414"])
        self.assertEqual(c["lost_from_65"], [414])
        self.assertEqual(len(c["gained_over_65"]), 8)
        self.assertEqual(sum(c["picks"].values()), 50)
        self.assertEqual((h["under_0.01%"], h["exact"]), (49, 39))
        self.assertAlmostEqual(c["core_hours"], 20.88, places=2)
        self.assertGreater(check["cost_ratio"], 1.75)

    def test_fresh_50_original_rerun(self):
        """The original repair, rerun on the two cases over the line, lands in the same place as the cheap run."""
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/fresh-50/vs-original.json").read_text())
        self.assertEqual(check["seeds"], [400, 414])
        for key in ("same_stage_65_outcome", "same_candidates", "same_pick", "same_chosen"):
            self.assertEqual(check[key], [400, 414], key)
        self.assertGreater(check["speedup"], 2.0)

    def test_keep_65_evidence(self):
        """#79: keeping #65's repair in #71's chooser, ties last, fixes 414 and takes no case over the line."""
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/keep-65/check.json").read_text())
        self.assertEqual(check["seeds"], list(range(300, 350)) + list(range(400, 450)))
        w, last, first = check["without"], check["with_last"], check["with_first"]
        self.assertEqual((w["under_0.01%"], w["exact"]), (97, 78))
        self.assertEqual((last["under_0.01%"], last["under_0.1%"], last["exact"]), (98, 100, 79))
        self.assertEqual(last["took_65"], [330, 400, 414, 426, 430])
        self.assertEqual((last["gained"], last["lost"], last["worse"]), ([414], [], [319]))
        self.assertEqual((first["exact"], len(first["took_65"]), len(first["worse"])), (78, 24, 8))
        self.assertEqual((check["repair65"]["under_0.01%"], check["probe_best"]["under_0.01%"]), (80, 98))
        self.assertEqual((check["rebuilt_65"]["same_faults_found"], check["rebuilt_65"]["within_0.001%"]), (100, 83))
        self.assertEqual(check["labels_asked"], {"without": 579, "with": 1155})
        self.assertAlmostEqual(check["core_hours"], 20.26, places=2)

    def test_saved_65_evidence(self):
        """#80: the saved #65 repair matches its record; the rebuild costs 508 its pick; two rounds trades small losses."""
        import json
        folder = ROOT / "experiments/locate-repair/evidence/saved-65"
        check = json.loads((folder / "check.json").read_text())
        self.assertEqual(check["seeds"], list(range(500, 550)))
        self.assertEqual(check["saved_65"], {"matches_recorded": 50, "under_0.01%": 42, "exact": 36})
        self.assertEqual((check["rebuilt_65"]["within_0.001%"], check["rebuilt_65"]["same_side_of_line"]), (44, 49))
        for name in ("without", "with_saved", "with_rebuilt", "two_round", "probe_best"):
            self.assertEqual((check[name]["under_0.01%"], check[name]["exact"]), (50, 39), name)
        self.assertEqual((check["with_saved"]["took_65"], check["with_saved"]["worse"]), ([531, 543], []))
        self.assertEqual(check["with_rebuilt"]["worse"], [508])
        self.assertEqual(check["two_round"]["worse"], [503, 513, 536])
        self.assertEqual(check["labels_asked"], {"without": 296, "with_saved": 449, "two_round": 704})
        self.assertAlmostEqual(check["core_hours"], 22.28, places=2)
        rescore = json.loads((folder / "rescore-79-check.json").read_text())
        self.assertEqual((rescore["without"]["under_0.01%"], rescore["without"]["exact"]), (97, 78))
        two = rescore["two_round"]
        self.assertEqual((two["under_0.01%"], two["exact"], two["lost"]), (98, 79, []))
        self.assertEqual(two["better"], [311, 400, 402, 414, 430, 449])
        self.assertEqual(two["worse"], [304, 321, 325, 339, 341, 437])
        self.assertEqual(rescore["with_last"]["worse"], [319])
        self.assertEqual(rescore["labels_asked"], {"without": 579, "with_last": 1155, "two_round": 1601})

    def test_gated_duel_evidence(self):
        """#81: a duel gated on 32 split inputs makes no case worse on 150 cases and keeps 400 and 414."""
        import json
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        import gated_duel as gd
        check = json.loads((ROOT / "experiments/locate-repair/evidence/gated-duel/check.json").read_text())
        self.assertEqual(check["rule"], {"min_split_inputs": 32, "min_margin": 1})
        tune, fresh = check["tune_79"]["gated"], check["check_80"]["gated"]
        self.assertEqual((tune["under_0.01%"], tune["exact"], tune["better"], tune["worse"]), (98, 79, [400, 402, 414], []))
        self.assertEqual((fresh["under_0.01%"], fresh["exact"], fresh["took_65"], fresh["worse"]), (50, 39, [], []))
        self.assertEqual((tune["labels_asked"], fresh["labels_asked"]), (1123, 392))
        for k in (8, 16, 32):
            self.assertEqual(check["grid_79"][f"k={k},m=1"][:2], [3, 0])
        tune_rows, check_rows = gd.load()
        self.assertEqual(gd.score(tune_rows, 32, 1), tune)
        self.assertEqual(gd.score(check_rows, 1, 1)["worse"], [503, 513, 536])

    def test_value_labels_evidence(self):
        """#82: the flat-line step puts all 150 cases under 0.01%; region labels leave 331 and 400 over it."""
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/value-labels/check.json").read_text())
        self.assertEqual(len(check["seeds"]), 150)
        self.assertEqual(sorted(check["chosen"]["over_line"]), ["331", "400"])
        self.assertEqual(len(check["flat"]), 27)
        middle, last = check["line_middle"], check["line_after_32"]
        self.assertEqual((middle["under_0.01%"], middle["gained"], middle["lost"]), (150, [331, 400], []))
        self.assertEqual((last["under_0.01%"], last["worse"]), (150, [349, 418, 428]))
        self.assertEqual(last["better"], [331, 400, 402, 443, 526, 542, 547])
        self.assertLess(check["per_case_flat"]["331"][1], 1e-4)
        region = check["region_after_32"]
        self.assertEqual((region["under_0.01%"], sorted(region["over_line"])), (58, ["331", "400"]))
        self.assertEqual(len(check["control"]["worse"]), 45)

    def test_fresh_line_evidence(self):
        """#83: on fresh seeds the gated duel and the line step change no outcome; 641 has a perfect fit no candidate tries."""
        import json
        folder = ROOT / "experiments/locate-repair/evidence/fresh-line"
        check = json.loads((folder / "check.json").read_text())
        self.assertEqual(check["seeds"], list(range(600, 650)))
        for name in ("without", "gated", "line_after_32"):
            r = check[name]
            self.assertEqual((r["under_0.01%"], r["under_0.1%"], r["exact"]), (48, 50, 38))
            self.assertEqual(sorted(r["over_line"]), ["611", "641"])
        self.assertEqual(check["repair65"]["under_0.01%"], 37)
        self.assertEqual(check["gated_took_65"], [])
        self.assertEqual(check["flat"], [604, 610, 622, 633, 644])
        self.assertEqual(check["steps"]["line_32_over_gated"], {"better": [], "worse": [644], "gained": [], "lost": []})
        self.assertEqual(check["steps"]["stage_71_over_65"]["worse"], [616, 618, 623, 626, 639])
        self.assertEqual(check["labels_asked"], {"without": 286, "gated_and_line": 734})
        truth = json.loads((folder / "truth.json").read_text())
        self.assertEqual(sorted(truth), ["611", "641"])
        self.assertEqual(truth["641"]["true"]["labels_right"], 1.0)
        self.assertLess(truth["641"]["true"]["disagreement"], 1e-4)
        self.assertLess(truth["641"]["pick"]["labels_right"], 1.0)
        self.assertGreaterEqual(truth["611"]["true"]["disagreement"], 1e-4)

    def test_keep_one_evidence(self):
        """#84: searching triples that keep one of the pick's weights fixes 641 exactly and makes no case worse."""
        import json
        folder = ROOT / "experiments/locate-repair/evidence/keep-one"
        check = json.loads((folder / "check.json").read_text())
        self.assertEqual(len(check["seeds"]), 27)
        self.assertEqual((check["pick"]["under_0.01%"], check["pick"]["exact"]), (26, 20))
        self.assertEqual(sorted(check["pick"]["over_line"]), ["641"])
        self.assertEqual((check["search"]["under_0.01%"], check["search"]["exact"]), (27, 22))
        self.assertEqual(check["search"]["over_line"], {})
        self.assertEqual(check["better"], [315, 444, 641, 642, 648])
        self.assertEqual(check["worse"], [])
        self.assertEqual((check["true_searched"], check["true_perfect"]), (27, 17))
        c641 = check["per_case"]["641"]
        self.assertEqual((c641["perfect"], c641["search"][1]), (1, 3))
        self.assertLess(c641["search"][0], 1e-4)
        gap = json.loads((folder / "gap.json").read_text())
        self.assertEqual(len(gap), 10)
        for v in gap.values():
            self.assertEqual(v["original_values_wrong"], 0)
            self.assertIn(v["fitted_wrong"], (1, 2))

    def test_width_32_repair_evidence(self):
        """#92: #90's fixed repair at width 32 puts all 12 of #60's seeds under the line, 9 exact."""
        import json
        folder = ROOT / "experiments/locate-repair/evidence/width-32-repair"
        check = json.loads((folder / "check.json").read_text())
        self.assertEqual((check["width"], check["seeds"]), (32, list(range(200, 212))))
        self.assertEqual((check["repair65"]["under_0.01%"], check["repair65"]["exact"]), (9, 5))
        self.assertEqual(sorted(check["repair65"]["over_line"]), ["202", "204", "205"])
        self.assertEqual((check["pick"]["under_0.01%"], check["pick"]["exact"]), (12, 8))
        self.assertEqual((check["result"]["under_0.01%"], check["result"]["exact"]), (12, 9))
        self.assertEqual(check["result"]["over_line"], {})
        self.assertEqual(check["result_vs_pick"]["gained_exact"], [202])
        self.assertEqual(check["result_vs_pick"]["lost_exact"], [])
        self.assertEqual(check["result_vs_pick"]["further"], [204])
        self.assertEqual((check["keep_one"], check["flat"]), ([], []))
        self.assertEqual(check["triples_fitted"], {"keep-one": 0, "keep-two": 5736})
        self.assertEqual(check["labels_asked"], {"duel": 466, "line": 0})
        rows = {r["seed"]: r for r in json.loads((folder / "width-32-repair.json").read_text())}
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["searched"] == 478 for r in rows.values()))
        short = sorted(s for s, r in rows.items() if r["every_winner"][1] < 3)
        self.assertEqual(short, [201, 209, 210])
        self.assertEqual([rows[s]["every_winner"][1] for s in short], [2, 2, 1])
        self.assertEqual(rows[210]["pick"]["faults_found"], 1)
        self.assertFalse(rows[210]["true_perfect"])
        self.assertTrue(rows[201]["true_perfect"] and rows[209]["true_perfect"])
        self.assertTrue(all(r["every_winner"][0] < 1e-4 for r in rows.values()))

    def test_tie_break_evidence(self):
        """#91: breaking the duel's ties on a fresh pool reproduces the duels but does not raise exact."""
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/tie-break/check.json").read_text())
        fresh, shaped = check["fresh"], check["shaped"]
        self.assertEqual((len(fresh["seeds"]), len(shaped["seeds"])), (50, 200))
        self.assertEqual((len(fresh["duel_reproduced"]), len(shaped["duel_reproduced"])), (48, 197))
        self.assertEqual((fresh["before"]["exact"], fresh["after"]["exact"]), (36, 35))
        self.assertEqual(fresh["after"]["under_0.01%"], 50)
        self.assertEqual((shaped["before"]["under_0.01%"], shaped["after"]["under_0.01%"]), (200, 199))
        self.assertEqual(sorted(shaped["after"]["over_line"]), ["331"])
        self.assertEqual((shaped["before"]["exact"], shaped["after"]["exact"]), (159, 159))
        self.assertEqual(shaped["moves"]["gained_exact"], [317, 325, 535])
        self.assertEqual(shaped["moves"]["lost_exact"], [311, 335, 444])
        self.assertEqual((len(fresh["tied"]), len(shaped["tied"])), (43, 170))
        self.assertEqual((fresh["labels_asked"], shaped["labels_asked"]), (127, 449))
        rows = json.loads((ROOT / "experiments/locate-repair/evidence/tie-break/fresh.json").read_text())
        self.assertEqual(sum(1 for r in rows if r["tie_break"] and not r["tie_break"]["rounds"]), 20)

    def test_fresh_end_evidence(self):
        """#90: the whole repair, fixed in advance, puts all 50 fresh cases under the line, 36 exact."""
        import json
        folder = ROOT / "experiments/locate-repair/evidence/fresh-end"
        check = json.loads((folder / "check.json").read_text())
        self.assertEqual(check["seeds"], list(range(700, 750)))
        self.assertEqual((check["pick"]["under_0.01%"], check["pick"]["exact"]), (49, 34))
        self.assertEqual(sorted(check["pick"]["over_line"]), ["744"])
        self.assertEqual((check["result"]["under_0.01%"], check["result"]["exact"]), (50, 36))
        self.assertEqual(check["result"]["over_line"], {})
        self.assertEqual(check["keep_one"], [709, 724, 744])
        self.assertEqual(check["result_vs_pick"]["gained_exact"], [706, 723, 744])
        self.assertEqual(check["result_vs_pick"]["lost_exact"], [736])
        self.assertEqual(check["result_vs_every_winner"]["further"], [719])
        self.assertEqual(check["line_applied"], [705, 717, 722, 738, 742])
        self.assertEqual(check["labels_asked"], {"first_part": 413, "duel": 2602, "line": 160})
        rows = {r["seed"]: r for r in json.loads((folder / "fresh-end.json").read_text())}
        self.assertEqual(len(rows), 50)
        self.assertEqual(sum(bool(r["perfect"]) for r in rows.values()), 48)
        self.assertTrue(all(r["true_perfect"] for r in rows.values() if r["perfect"]))
        self.assertEqual((rows[744]["result"][1], rows[744]["searched"]), (3, 9481))
        for r in rows.values():  # every result short of exact is a tie at the top of the duel
            if r["result"][1] != 3:
                correct = r["duel"]["correct"]
                self.assertGreater(sum(v == max(correct.values()) for v in correct.values()), 1)
        self.assertEqual(len(list((folder / "labels").glob("case-*.npz"))), 50)

    def test_keep_two_evidence(self):
        """#89: the keep-two search with the refit duel puts all 200 under the line without a hand, and reproduces #88."""
        import json
        folder = ROOT / "experiments/locate-repair/evidence/keep-two"
        check = json.loads((folder / "check.json").read_text())
        self.assertEqual(len(check["seeds"]), 200)
        self.assertEqual(check["keep_two_cases"], 173)
        self.assertEqual((check["result"]["under_0.01%"], check["result"]["exact"]), (200, 159))
        self.assertEqual(check["result"]["over_line"], {})
        self.assertEqual(sorted(check["after_86"]["over_line"]), ["611"])
        self.assertEqual((check["vs_86"]["gained"], check["vs_86"]["lost"]), ([611], []))
        self.assertEqual((len(check["vs_86"]["gained_exact"]), len(check["vs_86"]["lost_exact"])), (10, 9))
        self.assertEqual(len(check["flat"]), 38)
        self.assertEqual(check["labels_asked"], {"duel": 7702, "line": 1216})
        rows = {r["seed"]: r for r in json.loads((folder / "keep-two.json").read_text())}
        self.assertEqual(rows[611]["search"], "keep-two")
        self.assertEqual(rows[611]["searched"], 238)
        self.assertEqual(len(rows[611]["perfect"]), 2)
        self.assertEqual(rows[611]["result"][1], 3)
        duel = {r["seed"]: r for r in json.loads(
            (ROOT / "experiments/locate-repair/evidence/refit-duel/refit-duel.json").read_text())}
        for s, r in duel.items():
            self.assertEqual((rows[s]["duel"]["winner"], rows[s]["chosen"]["disagreement"]), (r["winner"], r["result"][0]))

    def test_refit_duel_evidence(self):
        """#88: refitting the candidates on the labels where they split fixes 611 exactly and loses no case."""
        import json
        check = json.loads((ROOT / "experiments/locate-repair/evidence/refit-duel/check.json").read_text())
        self.assertEqual(len(check["seeds"]), 28)
        self.assertEqual((check["before"]["under_0.01%"], check["before"]["exact"]), (27, 22))
        self.assertEqual((check["after"]["under_0.01%"], check["after"]["exact"]), (28, 24))
        self.assertEqual((check["gained"], check["lost"]), ([611], []))
        c611 = check["per_case"]["611"]
        self.assertEqual(c611["after"][1], 3)
        self.assertLess(c611["after"][0], 1e-5)
        self.assertEqual(check["labels_asked"], 966)

    def test_ungated_search_evidence(self):
        """#87: searching 611 finds the true triple as a perfect fit, but its fit is over the line and the pick stays."""
        import json
        row = json.loads((ROOT / "experiments/locate-repair/evidence/ungated-search/case-611.json").read_text())
        self.assertEqual(row["pick_wrong"], 0)
        self.assertEqual((row["searched"], row["total"]), (9481, 9481))
        self.assertEqual(row["perfect_count"], 2)
        self.assertTrue(row["true_perfect"])
        true = [p for p in row["perfect"] if p["faults_found"] == 3]
        self.assertEqual(len(true), 1)
        self.assertGreater(true[0]["disagreement"], 1e-4)
        self.assertEqual(row["chooser"]["pick"], "pick")
        self.assertFalse(row["result"]["under_line"])

    def test_flat_stack_evidence(self):
        """#86: the flat-line step on top of the search puts 199 of 200 under the line and reproduces #82 and #83."""
        import json
        folder = ROOT / "experiments/locate-repair/evidence/flat-stack"
        check = json.loads((folder / "check.json").read_text())
        self.assertEqual(len(check["seeds"]), 200)
        self.assertEqual(sorted(check["before"]["over_line"]), ["331", "400", "611"])
        self.assertEqual((check["after"]["under_0.01%"], check["after"]["exact"]), (199, 158))
        self.assertEqual(sorted(check["after"]["over_line"]), ["611"])
        self.assertEqual((check["gained"], check["lost"]), ([331, 400], []))
        self.assertEqual(len(check["flat"]), 33)
        self.assertEqual(check["earlier_check"]["same_network"], check["earlier_check"]["agree"])
        truth = json.loads((folder / "truth.json").read_text())["611"]
        self.assertEqual(truth["original_wrong"], 0)
        self.assertLessEqual(truth["singular_values"][-1], 1e-6 * truth["singular_values"][0])
        self.assertTrue(all(v < 1e-4 for v in truth["disagreement"].values()))

    def test_perfect_fit_evidence(self):
        """#85: the margin step makes the true triple a perfect fit in all 27 searched cases and changes no outcome."""
        import json
        folder = ROOT / "experiments/locate-repair/evidence/perfect-fit"
        check = json.loads((folder / "check.json").read_text())
        self.assertEqual(len(check["seeds"]), 27)
        self.assertEqual((check["true_searched"], check["true_perfect"]), (27, 27))
        self.assertEqual(check["vs_84"]["true_perfect_84"], 17)
        self.assertEqual(check["vs_84"]["fewer_perfect"], [])
        self.assertEqual(check["vs_84"]["further"], [])
        self.assertEqual(check["vs_84"]["closer"], [348, 630, 647])
        self.assertEqual(check["better"], [315, 348, 444, 630, 641, 642, 647, 648])
        self.assertEqual(check["worse"], [])
        self.assertEqual((check["search"]["under_0.01%"], check["search"]["exact"]), (27, 22))
        true = json.loads((folder / "true.json").read_text())
        self.assertEqual(len(true["true"]), 10)
        for v in true["true"].values():
            self.assertIn(v["before"]["wrong"], (1, 2))
            self.assertEqual(v["after"]["wrong"], 0)
        moved = true["pick_moved"]
        self.assertEqual(sum(v["after"] > v["before"] for v in moved.values()), 21)

    def test_flat_line_segment_on_seed_331(self):
        """#82: the least-constrained direction at 331's pick is flat and its segment's middle fits every label."""
        import numpy as np
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        import value_labels as vl
        from rebuild_committee import labels
        chosen = vl.cases()[331]
        case = vl.ev.rs.resize(vl.ev.t3.corrupt(16, 331))
        x, y = labels(case, 331, vl.SOURCES[300])
        triple = tuple(tuple(e) for e in chosen["winner"])
        scale = np.array([case["scale"][n] for n, _ in triple])
        z = np.array([v for _, _, v in chosen["edits"]]) / scale
        d, a, b, sv = vl.segment(case, x, y, triple, z, scale)
        self.assertLessEqual(sv[-1], vl.FLAT * sv[0])
        self.assertGreater(b - a, 1.0)
        mid = (z + (a + b) / 2 * d) * scale
        self.assertEqual(int((vl.wt.signed_margins(case, triple, mid, x, y) <= 0).sum()), 0)

    def test_cut_refine_losses_match_the_original(self):
        import numpy as np
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        import early_values as ev
        import cut_refine as cr
        case = ev.rs.resize(ev.t3.corrupt(16, 264))
        px, py, _ = ev.tc.lr.split(case["practice"])
        x, y = px[:200], py[:200]
        entries = ev.tc.lr.entries(case["arrays"])
        rng = np.random.default_rng(5)
        names = set()
        for _ in range(6):
            t = tuple(tuple(entries[j]) for j in rng.choice(len(entries), 3, replace=False))
            names |= {n for n, _ in t}
            start = ev.tc.wt.score(case, x, y, t, ev.tc.ct.current_values(case, t))
            scale = np.array([case["scale"][n] for n, _ in t])
            s0 = np.array([v for _, _, v in start[1]])
            exact, columns = cr.margin_loss(case, t, s0, scale, x, y), cr.column_loss(case, t, s0, scale, x, y)
            for z in rng.normal(0, 0.05, (4, 3)):
                want = float(np.maximum(ev.tc.wt.EPS - ev.tc.wt.signed_margins(case, t, s0 + z * scale, x, y), 0).mean())
                self.assertEqual(exact(z), want)
                self.assertAlmostEqual(columns(z), want, places=12)
            self.assertEqual(cr.refine(case, x, y, t, start, "exact"), cr.ORIGINAL(case, x, y, t, start))
        self.assertGreaterEqual(len(names), 3)

    def test_time_repair_clock_times_steps_exclusively(self):
        import time as _time
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from time_repair import Clock
        clock = Clock()
        inner = clock.tap(lambda: _time.sleep(0.05), "inner")
        probe = clock.tap(lambda: _time.sleep(0.05), "probe")
        outer = clock.tap(lambda: (_time.sleep(0.05), inner(), probe()), "outer")
        with clock.part("committee"):
            outer()
        self.assertAlmostEqual(clock.steps["outer"], 0.05, delta=0.03)
        self.assertAlmostEqual(clock.steps["inner"], 0.05, delta=0.03)
        self.assertAlmostEqual(clock.probe, 0.05, delta=0.03)
        self.assertAlmostEqual(clock.seconds["committee"], 0.10, delta=0.04)  # probe time left out of the part
        clock.reset()
        self.assertEqual((dict(clock.seconds), dict(clock.steps)), ({}, {}))

    def test_time_repair_score_counts_parts_and_cheap_order(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from time_repair import PARTS, score
        cand = lambda d, fs=0.0, faults=3: {"disagreement": d, "final_seconds": fs, "faults_found": faults}
        secs = lambda **kw: {k: kw.get(k, 0.0) for k in PARTS}
        rows = {1: {"gate": False, "seconds_65": 3600.0, "probe_seconds": 36.0, "steps": {"margin refine": 1800.0},
                    "seconds": secs(committee=3600.0, restarts=360.0, finals=720.0),
                    "candidates": {"pick68": cand(5e-5), "restarts": cand(2e-4, 360.0), "boundary": cand(5e-5)},
                    "full": {"pick": "restarts", "asked": 4}, "cheap": {"pick": "pick68", "asked": 2}},
                2: {"gate": True, "seconds_65": 0.0, "probe_seconds": 0.0, "steps": {"margin refine": 1800.0},
                    "seconds": secs(committee=3600.0, bigger=3600.0),
                    "candidates": {"pick68": cand(2e-4), "restarts": cand(2e-4), "boundary": cand(2e-4, faults=2),
                                   "bigger": cand(5e-5)},
                    "full": {"pick": "bigger", "asked": 3}, "cheap": {"pick": "bigger", "asked": 3}}}
        out = score(rows, {1: {"choosers": {"disagree": {"pick": "restarts"}}},
                           2: {"choosers": {"disagree": {"pick": "pick68"}}}})
        self.assertEqual(out["core_hours"]["committee"], 2.0)
        self.assertEqual(out["step_core_hours"], {"margin refine": 1.0})
        self.assertAlmostEqual(out["core_hours_full"], 1 + 2 + 0.1 + 0.2 + 1)
        self.assertAlmostEqual(out["core_hours_cheap"], out["core_hours_full"] - 0.2)  # case 1's restarts and its final
        self.assertEqual(out["gate"], [2])
        self.assertEqual((out["full"]["under_0.01%"], out["cheap"]["under_0.01%"]), (1, 2))
        self.assertEqual(out["cheap_vs_full"], {"gained": [1], "lost": [], "different_pick": [1]})
        self.assertEqual(out["reproduces_disagree_pick"], 1)

    def test_unseen_score_counts_stages_and_lines(self):
        sys.path.insert(0, str(ROOT / "experiments/locate-repair"))
        from unseen_check import score
        row = lambda seed, d, acc=1.0, k=1, sec=10.0: {"seed": seed, "stages_run": k, "seconds": sec,
            "committee_has_true_triple": d == 0,
            "switch": [{"disagreement": d, "faults_found": 3 if d == 0 else 2, "found_visible": d == 0,
                        "practice_accuracy": acc}]}
        final = {1: row(1, 0.0), 2: row(2, 5e-4, k=2, sec=30.0), 3: row(3, 2e-3, acc=0.9, k=3, sec=60.0)}
        plain = {1: row(1, 0.0), 2: row(2, 3e-3, acc=0.9), 3: row(3, 2e-3, acc=0.9)}
        out = score(final, plain)
        self.assertEqual(out["stages_run"], {"1": 1, "2": 1, "3": 1})
        self.assertEqual(out["reran_with_bigger_committee"], [2, 3])
        self.assertEqual(out["line_0.001"], {"gained": [2], "lost": []})
        self.assertEqual(out["failures_left"], [3])
        self.assertEqual(out["still_misses_a_label"], [3])
        self.assertAlmostEqual(out["cost_ratio"], 100 / 30)
        with self.assertRaises(ValueError):
            score(final, {1: plain[1]})

    def test_two_step_reruns_while_the_repair_misses_a_label(self):
        import tempfile
        from pathlib import Path
        calls = []
        accuracies = iter([0.9, 0.95, 1.0])
        real = ev.analyse

        def analyse(case, early, **options):
            calls.append(options)
            return {"switch": [{"practice_accuracy": next(accuracies), "disagreement": 0.0, "faults_found": 3}],
                    "committee_has_true_triple": True, "seconds": 2.0}

        ev.analyse = analyse
        try:
            with tempfile.TemporaryDirectory() as d:
                saved = lambda k: Path(d) / f"stage{k}.json"
                row = ev.two_step(self.case, saved=saved)
                self.assertEqual(calls, [{}, {"committee_size": 1000}, {"pairs_kept": 1000}])
                self.assertEqual((row["stages_run"], row["seconds"]), (3, 6.0))
                calls.clear()
                again = ev.two_step(self.case, saved=saved)
                self.assertEqual(calls, [])
                self.assertEqual(again["two_step"], row["two_step"])
            accuracies = iter([1.0])
            self.assertEqual(ev.two_step(self.case)["stages_run"], 1)
        finally:
            ev.analyse = real

    def test_final_choice_keeps_the_last_stage_only_on_a_perfect_fit(self):
        real = ev.analyse
        plan = []

        def analyse(case, early, **options):
            acc, d = plan.pop(0)
            return {"switch": [{"practice_accuracy": acc, "disagreement": d, "faults_found": 1}],
                    "committee_has_true_triple": False, "seconds": 1.0}

        ev.analyse = analyse
        try:
            plan[:] = [(0.9, 5e-4), (0.95, 8e-5), (0.97, 3e-4)]
            row = ev.two_step(self.case, final_choice="last-if-perfect")
            self.assertEqual((row["kept_stage"], row["switch"][-1]["disagreement"]), (1, 8e-5))
            self.assertEqual((row["stages_run"], row["seconds"], len(row["two_step"])), (3, 3.0, 3))
            plan[:] = [(0.9, 5e-4), (0.95, 8e-5), (1.0, 2e-5)]
            self.assertEqual(ev.two_step(self.case, final_choice="last-if-perfect")["kept_stage"], 2)
            plan[:] = [(0.9, 5e-4), (0.95, 8e-5), (0.97, 3e-4)]
            plain = ev.two_step(self.case)
            self.assertEqual(plain["switch"][-1]["disagreement"], 3e-4)
            self.assertNotIn("kept_stage", plain)
            self.assertEqual(ev.kept_stage([{"switch": [{"practice_accuracy": 0.9}]}], "last-if-perfect"), 0)
        finally:
            ev.analyse = real

    def test_tie_break_starts_at_the_given_stage(self):
        calls = []
        real = ev.analyse

        def analyse(case, early, **options):
            calls.append(options)
            return {"switch": [{"practice_accuracy": 0.9, "disagreement": 0.0, "faults_found": 3}],
                    "committee_has_true_triple": True, "seconds": 1.0}

        ev.analyse = analyse
        try:
            row = ev.two_step(self.case, tie_break="largest-edit", tie_break_from=2)
            self.assertEqual(calls, [{}, {"committee_size": 1000}, {"pairs_kept": 1000, "tie_break": "largest-edit"}])
            self.assertEqual(row["tie_break_from"], 2)
            calls.clear()
            self.assertNotIn("tie_break_from", ev.two_step(self.case, tie_break="largest-edit"))
            self.assertEqual(calls[0], {"tie_break": "largest-edit"})
        finally:
            ev.analyse = real

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
