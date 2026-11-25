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
