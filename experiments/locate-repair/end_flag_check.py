"""Score the end-flag widening rule on all 100 three-fault cases (roadmap #50).

Reads only committed evidence and writes evidence/end-flag/check.json. The rule: run every case with
200 pairs; if the finished repair still misses at least one of its 64 labels, rerun that case with
1,000 pairs and keep the rerun. Seeds 0 to 49 use #48's 1,000-pair run of every case; seeds 50 to 99
use the 16 reruns made here.
"""
import json

from adaptive_check import LINE, end_flag
from wide_compare import EVIDENCE, cases, last, stats


def apply_rule(narrow, wide):
    flagged = [s for s in sorted(narrow) if end_flag(narrow[s])]
    missing = [s for s in flagged if s not in wide]
    if missing:
        raise ValueError(f"no wide rerun for flagged seeds {missing}")
    arm = {s: wide[s] if s in flagged else narrow[s] for s in narrow}
    summary = stats(arm, sorted(arm))
    del summary["median_seconds"]
    summary["cpu_hours"] = (sum(narrow[s]["seconds"] for s in narrow) + sum(wide[s]["seconds"] for s in flagged)) / 3600
    d = lambda a, s: last(a[s])["disagreement"]
    changes = {"flagged": flagged,
               "fixed": [s for s in flagged if d(wide, s) < LINE <= d(narrow, s)],
               "still_failing": [s for s in flagged if d(wide, s) >= LINE],
               "broken": [s for s in flagged if d(narrow, s) < LINE <= d(wide, s)],
               "closer": [s for s in flagged if d(wide, s) < d(narrow, s)],
               "further": [s for s in flagged if d(wide, s) > d(narrow, s)],
               "committee_gained": [s for s in flagged if wide[s]["committee_has_true_triple"] and not narrow[s]["committee_has_true_triple"]],
               "rerun_still_misses_a_label": [s for s in flagged if end_flag(wide[s])]}
    rows = [{"seed": s, "pairs_200": d(narrow, s), "pairs_1000": d(wide, s),
             "true_triple_200": narrow[s]["committee_has_true_triple"], "true_triple_1000": wide[s]["committee_has_true_triple"],
             "exact_1000": last(wide[s])["faults_found"] == 3, "final_practice_accuracy_1000": last(wide[s])["practice_accuracy"]}
            for s in flagged]
    return arm, summary, changes, rows


def check(old_narrow, old_wide, fresh_narrow, fresh_wide):
    out = {}
    arms = {}
    for name, narrow, wide in (("seeds_0_49", old_narrow, old_wide), ("seeds_50_99", fresh_narrow, fresh_wide)):
        arm, summary, changes, rows = apply_rule(narrow, wide)
        arms.update(arm)
        out[name] = {"pairs_200": stats(narrow, sorted(narrow)), "end_flag_rule": summary, **changes, "flagged_cases": rows}
        del out[name]["pairs_200"]["median_seconds"]
    total = stats(arms, sorted(arms))
    del total["median_seconds"]
    total["cpu_hours"] = out["seeds_0_49"]["end_flag_rule"]["cpu_hours"] + out["seeds_50_99"]["end_flag_rule"]["cpu_hours"]
    out["all_100"] = total
    out["failures_left"] = [s for s in sorted(arms) if last(arms[s])["disagreement"] >= LINE]
    return out


def main():
    old_narrow = {**cases(EVIDENCE / "switch-point/early-2.json"), **cases(EVIDENCE / "fresh-switch/fresh-early-2.json")}
    old_wide = cases(EVIDENCE / "wide-screen/wide-screen.json")
    fresh_narrow = cases(EVIDENCE / "adaptive-widen/adaptive-widen.json")
    fresh_wide = cases(EVIDENCE / "end-flag/end-flag.json")
    result = check(old_narrow, old_wide, fresh_narrow, fresh_wide)
    (EVIDENCE / "end-flag/check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items()}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
