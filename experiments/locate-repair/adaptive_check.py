"""Check #48's adaptive widening flag on 50 fresh three-fault cases (roadmap #49).

Reads only committed evidence and writes evidence/adaptive-widen/check.json. Two flags are scored:

- start flag (#48, built into early_values.py --widen-to): before any targeted label, no member of the
  200-pair committee fits every practice label;
- end flag (found here): after all 64 labels, the chosen repair still misses at least one label.
"""
from pathlib import Path
import json

from wide_compare import EVIDENCE, cases, last, stats

LINE = 1e-3


def start_flag(case):
    return case.get("first_best_practice_accuracy", case["switch"][0]["practice_accuracy"]) < 1


def end_flag(case):
    return last(case)["practice_accuracy"] < 1


def failures(arm):
    return [s for s in sorted(arm) if last(arm[s])["disagreement"] >= LINE]


def flag_report(arm, flag):
    flagged = [s for s in sorted(arm) if flag(arm[s])]
    fails = failures(arm)
    return {"flagged": flagged, "failures": fails,
            "failures_missed": [s for s in fails if s not in flagged],
            "flagged_but_fine": [s for s in flagged if s not in fails]}


def check(old_narrow, old_wide, fresh):
    out = {"fresh": {**stats(fresh, sorted(fresh)), "widened": [s for s in sorted(fresh) if fresh[s]["widened"]]}}
    out["fresh"]["failures"] = [{"seed": s, "disagreement": last(fresh[s])["disagreement"],
                                 "true_triple_in_committee": fresh[s]["committee_has_true_triple"],
                                 "first_best_practice_accuracy": fresh[s]["first_best_practice_accuracy"],
                                 "final_practice_accuracy": last(fresh[s])["practice_accuracy"]}
                                for s in failures(fresh)]
    out["lacks_true_triple"] = [s for s in sorted(fresh) if not fresh[s]["committee_has_true_triple"]]
    out["start_flag"] = {"seeds_0_49": flag_report(old_narrow, start_flag), "seeds_50_99": flag_report(fresh, start_flag)}
    out["end_flag"] = {"seeds_0_49": flag_report(old_narrow, end_flag), "seeds_50_99": flag_report(fresh, end_flag)}
    flagged = out["end_flag"]["seeds_0_49"]["flagged"]
    adaptive = {s: old_wide[s] if s in flagged else old_narrow[s] for s in old_narrow}
    out["end_flag_adaptive_seeds_0_49"] = {**stats(adaptive, sorted(adaptive)),
                                           "cpu_hours": (sum(old_narrow[s]["seconds"] for s in old_narrow)
                                                         + sum(old_wide[s]["seconds"] for s in flagged)) / 3600}
    del out["end_flag_adaptive_seeds_0_49"]["median_seconds"]
    return out


def main():
    old_narrow = {**cases(EVIDENCE / "switch-point/early-2.json"), **cases(EVIDENCE / "fresh-switch/fresh-early-2.json")}
    old_wide = cases(EVIDENCE / "wide-screen/wide-screen.json")
    fresh = cases(EVIDENCE / "adaptive-widen/adaptive-widen.json")
    result = check(old_narrow, old_wide, fresh)
    (EVIDENCE / "adaptive-widen/check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
