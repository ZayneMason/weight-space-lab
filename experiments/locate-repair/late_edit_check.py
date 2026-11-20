"""Score the largest-edit tie-break used only in the 1,000-pair rerun (roadmap #59).

Reads evidence/late-edit/late-edit.json (``early_values.py --early 2 --two-step --tie-break largest-edit
--tie-break-from 2``) and compares each case with the coarse-order two-step rule: #53's arm for seeds 52 to 96
(built from #50's and #52's runs) and #54's run for the unseen seeds. Writes evidence/late-edit/check.json.
"""
import json

from big_committee_check import two_step
from wide_compare import EVIDENCE, cases, last

LINES = (1e-3, 1e-4)
OUT = EVIDENCE / "late-edit"
FIVE = (55, 63, 76, 93, 96)  # #57's five cases the new order kept


def compare(coarse, late):
    seeds = sorted(late)
    missing = [s for s in seeds if s not in coarse]
    if missing:
        raise ValueError(f"no coarse-order row for seeds {missing}")
    d = lambda arm, s: last(arm[s])["disagreement"]
    found = lambda arm, s: last(arm[s])["faults_found"]
    reached = [s for s in seeds if late[s]["stages_run"] == 3]
    out = {"seeds": len(seeds), "reached_last_stage": reached,
           "stopped_earlier": [s for s in seeds if late[s]["stages_run"] < 3]}
    out["unchanged_before_last_stage"] = all(d(late, s) == d(coarse, s) and found(late, s) == found(coarse, s)
                                             for s in out["stopped_earlier"])
    for line in LINES:
        out[f"line_{line}"] = {"gained": [s for s in seeds if d(late, s) < line <= d(coarse, s)],
                               "lost": [s for s in seeds if d(coarse, s) < line <= d(late, s)]}
    out["more_true_weights"] = [s for s in seeds if found(late, s) > found(coarse, s)]
    out["fewer_true_weights"] = [s for s in seeds if found(late, s) < found(coarse, s)]
    out["closer"] = [s for s in seeds if d(late, s) < d(coarse, s)]
    out["further"] = [s for s in seeds if d(late, s) > d(coarse, s)]
    out["exact"] = {"late": [s for s in seeds if found(late, s) == 3], "coarse": [s for s in seeds if found(coarse, s) == 3]}
    out["five_from_57"] = {str(s): {"stages_run": late[s]["stages_run"], "faults_found": found(late, s)}
                           for s in FIVE if s in late}
    out["true_triple_in_committee"] = {"late": [s for s in seeds if late[s].get("committee_has_true_triple")],
                                       "coarse": [s for s in seeds if coarse[s].get("committee_has_true_triple")]}
    stage_seconds = lambda arm, s: arm[s]["two_step"][-1]["seconds"]
    if reached:
        out["last_stage_cpu_hours"] = {"late": sum(stage_seconds(late, s) for s in reached) / 3600,
                                       "coarse": sum(stage_seconds(coarse, s) for s in reached) / 3600}
    out["per_seed"] = {str(s): {"coarse": [found(coarse, s), d(coarse, s)], "late": [found(late, s), d(late, s)]}
                       for s in seeds}
    return out


def coarse_arm():
    narrow = cases(EVIDENCE / "adaptive-widen/adaptive-widen.json")
    arm, _, _ = two_step(narrow, cases(EVIDENCE / "big-committee/big-committee.json"),
                         cases(EVIDENCE / "end-flag/end-flag.json"))
    return {**arm, **cases(EVIDENCE / "two-step-unseen/two-step-unseen.json")}


def main():
    result = compare(coarse_arm(), cases(OUT / "late-edit.json"))
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
