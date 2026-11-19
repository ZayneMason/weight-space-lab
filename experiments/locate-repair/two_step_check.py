"""Score the two-step rule on all 100 three-fault cases (roadmap #53).

Reads only committed evidence and writes evidence/two-step/check.json. The rule: run every case with 200
pairs and 200 triples; if the finished repair misses a label, rerun it keeping 1,000 triples; if that repair
still misses a label, use the 1,000-pair rerun. Seeds 0 to 49 use #48's 1,000-pair runs and the 11
larger-committee runs made here; seeds 50 to 99 use #50's and #52's runs.
"""
import json

from big_committee_check import two_step
from end_flag_check import apply_rule
from wide_compare import EVIDENCE, cases, last, stats

LINE = 1e-3


def score(narrow, big, wide):
    plain = stats(narrow, sorted(narrow))
    del plain["median_seconds"]
    _, committee_only, changes, _ = apply_rule(narrow, big)
    arm, both, steps = two_step(narrow, big, wide)
    return arm, {"pairs_200": plain,
                 "larger_committee": {**committee_only, "fixed": changes["fixed"], "broken": changes["broken"],
                                      "failures_left": [s for s in sorted(narrow) if last((big if s in changes["flagged"]
                                                        else narrow)[s])["disagreement"] >= LINE]},
                 "both_in_turn": {**both, **steps}}


def check(sets):
    out, arms = {}, {}
    for name, (narrow, big, wide) in sets.items():
        arm, out[name] = score(narrow, big, wide)
        arms.update(arm)
    total = stats(arms, sorted(arms))
    del total["median_seconds"]
    total["cpu_hours"] = sum(out[n]["both_in_turn"]["cpu_hours"] for n in sets)
    total["plain_cpu_hours"] = sum(out[n]["pairs_200"]["cpu_hours"] for n in sets)
    total["failures_left"] = [s for s in sorted(arms) if last(arms[s])["disagreement"] >= LINE]
    out["all_100"] = total
    return out


def main():
    sets = {"seeds_0_49": ({**cases(EVIDENCE / "switch-point/early-2.json"), **cases(EVIDENCE / "fresh-switch/fresh-early-2.json")},
                           cases(EVIDENCE / "two-step/old-big.json"), cases(EVIDENCE / "wide-screen/wide-screen.json")),
            "seeds_50_99": (cases(EVIDENCE / "adaptive-widen/adaptive-widen.json"),
                            cases(EVIDENCE / "big-committee/big-committee.json"), cases(EVIDENCE / "end-flag/end-flag.json"))}
    result = check(sets)
    (EVIDENCE / "two-step/check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
