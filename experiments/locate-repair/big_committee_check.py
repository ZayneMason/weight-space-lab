"""Score a committee of 1,000 triples on the 16 fresh end-flag cases (roadmap #52).

Reads only committed evidence and writes evidence/big-committee/check.json. On seeds 50 to 99, three rules
are scored against 200 pairs and 200 triples everywhere: when the finished repair misses a label,
- wider screen (#50): rerun with 1,000 pairs;
- larger committee (this run): rerun keeping 1,000 triples from 200 pairs;
- both in turn: rerun with the larger committee, and if that repair still misses a label, use #50's
  1,000-pair rerun.
"""
import json

from adaptive_check import LINE, end_flag
from end_flag_check import apply_rule
from wide_compare import EVIDENCE, cases, last, stats


def two_step(narrow, big, wide):
    first = [s for s in sorted(narrow) if end_flag(narrow[s])]
    second = [s for s in first if end_flag(big[s])]
    arm = {s: (wide[s] if s in second else big[s]) if s in first else narrow[s] for s in narrow}
    summary = stats(arm, sorted(arm))
    del summary["median_seconds"]
    summary["cpu_hours"] = (sum(narrow[s]["seconds"] for s in narrow) + sum(big[s]["seconds"] for s in first)
                            + sum(wide[s]["seconds"] for s in second)) / 3600
    return arm, summary, {"flagged": first, "flagged_again": second,
                          "failures_left": [s for s in sorted(arm) if last(arm[s])["disagreement"] >= LINE]}


def check(narrow, wide, big):
    plain = stats(narrow, sorted(narrow))
    del plain["median_seconds"]
    out = {"pairs_200": plain}
    for name, other in (("wider_screen", wide), ("larger_committee", big)):
        _, summary, changes, rows = apply_rule(narrow, other)
        out[name] = {**summary, **changes, "failures_left": [s for s in sorted(narrow)
                     if last((other if s in changes["flagged"] else narrow)[s])["disagreement"] >= LINE],
                     "median_seconds_flagged": sorted(other[s]["seconds"] for s in changes["flagged"])[len(changes["flagged"]) // 2],
                     "flagged_cases": [{k.replace("1000", "rerun"): v for k, v in r.items()} for r in rows]}
    _, summary, changes = two_step(narrow, big, wide)
    out["both_in_turn"] = {**summary, **changes}
    return out


def main():
    narrow = cases(EVIDENCE / "adaptive-widen/adaptive-widen.json")
    wide = cases(EVIDENCE / "end-flag/end-flag.json")
    big = cases(EVIDENCE / "big-committee/big-committee.json")
    result = check(narrow, wide, big)
    (EVIDENCE / "big-committee/check.json").write_text(json.dumps(result, indent=1) + "\n")
    for k, v in result.items():
        print(k, {a: b for a, b in v.items() if a != "flagged_cases"})


if __name__ == "__main__":
    main()
