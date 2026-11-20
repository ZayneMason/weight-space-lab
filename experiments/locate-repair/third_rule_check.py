"""Score the third-stage rule on 50 fresh width-16 cases (roadmap #62).

#61 picked "keep the third stage only when it fits every practice label" after seeing seeds 100 to 149. This run
used `early_values.py --two-step --final-choice last-if-perfect` on seeds 150 to 199, which keeps every stage's
result in `two_step`, so both final choices are scored from the same run. Also pools the fresh cases with #54's
seeds 100 to 149. Writes evidence/third-stage-rule/check.json.
"""
import json

from final_stage_choice import compare, current, third_only_if_perfect
from wide_compare import EVIDENCE, cases

OUT = EVIDENCE / "third-stage-rule"


def third_stage_moves(rows):
    """For each case that ran a third stage without a perfect fit: was the third stage closer than the second?"""
    out = {}
    for s, row in sorted(rows.items()):
        st = row["two_step"]
        if len(st) == 3 and st[2]["practice_accuracy"] < 1:
            out[str(s)] = "closer" if st[2]["disagreement"] < st[1]["disagreement"] else "further"
    return out


def misses(rows):
    """Under the current rule: cases at or over 0.01%, and which of them edit exactly the three broken weights."""
    end = {s: r["two_step"][-1] for s, r in rows.items()}
    over = sorted(s for s, e in end.items() if e["disagreement"] >= 1e-4)
    return {"over_0.01%": len(over), "exact_over_0.01%": [s for s in over if end[s]["faults_found"] == 3]}


def check(fresh, seen):
    kept_matches = all(r["kept_stage"] == third_only_if_perfect(r["two_step"]) for r in fresh.values())
    result = compare({"fresh": fresh, "pooled": {**seen, **fresh}})
    for arm in result.values():
        arm["rules"] = {k: v for k, v in arm["rules"].items() if k in (current.__name__, third_only_if_perfect.__name__,
                                                                      "no_third_stage", "oracle")}
    result["stages_run"] = {str(k): sum(len(r["two_step"]) == k for r in fresh.values()) for k in (1, 2, 3)}
    result["kept_stage_matches_rescore"] = kept_matches
    result["imperfect_third_stage"] = {"fresh": third_stage_moves(fresh), "seen": third_stage_moves(seen)}
    result["pooled_misses"] = misses({**seen, **fresh})
    return result


def main():
    fresh = cases(OUT / "third-stage-rule.json")
    seen = cases(EVIDENCE / "two-step-unseen/two-step-unseen.json")
    result = check(fresh, seen)
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print("stages run", result["stages_run"], "kept matches rescore", result["kept_stage_matches_rescore"])
    for name in ("fresh", "pooled"):
        arm = result[name]
        print(name, arm["seeds"], "third stage:", arm["reached_third_stage"])
        for r, v in arm["rules"].items():
            print(f"  {r:24s} {v['under_0.1%']:3d} {v['under_0.01%']:3d} {v['exact']:3d} worst {v['worst']:.6f} "
                  f"cpu {v['cpu_hours']:.1f} better {v['better']} worse {v['worse']}")
    print("pooled misses", result["pooled_misses"])
    print("imperfect third stages", result["imperfect_third_stage"])


if __name__ == "__main__":
    main()
