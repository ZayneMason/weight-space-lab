"""Score a third rerun with 1,000 pairs and 1,000 triples on the cases the two-step rule leaves flagged (roadmap #56).

Reads evidence/both-wide/both-wide.json and the #54 evidence, writes evidence/both-wide/check.json.
"""
import json

from wide_compare import EVIDENCE, cases, last, stats

LINES = (1e-3, 1e-4)


def score(final, both):
    """Compare the two-step finals with a third step that reruns the still-flagged cases with both settings."""
    flagged = sorted(s for s, c in final.items() if last(c)["practice_accuracy"] < 1)
    missing = [s for s in flagged if s not in both]
    if missing:
        raise ValueError(f"no both-wide run for flagged seeds {missing}")
    seeds = sorted(final)
    three = {s: both[s] if s in flagged else final[s] for s in seeds}
    cost = {s: {"seconds": both[s]["seconds"], "two_step_seconds": final[s]["seconds"]} for s in flagged}
    # the third step reruns from scratch, so its cost is added to the two-step cost
    for s in flagged:
        three[s] = dict(both[s], seconds=final[s]["seconds"] + both[s]["seconds"])
    d = lambda arm, s: last(arm[s])["disagreement"]
    per_case = [{"seed": s, "two_step": d(final, s), "both_wide": d(both, s),
                 "faults_found": last(both[s])["faults_found"], "true_triple_in_committee": bool(both[s].get("committee_has_true_triple")),
                 "fits_every_label": last(both[s])["practice_accuracy"] == 1, **cost[s]} for s in flagged]
    out = {"flagged": flagged, "cases": per_case, "two_step": stats(final, seeds), "three_step": stats(three, seeds)}
    for k in ("two_step", "three_step"):
        out[k].pop("median_seconds")
    for line in LINES:
        out[f"line_{line}"] = {"gained": [s for s in flagged if d(both, s) < line <= d(final, s)],
                               "lost": [s for s in flagged if d(final, s) < line <= d(both, s)]}
    out["worse"] = [s for s in flagged if d(both, s) > d(final, s)]
    out["still_misses_a_label"] = [s for s in flagged if last(both[s])["practice_accuracy"] < 1]
    out["third_step_cpu_hours"] = sum(both[s]["seconds"] for s in flagged) / 3600
    return out


def main():
    final = cases(EVIDENCE / "two-step-unseen/two-step-unseen.json")
    both = cases(EVIDENCE / "both-wide/both-wide.json")
    out = score(final, both)
    (EVIDENCE / "both-wide/check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
