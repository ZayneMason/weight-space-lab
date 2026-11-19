"""Compare the largest-edit tie-break with the coarse-fit order on seeds 100 to 149 (roadmap #58).

Both arms run the full two-step repair (``early_values.py --early 2 --two-step``); the only change is how the
committee orders perfect practice fits that tie. Reads evidence/largest-edit/largest-edit.json and the coarse-order
run in evidence/two-step-unseen/two-step-unseen.json, and writes evidence/largest-edit/check.json.
"""
import json

from wide_compare import EVIDENCE, cases, last, stats

LINES = (1e-3, 1e-4)
OUT = EVIDENCE / "largest-edit"


def compare(coarse, largest):
    seeds = sorted(largest)
    missing = [s for s in seeds if s not in coarse]
    if missing:
        raise ValueError(f"no coarse-order row for seeds {missing}")
    d = lambda arm, s: last(arm[s])["disagreement"]
    found = lambda arm, s: last(arm[s])["faults_found"]
    out = {"seeds": len(seeds), "largest_edit": stats(largest, seeds), "coarse": stats(coarse, seeds)}
    for arm in ("largest_edit", "coarse"):
        del out[arm]["median_seconds"]
    for line in LINES:
        out[f"line_{line}"] = {"gained": [s for s in seeds if d(largest, s) < line <= d(coarse, s)],
                               "lost": [s for s in seeds if d(coarse, s) < line <= d(largest, s)]}
    out["more_true_weights"] = [s for s in seeds if found(largest, s) > found(coarse, s)]
    out["fewer_true_weights"] = [s for s in seeds if found(largest, s) < found(coarse, s)]
    out["true_triple_gained"] = [s for s in seeds if largest[s].get("committee_has_true_triple")
                                 and not coarse[s].get("committee_has_true_triple")]
    out["true_triple_lost"] = [s for s in seeds if coarse[s].get("committee_has_true_triple")
                               and not largest[s].get("committee_has_true_triple")]
    out["closer"] = sum(d(largest, s) < d(coarse, s) for s in seeds)
    out["further"] = sum(d(largest, s) > d(coarse, s) for s in seeds)
    out["stages_run"] = {arm: {str(k): sum(a[s]["stages_run"] == k for s in seeds) for k in (1, 2, 3)}
                         for arm, a in (("largest_edit", largest), ("coarse", coarse))}
    out["failures_left"] = [s for s in seeds if d(largest, s) >= LINES[0]]
    out["worst_seed"] = max(seeds, key=lambda s: d(largest, s))
    out["per_seed"] = {str(s): {"coarse": [found(coarse, s), d(coarse, s)], "largest_edit": [found(largest, s), d(largest, s)]}
                       for s in seeds}
    out["cost_ratio"] = out["largest_edit"]["cpu_hours"] / out["coarse"]["cpu_hours"]
    return out


def main():
    result = compare(cases(EVIDENCE / "two-step-unseen" / "two-step-unseen.json"), cases(OUT / "largest-edit.json"))
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
