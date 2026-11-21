"""Score #64's middle step on 50 fresh width-16 cases (roadmap #64).

The run used `early_values.py --early 2 --two-step --middle` on seeds 200 to 249. Every case's row keeps the
repair as before (`switch`) and adds `middle`: the 1,600-step walk's mean chord, its middle's practice accuracy and
disagreement, and whether the rule fixed beforehand (mean chord at least 0.005 spreads, and the middle fits every
label the repair fits) would use it. Scores three final values per case: the repair, the rule, and the middle
whenever it fits as many labels as the repair (#63's unconditional arm). Writes evidence/middle-step/check.json.
"""
import json

import numpy as np

from wide_compare import EVIDENCE, cases

OUT = EVIDENCE / "middle-step"
BANDS = ((0, 0.002), (0.002, 0.005), (0.005, float("inf")))


def arms(rows):
    """Seed -> disagreement for the repair, the rule and the unconditional middle."""
    repair = {s: r["switch"][-1]["disagreement"] for s, r in rows.items()}
    rule = {s: r["middle"]["disagreement"] if r["middle"]["used"] else repair[s] for s, r in rows.items()}
    fits = lambda r: r["middle"]["accuracy"] >= r["switch"][-1]["practice_accuracy"]
    always = {s: r["middle"]["disagreement"] if fits(r) else repair[s] for s, r in rows.items()}
    return {"repair": repair, "rule": rule, "always": always}


def lines(d, base):
    v = list(d.values())
    return {"under_0.1%": sum(x < 1e-3 for x in v), "under_0.01%": sum(x < 1e-4 for x in v),
            "median": float(np.median(v)), "worst": max(v),
            "closer": sorted(s for s in d if d[s] < base[s]), "further": sorted(s for s in d if d[s] > base[s]),
            "crossed_under_0.01%": sorted(s for s in d if base[s] >= 1e-4 > d[s]),
            "crossed_over_0.01%": sorted(s for s in d if base[s] < 1e-4 <= d[s])}


def bands(rows, base):
    out = []
    for lo, hi in BANDS:
        g = [s for s, r in rows.items() if lo <= r["middle"]["mean_chord"] < hi]
        ratio = [rows[s]["middle"]["disagreement"] / base[s] for s in g]
        out.append({"from": lo, "to": None if hi == float("inf") else hi, "cases": len(g),
                    "geometric_mean_ratio": float(np.exp(np.mean(np.log(ratio)))) if g else None,
                    "closer": sum(q < 1 for q in ratio)})
    return out


def check(rows):
    a = arms(rows)
    end = {s: r["switch"][-1] for s, r in rows.items()}
    return {"seeds": len(rows), "exact": sum(e["faults_found"] == 3 for e in end.values()),
            "stages_run": {str(k): sum(r["stages_run"] == k for r in rows.values()) for k in (1, 2, 3)},
            "cpu_hours": sum(r["seconds"] for r in rows.values()) / 3600,
            "rule_used": sorted(s for s, r in rows.items() if r["middle"]["used"]),
            "middle_fits_fewer": sorted(s for s, r in rows.items()
                                        if r["middle"]["accuracy"] < end[s]["practice_accuracy"]),
            "arms": {k: lines(v, a["repair"]) for k, v in a.items()}, "by_chord": bands(rows, a["repair"])}


def main():
    result = check(cases(OUT / "middle-step.json"))
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=1))
    for k, v in result["arms"].items():
        print(k, v["under_0.1%"], v["under_0.01%"], f"median {v['median']:.2e} worst {v['worst']:.2e}",
              "closer", len(v["closer"]), "further", len(v["further"]), "in", v["crossed_under_0.01%"],
              "out", v["crossed_over_0.01%"])


if __name__ == "__main__":
    main()
