"""Score #65's extra round of 8 labels on 50 fresh width-16 cases (roadmap #65).

The run used `early_values.py --early 2 --two-step --middle --extra` on seeds 250 to 299. Every case's row keeps the
repair (`switch`), #64's middle step on it (`middle`), and `extra`: one more round of 8 labels from a fresh uniform
pool of 5,000, picked two ways from the same pool. `splits` asks where the committee splits most evenly (the usual
rule); `region` asks where networks built from 200 points of the middle step's walk split most evenly. Each arm refit
every committee member on all labels, picked the winner again and took that winner's middle.

Each arm's final value is its middle whenever the middle fits as many labels as the arm's own fit, else the fit
itself (#64's unconditional rule). Scores five arms per case: the repair, the repair's middle, and each extra-round
arm with and without its middle. Writes evidence/region-labels/check.json.
"""
import json

import numpy as np

from wide_compare import EVIDENCE, cases

OUT = EVIDENCE / "region-labels"


def final(fit, middle):
    """The middle when it fits as many labels as the fit, else the fit's own disagreement."""
    return middle["disagreement"] if middle["accuracy"] >= fit["practice_accuracy"] else fit["disagreement"]


def arms(rows):
    """Seed -> (disagreement, faults found) for every arm."""
    out = {"repair": {}, "middle": {}}
    for arm in ("splits", "region"):
        out[arm] = {}
        out[f"{arm}+middle"] = {}
    for s, r in rows.items():
        end = r["switch"][-1]
        out["repair"][s] = (end["disagreement"], end["faults_found"])
        out["middle"][s] = (final(end, r["middle"]), end["faults_found"])
        for arm in ("splits", "region"):
            e = r["extra"][arm]
            out[arm][s] = (e["disagreement"], e["faults_found"])
            out[f"{arm}+middle"][s] = (final(e, e["middle"]), e["faults_found"])
    return out


def lines(d, base):
    v = [x for x, _ in d.values()]
    b = {s: x for s, (x, _) in base.items()}
    x = {s: x for s, (x, _) in d.items()}
    return {"under_0.1%": sum(q < 1e-3 for q in v), "under_0.01%": sum(q < 1e-4 for q in v),
            "exact": sum(f == 3 for _, f in d.values()),
            "median": float(np.median(v)), "worst": max(v),
            "closer": sorted(s for s in x if x[s] < b[s]), "further": sorted(s for s in x if x[s] > b[s]),
            "crossed_under_0.1%": sorted(s for s in x if b[s] >= 1e-3 > x[s]),
            "crossed_over_0.1%": sorted(s for s in x if b[s] < 1e-3 <= x[s]),
            "crossed_under_0.01%": sorted(s for s in x if b[s] >= 1e-4 > x[s]),
            "crossed_over_0.01%": sorted(s for s in x if b[s] < 1e-4 <= x[s])}


def head_to_head(a, b):
    """Cases where arm ``a`` ends closer than arm ``b``, and the reverse."""
    return {"closer": sorted(s for s in a if a[s][0] < b[s][0]), "further": sorted(s for s in a if a[s][0] > b[s][0])}


def check(rows):
    a = arms(rows)
    end = {s: r["switch"][-1] for s, r in rows.items()}
    return {"seeds": len(rows),
            "stages_run": {str(k): sum(r["stages_run"] == k for r in rows.values()) for k in (1, 2, 3)},
            "cpu_hours": sum(r["seconds"] for r in rows.values()) / 3600,
            "winner_changed": {arm: sorted(s for s, r in rows.items()
                                           if r["extra"][arm]["winner"] != [list(e) for e in end[s]["winner"]])
                               for arm in ("splits", "region")} if all("winner" in e for e in end.values()) else None,
            "splitting_median": {arm: float(np.median([r["extra"][arm]["splitting"] for r in rows.values()]))
                                 for arm in ("splits", "region")},
            "arms": {k: lines(v, a["repair"]) for k, v in a.items()},
            "against_middle": {k: lines(a[k], a["middle"]) for k in ("splits+middle", "region+middle")},
            "region_vs_splits": head_to_head(a["region+middle"], a["splits+middle"])}


def main():
    result = check(cases(OUT / "region-labels.json"))
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("arms", "against_middle")}, indent=1))
    for group in ("arms", "against_middle"):
        print(group)
        for k, v in result[group].items():
            print(" ", k, v["under_0.1%"], v["under_0.01%"], "exact", v["exact"],
                  f"median {v['median']:.2e} worst {v['worst']:.2e}", "closer", len(v["closer"]),
                  "further", len(v["further"]), "in", v["crossed_under_0.01%"], "out", v["crossed_over_0.01%"],
                  "in0.1", v["crossed_under_0.1%"], "out0.1", v["crossed_over_0.1%"])


if __name__ == "__main__":
    main()
