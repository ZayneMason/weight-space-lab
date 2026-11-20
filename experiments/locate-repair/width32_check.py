"""Score the full two-step repair at hidden width 32 against the width-16 baseline (roadmap #60).

Width 32 uses fresh seeds 200 to 211 (a different network per seed), so the comparison is of shares, not seed by
seed. Reads evidence/width-32/width-32.json and the width-16 coarse-order run on seeds 100 to 149 in
evidence/two-step-unseen/two-step-unseen.json, and writes evidence/width-32/check.json.
"""
import json

import numpy as np

from wide_compare import EVIDENCE, cases, last, stats

OUT = EVIDENCE / "width-32"


def arm(rows):
    seeds = sorted(rows)
    out = stats(rows, seeds)
    out["seeds"] = len(seeds)
    for key in ("under_0.1%", "under_0.01%", "exact", "true_triple_in_committee"):
        out[f"share_{key}"] = out[key] / len(seeds)
    out["cpu_hours_per_case"] = out["cpu_hours"] / len(seeds)
    out["median_hours"] = out.pop("median_seconds") / 3600
    out["max_hours"] = max(rows[s]["seconds"] for s in seeds) / 3600
    out["median_triples"] = float(np.median([rows[s]["triples"] for s in seeds]))
    out["stages_run"] = {str(k): sum(rows[s]["stages_run"] == k for s in seeds) for k in (1, 2, 3)}
    return out


def third_stage(rows):
    """What the 1,000-pair third stage did, from each case's saved per-stage results, and the arm without it."""
    reached = [s for s in sorted(rows) if rows[s]["stages_run"] == 3]
    before = {s: rows[s]["two_step"][1] for s in reached}
    after = {s: rows[s]["two_step"][2] for s in reached}
    end = lambda s: before[s] if s in before else last(rows[s])
    d = [end(s)["disagreement"] for s in sorted(rows)]
    return {"reached": reached,
            "better": [s for s in reached if after[s]["disagreement"] < before[s]["disagreement"]],
            "worse": [s for s in reached if after[s]["disagreement"] > before[s]["disagreement"]],
            "ended_exact": [s for s in reached if after[s]["faults_found"] == 3],
            "perfect_practice": [s for s in reached if after[s]["practice_accuracy"] == 1],
            "true_triple_in_committee": [s for s in reached if after[s]["committee_has_true_triple"]],
            "cpu_hours": sum(after[s]["seconds"] for s in reached) / 3600,
            "share_of_cpu": sum(after[s]["seconds"] for s in reached) / sum(rows[s]["seconds"] for s in rows),
            "without": {"under_0.1%": sum(v < 1e-3 for v in d), "under_0.01%": sum(v < 1e-4 for v in d),
                        "exact": sum(end(s)["faults_found"] == 3 for s in sorted(rows))}}


def compare(base, wide):
    out = {"width_16": arm(base), "width_32": arm(wide)}
    out["cost_ratio_per_case"] = out["width_32"]["cpu_hours_per_case"] / out["width_16"]["cpu_hours_per_case"]
    out["triples_ratio"] = out["width_32"]["median_triples"] / out["width_16"]["median_triples"]
    d = lambda s: last(wide[s])["disagreement"]
    out["width_32_not_under_0.01%"] = [s for s in sorted(wide) if d(s) >= 1e-4]
    out["width_32_not_exact"] = [s for s in sorted(wide) if last(wide[s])["faults_found"] != 3]
    out["third_stage"] = {arm: third_stage(rows) for arm, rows in (("width_16", base), ("width_32", wide))}
    out["per_seed"] = {str(s): {"faults_found": last(wide[s])["faults_found"], "disagreement": d(s),
                                "hours": wide[s]["seconds"] / 3600, "stages_run": wide[s]["stages_run"],
                                "true_triple_in_committee": bool(wide[s].get("committee_has_true_triple"))}
                       for s in sorted(wide)}
    return out


def main():
    result = compare(cases(EVIDENCE / "two-step-unseen" / "two-step-unseen.json"), cases(OUT / "width-32.json"))
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
