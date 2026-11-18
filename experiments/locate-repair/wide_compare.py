"""Compare the 1,000-pair screen with the 200-pair screen on all 50 three-fault cases (roadmap #48).

Reads only committed evidence and writes evidence/wide-screen/comparison.json. Also scores an
adaptive rule: keep the 200-pair run unless no committee member fits every practice label before
any targeted label, and only then rerun the case with 1,000 pairs.
"""
from pathlib import Path
import json

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
LINES = (1e-3, 1e-4)


def cases(path):
    return {c["seed"]: c for c in json.loads(Path(path).read_text())["cases"]}


def last(case):
    return case["switch"][-1]


def stats(arm, seeds):
    ends = [last(arm[s]) for s in seeds]
    d = [e["disagreement"] for e in ends]
    return {"under_0.1%": sum(v < 1e-3 for v in d), "under_0.01%": sum(v < 1e-4 for v in d),
            "exact": sum(e["faults_found"] == 3 for e in ends), "every_visible": sum(bool(e["found_visible"]) for e in ends),
            "true_triple_in_committee": sum(bool(arm[s].get("committee_has_true_triple")) for s in seeds),
            "median": float(np.median(d)), "worst": max(d),
            "median_seconds": float(np.median([arm[s]["seconds"] for s in seeds])),
            "cpu_hours": sum(arm[s]["seconds"] for s in seeds) / 3600}


def needs_wider(case):
    """True when the 200-pair committee's best member misses a practice label before targeted labels."""
    return case["switch"][0]["practice_accuracy"] < 1


def compare(narrow, wide):
    seeds = sorted(wide)
    d = lambda arm, s: last(arm[s])["disagreement"]
    same = lambda s: sorted(map(tuple, last(wide[s])["winner"])) == sorted(map(tuple, last(narrow[s])["winner"]))
    out = {"seeds": seeds, "pairs_200": stats(narrow, seeds), "pairs_1000": stats(wide, seeds)}
    for line in LINES:
        out[f"line_{line}"] = {"gained": [s for s in seeds if d(wide, s) < line <= d(narrow, s)],
                               "lost": [s for s in seeds if d(narrow, s) < line <= d(wide, s)]}
    out["closer"] = sum(d(wide, s) < d(narrow, s) for s in seeds)
    out["further"] = sum(d(wide, s) > d(narrow, s) for s in seeds)
    ratios = [d(wide, s) / max(d(narrow, s), 1e-7) for s in seeds]
    out["median_ratio"] = float(np.median(ratios))
    out["same_winner"] = [s for s in seeds if same(s)]
    same_ratios = [r for s, r in zip(seeds, ratios) if same(s)]
    out["same_winner_ratio_range"] = [min(same_ratios), max(same_ratios)]
    out["committee_gained"] = [s for s in seeds if wide[s]["committee_has_true_triple"] and not narrow[s]["committee_has_true_triple"]]
    out["committee_lost"] = [s for s in seeds if narrow[s]["committee_has_true_triple"] and not wide[s]["committee_has_true_triple"]]
    out["committee_still_lacks"] = [s for s in seeds if not wide[s]["committee_has_true_triple"]]
    flagged = [s for s in seeds if needs_wider(narrow[s])]
    adaptive = {s: wide[s] if s in flagged else narrow[s] for s in seeds}
    out["adaptive"] = {**stats(adaptive, seeds), "flagged": flagged,
                       "cpu_hours": (sum(narrow[s]["seconds"] for s in seeds) + sum(wide[s]["seconds"] for s in flagged)) / 3600}
    del out["adaptive"]["median_seconds"]
    out["per_case"] = {s: {"pairs_200": d(narrow, s), "pairs_1000": d(wide, s), "same_winner": same(s),
                           "committee_200": narrow[s]["committee_has_true_triple"],
                           "committee_1000": wide[s]["committee_has_true_triple"]} for s in seeds}
    return out


def main():
    narrow = {**cases(EVIDENCE / "switch-point/early-2.json"), **cases(EVIDENCE / "fresh-switch/fresh-early-2.json")}
    wide = cases(EVIDENCE / "wide-screen/wide-screen.json")
    result = compare(narrow, wide)
    (EVIDENCE / "wide-screen/comparison.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=1))


if __name__ == "__main__":
    main()
