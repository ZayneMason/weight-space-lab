"""Duel only when the labels can tell the two apart (roadmap #81).

#80's two-round chooser runs #71's disagree chooser on its own candidates, then duels its pick against #65's repair on
up to 32 labels where the two disagree, and takes #65's repair when it answers more of them right. On #79's 100 cases
it kept 319's pick but made 6 cases further from the original, and 3 more on #80's 50, all in duels with 1 to 7 split
inputs. This script scores, from the duels #80 saved (no network is refitted and no new label is drawn), a second round
that is skipped unless the duel has at least ``k`` split inputs and #65's repair wins by at least ``m`` labels. The
rule is picked on #79's 100 cases (``tune``) and checked on #80's 50 (``check``). Counting split inputs needs no labels,
so a skipped duel asks none.

    python experiments/locate-repair/gated_duel.py
"""

import json

from pick_fixes import LINE
from wide_compare import EVIDENCE

OUT = EVIDENCE / "gated-duel"
SAVED = EVIDENCE / "saved-65"
KS = (1, 2, 4, 8, 16, 32, 64)
MS = (1, 2, 4, 8)


def load():
    """One row per case: round one's pick, the duel, and every candidate's disagreement and fault count."""
    tune = [{"seed": r["seed"], "first": r["first_round"], "duel": r["duel"], "disagreement": r["disagreement"],
             "faults_found": r["faults_found"], "one_round": r["picks"]["with_last"],
             "asked_first": r["asked"]["without"], "asked_one_round": r["asked"]["with_last"]}
            for r in json.loads((SAVED / "rescore-79.json").read_text())]
    check = [{"seed": r["seed"], "first": r["without"]["pick"], "duel": r["two_round"],
              "disagreement": {n: c["disagreement"] for n, c in r["candidates"].items()},
              "faults_found": {n: c["faults_found"] for n, c in r["candidates"].items()},
              "one_round": r["with_saved"]["pick_last"],
              "asked_first": r["without"]["asked"], "asked_one_round": r["with_saved"]["asked"]}
             for r in json.loads((SAVED / "saved-65.json").read_text())]
    return tune, check


def pick(row, k, m):
    """Round one's pick, or #65's repair when the duel has ``k`` split inputs and #65's repair wins by ``m``."""
    d = row["duel"]
    first = row["first"]
    if d["split_inputs"] >= k and d["correct"]["repair65"] - d["correct"][first] >= m:
        return "repair65"
    return first


def asked(row, k):
    """Labels the rule asks: round one's, plus the duel's when it has enough split inputs to run."""
    return row["asked_first"] + (row["duel"]["asked"] if row["duel"]["split_inputs"] >= k else 0)


def tally(rows, picks, labels):
    d = lambda r, n: r["disagreement"][n]
    out = {"under_0.01%": sum(d(r, picks[r["seed"]]) < LINE for r in rows),
           "exact": sum(r["faults_found"][picks[r["seed"]]] == 3 for r in rows),
           "took_65": sorted(r["seed"] for r in rows if picks[r["seed"]] == "repair65"),
           "better": sorted(r["seed"] for r in rows if d(r, picks[r["seed"]]) < d(r, r["first"])),
           "worse": sorted(r["seed"] for r in rows if d(r, picks[r["seed"]]) > d(r, r["first"])),
           "labels_asked": labels}
    out["gained"] = sorted(r["seed"] for r in rows if d(r, picks[r["seed"]]) < LINE <= d(r, r["first"]))
    out["lost"] = sorted(r["seed"] for r in rows if d(r, r["first"]) < LINE <= d(r, picks[r["seed"]]))
    return out


def score(rows, k, m):
    return tally(rows, {r["seed"]: pick(r, k, m) for r in rows}, sum(asked(r, k) for r in rows))


def baselines(rows):
    return {"without": tally(rows, {r["seed"]: r["first"] for r in rows}, sum(r["asked_first"] for r in rows)),
            "one_round": tally(rows, {r["seed"]: r["one_round"] for r in rows},
                               sum(r["asked_one_round"] for r in rows))}


def choose_rule(grid):
    """Fewest cases made worse, then most made better, then fewest labels, then the smallest k and m."""
    return min(grid, key=lambda km: (len(grid[km]["worse"]), -len(grid[km]["better"]), grid[km]["labels_asked"], km))


def main():
    tune, check = load()
    grid = {(k, m): score(tune, k, m) for k in KS for m in MS}
    k, m = choose_rule(grid)
    result = {
        "rule": {"min_split_inputs": k, "min_margin": m},
        "tune_79": {**baselines(tune), "gated": grid[(k, m)], "ungated": grid[(1, 1)]},
        "check_80": {**baselines(check), "gated": score(check, k, m), "ungated": score(check, 1, 1)},
        "grid_79": {f"k={a},m={b}": [len(v["better"]), len(v["worse"]), v["labels_asked"]] for (a, b), v in grid.items()},
        "grid_80": {f"k={a},m={b}": [len(v["better"]), len(v["worse"]), v["labels_asked"]]
                    for a in KS for b in MS for v in [score(check, a, b)]},
        "duel_split_inputs": {"tune_79": {str(r["seed"]): r["duel"]["split_inputs"] for r in tune},
                              "check_80": {str(r["seed"]): r["duel"]["split_inputs"] for r in check}},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in ("rule", "tune_79", "check_80")}, indent=1))


if __name__ == "__main__":
    main()
