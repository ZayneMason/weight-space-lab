"""Score other tie-breaks among perfect practice fits on the saved feature files (roadmap #57).

Reads evidence/tie-breaks/ties-{seed}-{pairs}.npz, written by tie_breaks.py, and writes
evidence/tie-breaks/summary.json: for each case and each tie-break, the true triple's rank in the committee
order (accuracy first, then the tie-break), and how many lost cases each tie-break would keep within 200.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "evidence" / "tie-breaks"
LOST = (52, 55, 62, 63, 66, 68, 76, 93, 96, 124, 126, 149)  # #51's nine and #55's three
COMMITTEE = 200

# name -> (key, sign): smaller key first after accuracy; sign -1 means larger is better
TIE_BREAKS = {"coarse_rank (current)": ("rank7", 1), "smallest_total_edit": ("cost", 1),
              "largest_min_margin": ("margin_min", -1), "largest_mean_margin": ("margin_mean", -1),
              "smallest_largest_edit": ("largest", 1), "screen_order": ("screen", 1), "fewest_biases": ("biases", 1)}


def rank(f, key, sign):
    """The true triple's 1-based position, or None if it was never extended."""
    if not f["is_true"].any():
        return None
    second = np.nan_to_num(sign * np.asarray(f[key], float), nan=np.inf)
    order = np.lexsort((second, -np.asarray(f["acc"])))
    return int(np.flatnonzero(f["is_true"][order])[0]) + 1


def summarise(files):
    rows = []
    for (seed, pairs), f in sorted(files.items()):
        perfect = int((f["acc"] == 1).sum())
        rows.append({"seed": seed, "pairs": pairs, "lost": seed in LOST, "triples": len(f["acc"]), "perfect_fits": perfect,
                     "true_fits_every_label": bool(f["is_true"].any() and f["acc"][f["is_true"]][0] == 1),
                     "ranks": {name: rank(f, k, s) for name, (k, s) in TIE_BREAKS.items()}})
    kept = lambda r, name: r["ranks"][name] is not None and r["ranks"][name] <= COMMITTEE
    totals = {name: {"lost_kept": sum(kept(r, name) for r in rows if r["lost"]),
                     "controls_kept": sum(kept(r, name) for r in rows if not r["lost"])} for name in TIE_BREAKS}
    return {"committee": COMMITTEE, "lost_cases": sum(r["lost"] for r in rows),
            "controls": sum(not r["lost"] for r in rows), "by_tie_break": totals, "cases": rows}


def main():
    files = {}
    for p in sorted(HERE.glob("ties-*.npz")):
        f = dict(np.load(p))
        files[(int(f["seed"]), int(f["limit"]))] = f
    out = summarise(files)
    (HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in out["cases"]:
        print(r["seed"], r["pairs"], r["perfect_fits"], r["ranks"])
    print(json.dumps(out["by_tie_break"], indent=1))


if __name__ == "__main__":
    main()
