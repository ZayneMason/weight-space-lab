"""Run the whole repair on 50 fresh cases (roadmap #72).

#71's gate for the bigger committee and its disagree chooser were designed after seeing #67's 50 width-16
three-fault cases (seeds 250 to 299), so its 47 of 50 under 0.01% is not a held-out result. This script runs the
whole repair as it stood after #71, unchanged, on 50 fresh cases (seeds 300 to 349):

1. #65's repair: ``early_values.py --width 16 --early 2 --two-step --middle --extra`` (the two-step rule, #64's
   middle and one extra round of 8 region labels), run in four processes.
2. ``--collect``: merge the four folders, and keep each case's final labels (the last stage's) in ``labels/``.
3. #71 on the collected labels: #66's rebuilt committee, #67's restarts, #68's tied-label rounds, then restarts,
   boundary labels and the bigger committee (where #68's pick misses a label), and the disagree chooser.
4. ``--score``: the probe scores #65's repair, #68's pick and the disagree chooser's pick. The chooser's pick is the
   result, decided before the run; the probe's best candidate is shown only as the ceiling.

    python experiments/locate-repair/early_values.py --width 16 --early 2 --two-step --middle --extra \\
        --only 300 304 ... --output runs/fresh-a/p0          # and p1 to p3 for the other seeds
    python experiments/locate-repair/fresh_repair.py --collect runs/fresh-a
    python experiments/locate-repair/fresh_repair.py --source runs/fresh-a --output runs/fresh-b --workers 4
    python experiments/locate-repair/fresh_repair.py --source runs/fresh-a --score runs/fresh-b
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import shutil

import numpy as np

from pick_fixes import LINE, check as pick_check, run_seed
from region_check import final
from wide_compare import EVIDENCE

OUT = EVIDENCE / "fresh-repair"
SEEDS = tuple(range(300, 350))


def collect(source):
    """Merge the per-process folders into ``source/cases.json`` and copy each case's last-stage labels."""
    rows = {}
    for folder in sorted(p for p in source.iterdir() if p.is_dir() and p.name.startswith("p")):
        for path in folder.glob("case-*.json"):
            if ".stage" in path.name:
                continue
            row = json.loads(path.read_text())
            seed = row["seed"]
            rows[seed] = row
            (source / "labels").mkdir(exist_ok=True)
            shutil.copy(folder / f"case-{seed}.stage{row['stages_run'] - 1}.npz", source / "labels" / f"case-{seed}.npz")
    (source / "cases.json").write_text(json.dumps({"cases": [rows[s] for s in sorted(rows)]}) + "\n")
    return rows


def repair_65(row):
    """#65's repair as #66 onward start from it: the region arm, with its middle when the middle fits as many labels."""
    e = row["extra"]["region"]
    return final(e, e["middle"]), e["faults_found"]


def score(before, rows):
    """#71's lines and crossings, plus #65's repair on the same cases, and what each pick costs in labels."""
    out = pick_check(rows)
    d65 = {s: repair_65(before[s]) for s in rows}
    out["repair_65"] = {"under_0.01%": sum(d < LINE for d, _ in d65.values()),
                        "under_0.1%": sum(d < 1e-3 for d, _ in d65.values()),
                        "exact": sum(f == 3 for _, f in d65.values())}
    pick = {s: r["choosers"]["disagree"]["pick"] for s, r in rows.items()}
    d = {s: rows[s]["candidates"][pick[s]]["disagreement"] for s in rows}
    out["disagree_vs_65"] = {"gained": sorted(s for s in rows if d65[s][0] >= LINE > d[s]),
                             "lost": sorted(s for s in rows if d65[s][0] < LINE <= d[s])}
    out["over_line"] = {str(s): d[s] for s in sorted(rows) if d[s] >= LINE}
    out["median"] = float(np.median(list(d.values())))
    out["labels_asked"] = sum(r["choosers"]["disagree"]["asked"] for r in rows.values())
    out["cpu_hours_65"] = sum(before[s]["seconds"] for s in rows) / 3600
    out["stages_run"] = {str(k): sum(before[s]["stages_run"] == k for s in rows) for k in (1, 2, 3)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--collect", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.collect:
        rows = collect(a.collect)
        print(len(rows), "cases collected")
        return
    before = {c["seed"]: c for c in json.loads((a.source / "cases.json").read_text())["cases"]}
    seeds = [s for s in SEEDS if s in before]
    if a.score:
        rows = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in seeds}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fresh-repair.json").write_text(json.dumps([rows[s] for s in seeds], indent=1) + "\n")
        (OUT / "repair-65.json").write_text(json.dumps([before[s] for s in seeds]) + "\n")
        result = score(before, rows)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    # cases #65 rebuilt with a bigger committee first, since they are the slowest
    jobs = [(s, str(a.output), before[s]["stages_run"], str(a.source))
            for s in sorted(seeds, key=lambda s: -before[s]["stages_run"])]
    with Pool(a.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
