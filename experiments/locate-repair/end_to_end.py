"""Run #71's stage on the labels #75's cheaper refines left, with the same refine (roadmap #76).

#74 checked the cheaper refines on #71's stage from #65's original labels, and #75 found the same refines change the
labels #65's stage ends with (columns in 4 of #75's 8 cases, loose in all 8). This script finishes the whole repair
the way a run with that refine would: it reruns #71's stage (#73's timed runner) with the refine swapped in, starting
from the final labels #75 saved for the same refine. Where #75's columns run kept #65's labels bit for bit, #74's
columns row already is the end-to-end run, so those cases are not rerun.

    python experiments/locate-repair/end_to_end.py --kind columns --labels runs/t65-columns --output runs/e2e-columns --only 250 291 297 338
    python experiments/locate-repair/end_to_end.py --kind loose --labels runs/t65-loose --output runs/e2e-loose
    python experiments/locate-repair/end_to_end.py --score columns=runs/e2e-columns loose=runs/e2e-loose
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cut_refine as cr
import early_values as ev
from pick_fixes import LINE
from time_65 import SAMPLE
from wide_compare import EVIDENCE

OUT = EVIDENCE / "end-to-end"


def final_labels(folder, seed):
    """#75's row for the seed and the path of the labels its stage ended with."""
    row = json.loads((Path(folder) / f"case-{seed}.json").read_text())
    return row, Path(folder) / f"case-{seed}.stage{row['outcome']['stages_run'] - 1}.npz"


def labels_from(folder):
    """A drop-in for ``rebuild_committee.labels`` that reads #75's saved labels instead of #65's."""
    def labels(case, seed, source=None):
        z = np.load(final_labels(folder, seed)[1])
        extra = z["added_region"]
        return (np.concatenate([z["x"], extra]), np.concatenate([z["y"], ev.tf.predict(case["teacher"], extra)]))
    return labels


def chosen(row):
    """The disagree chooser's pick in a #73-style row: its disagreement and faults found."""
    c = row["candidates"][row["full"]["pick"]]
    return c["disagreement"], c["faults_found"]


def stage(row):
    return sum(row["seconds"].values())


def score(runs):
    """Each refine's whole repair on #75's sample against the original's (#75's original run plus #73's stage)."""
    t73 = {r["seed"]: r for r in json.loads((EVIDENCE / "time-repair" / "time-repair.json").read_text())}
    t65 = {k: {r["seed"]: r for r in json.loads((EVIDENCE / "time-65" / f"{k}.json").read_text())}
           for k in ("original", "columns", "loose")}
    c74 = {k: {r["seed"]: r for r in json.loads((EVIDENCE / "cut-refine" / f"{k}.json").read_text())}
           for k in ("columns", "loose")}
    base = sum(t65["original"][s]["seconds"] + stage(t73[s]) for s in SAMPLE)
    out = {"original": {"core_hours": base / 3600, "under_0.01%": sum(chosen(t73[s])[0] < LINE for s in SAMPLE),
                        "exact": sum(chosen(t73[s])[1] == 3 for s in SAMPLE),
                        "chosen": {str(s): chosen(t73[s]) for s in SAMPLE}}}
    for kind, rows in runs.items():
        # a case #75 left with #65's labels takes #74's row for the same refine: the same run from the same labels
        full = {s: rows.get(s) or (c74[kind][s] if t65[kind][s]["same_labels"] else None) for s in SAMPLE}
        missing = [s for s in SAMPLE if full[s] is None]
        if missing:
            raise SystemExit(f"{kind}: no end-to-end row for {missing}")
        total = sum(t65[kind][s]["seconds"] + stage(full[s]) for s in SAMPLE)
        out[kind] = {"rerun": sorted(rows), "from_74": sorted(s for s in SAMPLE if s not in rows),
                     "core_hours": total / 3600, "speedup_vs_original": base / total,
                     "core_hours_stage_71": sum(stage(full[s]) for s in SAMPLE) / 3600,
                     "under_0.01%": sum(chosen(full[s])[0] < LINE for s in SAMPLE),
                     "exact": sum(chosen(full[s])[1] == 3 for s in SAMPLE),
                     "same_pick_as_73": sum(full[s]["full"]["pick"] == t73[s]["full"]["pick"] for s in SAMPLE),
                     "same_disagreement_as_73": sum(chosen(full[s])[0] == chosen(t73[s])[0] for s in SAMPLE),
                     "crossed": {"gained": [s for s in SAMPLE if chosen(t73[s])[0] >= LINE > chosen(full[s])[0]],
                                 "lost": [s for s in SAMPLE if chosen(full[s])[0] >= LINE > chosen(t73[s])[0]]},
                     "chosen": {str(s): chosen(full[s]) for s in SAMPLE},
                     "labels_asked": sum(full[s]["full"].get("asked", 0) for s in SAMPLE)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=cr.KINDS)
    p.add_argument("--labels", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", nargs="*")
    a = p.parse_args()
    if a.score:
        runs = {}
        for item in a.score:
            kind, folder = item.split("=")
            runs[kind] = {json.loads(f.read_text())["seed"]: json.loads(f.read_text())
                          for f in Path(folder).glob("case-*.json")}
        OUT.mkdir(parents=True, exist_ok=True)
        for kind, rows in runs.items():
            (OUT / f"{kind}.json").write_text(json.dumps([rows[s] for s in sorted(rows)], indent=1) + "\n")
        result = score(runs)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    cr.install(a.kind)  # before time_repair wraps the refine in its clock
    import time_repair
    time_repair.labels = labels_from(a.labels)
    time_repair.stages_of = lambda seed: final_labels(a.labels, seed)[0]["outcome"]["stages_run"]
    a.output.mkdir(parents=True, exist_ok=True)
    seeds = sorted(a.only or SAMPLE, key=lambda s: -final_labels(a.labels, s)[0]["seconds"])
    with Pool(a.workers) as pool:  # forked after the patches above, so every worker runs with them
        for seed in pool.imap_unordered(cr.run_seed, [(s, str(a.output), a.kind) for s in seeds]):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
