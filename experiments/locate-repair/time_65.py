"""Time #65's stage by step, and cut it with #74's refines (roadmap #75).

#73 timed #71's stage by step but took #65's stage (58% of the whole repair on its 35 cases) from the seconds #65
recorded. This script reruns #65's stage on a few of #73's cases with each of three refines swapped in
(``cut_refine.install``): the original, ``columns`` (same search, faster loss) and ``loose`` (``columns`` with looser
stopping). It times every step with #73's clock, and compares each run with what #65 recorded for the same case:

- the labels #65's stage ends with (``x``, ``y`` and the 8 region-split inputs that #71's stage starts from). If a
  cheaper refine leaves them bit-identical, #71's stage on top is #74's rerun, whose picks are already checked;
- #65's repair itself (the region arm, with its middle where the middle fits as many labels), its faults found and
  the probe's disagreement with the original network.

    python experiments/locate-repair/time_65.py --kind original --output runs/t65-original --workers 5 --only 264 ...
    python experiments/locate-repair/time_65.py --score runs/t65-original runs/t65-columns runs/t65-loose
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import cut_refine as cr
import early_values as ev
from fresh_repair import repair_65
import pick_fixes as pf
from wide_compare import EVIDENCE, cases

OUT = EVIDENCE / "time-65"
KINDS = ("original", "columns", "loose")
# a spread of #73's sample: 3 cases #65 finished in one stage, 3 in two and 2 in three
SAMPLE = (264, 320, 250, 307, 291, 297, 328, 338)


def recorded(seed):
    """#65's own row for the seed, and the folder holding its saved final labels."""
    if seed < 300:
        return cases(pf.SOURCE / "region-labels.json")[seed], pf.SOURCE
    rows = json.loads((EVIDENCE / "fresh-repair" / "repair-65.json").read_text())
    return next(r for r in rows if r["seed"] == seed), EVIDENCE / "fresh-repair"


def same_labels(path, saved):
    """Whether two saved label files hold the same arrays, bit for bit."""
    a, b = np.load(path), np.load(saved)
    return sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)


def outcome(row):
    """#65's repair as #66 onward start from it: disagreement and faults found."""
    d, found = repair_65(row)
    return {"disagreement": d, "faults_found": found, "stages_run": row["stages_run"],
            "switch_disagreement": row["switch"][-1]["disagreement"]}


def run_seed(args):
    seed, out, kind = args
    import time_repair
    clock = time_repair.CLOCK
    out = Path(out)
    path = out / f"case-{seed}.json"
    if path.exists():
        return seed
    clock.reset()
    cr.CALLS.update(calls=0, evaluations=0)
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    start = time.perf_counter()
    row = ev.two_step(case, 2, saved=lambda k: out / f"case-{seed}.stage{k}.json", middle=True, extra=True)
    wall = time.perf_counter() - start
    before, folder = recorded(seed)
    final = out / f"case-{seed}.stage{row['stages_run'] - 1}.npz"
    steps = {k: round(v, 1) for k, v in clock.steps.items() if k != "probe"}
    result = {"seed": seed, "kind": kind, "seconds": round(wall - clock.probe, 1), "probe_seconds": round(clock.probe, 1),
              "seconds_recorded_by_row": row["seconds"], "seconds_65": before["seconds"], "steps": steps,
              "other": round(wall - clock.probe - sum(steps.values()), 1),
              "refine_calls": cr.CALLS["calls"], "refine_evaluations": cr.CALLS["evaluations"],
              "outcome": outcome(row), "outcome_65": outcome(before),
              "same_labels": same_labels(final, folder / "labels" / f"case-{seed}.npz")}
    path.write_text(json.dumps(result) + "\n")
    return seed


def score(runs):
    """Per run: core-hours by step, and how many cases keep #65's labels and #65's repair."""
    out = {}
    for kind, rows in runs.items():
        steps = {}
        for r in rows.values():
            for k, v in list(r["steps"].items()) + [("other", r["other"])]:
                steps[k] = steps.get(k, 0) + v / 3600
        total = sum(r["seconds"] for r in rows.values()) / 3600
        out[kind] = {"seeds": sorted(rows), "core_hours": total,
                     "core_hours_recorded_65": sum(r["seconds_65"] for r in rows.values()) / 3600,
                     "step_core_hours": dict(sorted(steps.items(), key=lambda kv: -kv[1])),
                     "refine_share": steps.get("margin refine", 0) / total,
                     "evaluations_per_call": sum(r["refine_evaluations"] for r in rows.values())
                     / max(1, sum(r["refine_calls"] for r in rows.values())),
                     "same_labels": sum(r["same_labels"] for r in rows.values()),
                     "same_repair": sum(r["outcome"] == r["outcome_65"] for r in rows.values()),
                     "under_0.01%": sum(r["outcome"]["disagreement"] < pf.LINE for r in rows.values()),
                     "under_0.01%_65": sum(r["outcome_65"]["disagreement"] < pf.LINE for r in rows.values()),
                     "changed": {str(s): [r["outcome_65"], r["outcome"]] for s, r in sorted(rows.items())
                                 if r["outcome"] != r["outcome_65"]}}
    if "original" in out:
        base = out["original"]["core_hours"]
        for kind in out:
            out[kind]["speedup_vs_original"] = base / out[kind]["core_hours"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path, nargs="*")
    a = p.parse_args()
    if a.score:
        OUT.mkdir(parents=True, exist_ok=True)
        runs = {}
        for folder in a.score:
            rows = {json.loads(f.read_text())["seed"]: json.loads(f.read_text()) for f in folder.glob("case-*.json")
                    if ".stage" not in f.name}
            kind = next(iter(rows.values()))["kind"]
            runs[kind] = rows
            (OUT / f"{kind}.json").write_text(json.dumps([rows[s] for s in sorted(rows)], indent=1) + "\n")
        result = score(runs)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    cr.install(a.kind)  # before time_repair wraps the refine in its clock, so the clock times the swapped one
    import time_repair
    for name in ("middle_step", "extra_round"):
        setattr(ev, name, time_repair.CLOCK.tap(getattr(ev, name), name.replace("_", " ")))
    ev.tc.committee = time_repair.CLOCK.tap(ev.tc.committee, "committee (own)")
    ev.st.run = time_repair.CLOCK.tap(ev.st.run, "label rounds (own)")
    a.output.mkdir(parents=True, exist_ok=True)
    seeds = sorted(a.only or SAMPLE, key=lambda s: -recorded(s)[0]["seconds"])
    with Pool(a.workers) as pool:
        for seed in pool.imap_unordered(run_seed, [(s, str(a.output), a.kind) for s in seeds]):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
