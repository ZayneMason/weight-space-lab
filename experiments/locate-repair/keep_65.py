"""Keep #65's own repair in the running in #71's chooser (roadmap #79).

On seed 414 (#78) #65's stage ended with an exact repair at 0.0035% disagreement, but #71's disagree chooser picks
only among its own candidates (pick68, restarts, boundary and, where the gate opens, the bigger committee), and it
returned a one-fault fit at 0.069%. This script reruns #71's stage, with the cheap repair (columns for every refine
and the batched committee fits, as #77 and #78), on #72's 50 fresh seeds (300 to 349) and #78's (400 to 449), and
adds a fifth candidate:

- ``repair65``: the triple #65's stage picked, fitted on #65's final labels (the 1,096 labels #71's stage starts from)
  with the same fit the committee uses for every member (``warm_triples.fit``), and its final network taken the way
  every candidate's is (#64's middle when it fits as many labels as the fit). #65's own values are not saved, so this
  rebuilds them; the score reports how often its disagreement matches the one #65's stage recorded.

Each case runs the disagree chooser twice on the same pool: once on #71's candidates alone (``without``, the
baseline, as #78 ran it) and once with ``repair65`` added (``with``). Both choosers' correct counts are saved, so
the score can try two tie rules for ``repair65``: last (ties keep #71's order, pick68 first) and first (ties go to
#65's repair). Every candidate's network edits are saved too, so later choosers can be scored without a rerun.

    python experiments/locate-repair/keep_65.py --prepare runs/f78
    python experiments/locate-repair/keep_65.py --output runs/k65 --workers 4
    python experiments/locate-repair/keep_65.py --score runs/k65
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import shutil
import time

import numpy as np

import batched_refine as br
import cut_refine as cr
import early_values as ev
import pick_fixes as pf
from rebuild_committee import labels
import tied_labels
from wide_compare import EVIDENCE

OUT = EVIDENCE / "keep-65"
FRESH_72 = EVIDENCE / "fresh-repair"
FRESH_78 = EVIDENCE / "fresh-50"
SEEDS = tuple(range(300, 350)) + tuple(range(400, 450))
LINE = pf.LINE
LAST = pf.ORDER + ("repair65",)
FIRST = ("repair65",) + pf.ORDER


def prepare(folder):
    """Copy #78's final labels and #65 rows (from the cheap run's folder) into the evidence, as #72's are."""
    rows = []
    (FRESH_78 / "labels").mkdir(parents=True, exist_ok=True)
    for seed in range(400, 450):
        last = json.loads((folder / f"case-{seed}.json").read_text())["outcome"]["stages_run"] - 1
        row = json.loads((folder / f"case-{seed}.stage{last}.json").read_text())
        rows.append({"seed": seed, "stages_run": last + 1, "extra": row["extra"], "seconds": row["seconds"]})
        shutil.copy(folder / f"case-{seed}.stage{last}.npz", FRESH_78 / "labels" / f"case-{seed}.npz")
    (FRESH_78 / "repair-65.json").write_text(json.dumps(rows) + "\n")


def repair_65_rows():
    rows = json.loads((FRESH_72 / "repair-65.json").read_text()) + json.loads((FRESH_78 / "repair-65.json").read_text())
    return {r["seed"]: r for r in rows}


def recorded_65(row):
    """#65's repair as #72 and #78 scored it: the region arm, with its middle when the middle fits as many labels."""
    e = row["extra"]["region"]
    m = e["middle"]
    return (m["disagreement"] if m["accuracy"] >= e["practice_accuracy"] else e["disagreement"]), e["faults_found"]


def choose(correct, order):
    best = max(correct.values())
    return next(n for n in order if correct.get(n) == best)


def run_seed(args):
    seed, out = args
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    cr.CALLS.update(calls=0, evaluations=0)
    br.CALLS.update(calls=0, evaluations=0)
    start = time.perf_counter()
    r65 = repair_65_rows()[seed]
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x0, y0 = labels(case, seed, FRESH_72 if seed < 400 else FRESH_78)
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    options = ev.TWO_STEP[r65["stages_run"] - 1]
    keep = {}
    tied_labels.run_case(case, x0, y0, options, probe, seed, keep=keep)
    members, table, x, y = keep["members"], keep["table"], keep["x"], keep["y"]
    w = pf.pick(members, table, "coarse", case)
    built = {"pick68": (w, table[w], x, y)}
    t, f = pf.restarts(case, x, y, members, table, w, seed)
    built["restarts"] = (t, f, x, y)
    built["boundary"] = pf.boundary(case, x, y, w, table[w], seed)
    gate = bool(table[w][0][0] < 1.0)
    if gate:
        t, f, _ = pf.bigger(case, x, y, options, seed)
        built["bigger"] = (t, f, x, y)
    t65 = tuple(tuple(e) for e in r65["extra"]["region"]["winner"])
    built["repair65"] = (t65, pf.wt.fit(case, x0, y0, t65), x0, y0)
    rows, networks = {}, {}
    for n, (t, f, cx, cy) in built.items():
        edits, used = pf.final_edits(case, cx, cy, t, f)
        networks[n] = edits
        rows[n] = {"winner": [list(e) for e in t], "middle_used": used, "practice_accuracy": float(f[0][0]),
                   "faults_found": len(ev.t3.hits(t, case["faults"], case["width"])),
                   "disagreement": ev.aq.disagreement(case, edits, probe),
                   "edits": [[a, int(i), float(v)] for a, i, v in edits]}
    without = pf.choosers(case, {n: e for n, e in networks.items() if n != "repair65"}, seed)["disagree"]
    order, pf.ORDER = pf.ORDER, LAST  # pf.choose looks names up in ORDER; without repair65 there it raises StopIteration
    try:                              # when repair65 alone fits most labels, which a Pool swallows as a finished job
        with_65 = pf.choosers(case, networks, seed)["disagree"]
    finally:
        pf.ORDER = order
    with_65["pick_last"] = choose(with_65["correct"], LAST)
    with_65["pick_first"] = choose(with_65["correct"], FIRST)
    del with_65["pick"]
    row = {"seed": seed, "gate": gate, "candidates": rows, "without": without, "with": with_65,
           "recorded_65": recorded_65(r65), "seconds": round(time.perf_counter() - start, 1),
           "refine_calls": cr.CALLS["calls"] + br.CALLS["calls"], "batched_calls": br.CALLS["calls"]}
    path.write_text(json.dumps(row) + "\n")
    return seed


def score(rows):
    seeds = sorted(rows)
    d = lambda s, n: rows[s]["candidates"][n]["disagreement"]
    exact = lambda s, n: rows[s]["candidates"][n]["faults_found"] == 3
    picks = {"without": {s: rows[s]["without"]["pick"] for s in seeds},
             "with_last": {s: rows[s]["with"]["pick_last"] for s in seeds},
             "with_first": {s: rows[s]["with"]["pick_first"] for s in seeds},
             "repair65": {s: "repair65" for s in seeds},
             "probe_best": {s: min(rows[s]["candidates"], key=lambda n: (d(s, n), LAST.index(n))) for s in seeds}}
    out = {"seeds": seeds, "core_hours": sum(rows[s]["seconds"] for s in seeds) / 3600}
    base = picks["without"]
    for name, p in picks.items():
        out[name] = {
            "under_0.01%": sum(d(s, p[s]) < LINE for s in seeds), "under_0.1%": sum(d(s, p[s]) < 10 * LINE for s in seeds),
            "exact": sum(exact(s, p[s]) for s in seeds),
            "took_65": sorted(s for s in seeds if p[s] == "repair65"),
            "gained": sorted(s for s in seeds if d(s, p[s]) < LINE <= d(s, base[s])),
            "lost": sorted(s for s in seeds if d(s, base[s]) < LINE <= d(s, p[s])),
            "better": sorted(s for s in seeds if d(s, p[s]) < d(s, base[s])),
            "worse": sorted(s for s in seeds if d(s, p[s]) > d(s, base[s])),
            "over_line": {str(s): d(s, p[s]) for s in seeds if d(s, p[s]) >= LINE}}
        for half, part in (("fresh_72", [s for s in seeds if s < 400]), ("fresh_78", [s for s in seeds if s >= 400])):
            out[name][half] = [sum(d(s, p[s]) < LINE for s in part), sum(exact(s, p[s]) for s in part)]
    rec = {s: rows[s]["recorded_65"] for s in seeds}
    out["rebuilt_65"] = {
        "same_faults_found": sum(rows[s]["candidates"]["repair65"]["faults_found"] == rec[s][1] for s in seeds),
        "within_0.001%": sum(abs(d(s, "repair65") - rec[s][0]) <= 1e-5 for s in seeds),
        "same_side_of_line": sum((d(s, "repair65") < LINE) == (rec[s][0] < LINE) for s in seeds),
        "recorded_under_0.01%": sum(r[0] < LINE for r in rec.values()),
        "rebuilt_under_0.01%": sum(d(s, "repair65") < LINE for s in seeds),
        "differs": {str(s): [rec[s][0], d(s, "repair65")] for s in seeds if abs(d(s, "repair65") - rec[s][0]) > 1e-5}}
    out["labels_asked"] = {"without": sum(rows[s]["without"]["asked"] for s in seeds),
                           "with": sum(rows[s]["with"]["asked"] for s in seeds)}
    out["no_split"] = {"without": sorted(s for s in seeds if rows[s]["without"]["split_inputs"] == 0),
                       "with": sorted(s for s in seeds if rows[s]["with"]["split_inputs"] == 0)}
    out["bigger_ran"] = sorted(s for s in seeds if rows[s]["gate"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prepare", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare)
        return
    if a.score:
        rows = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in (a.only or SEEDS)}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "keep-65.json").write_text(json.dumps([rows[s] for s in sorted(rows)]) + "\n")
        result = score(rows)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    cr.install("columns")
    br.install()
    a.output.mkdir(parents=True, exist_ok=True)
    stages = {s: r["stages_run"] for s, r in repair_65_rows().items()}
    jobs = [(s, str(a.output)) for s in sorted(a.only or SEEDS, key=lambda s: -stages[s])]
    print("jobs", len(jobs), flush=True)
    with Pool(a.workers) as pool:  # forked after the installs above, so every worker runs the cheap repair
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
