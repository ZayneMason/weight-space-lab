"""Run the whole repair, fixed in advance, on fresh seeds (roadmap #90).

Every step since #84 was built after looking at the same 200 cases. This runs the whole repair on 50 seeds nothing
has been run on (700 to 749), with every step and threshold fixed before the run:

1. #83's first part as it ran there: #65's stage with its network saved, #71's stage and #81's gated duel
   (``saved_65.run_seed`` and ``fresh_line.gated``), giving the pick and #65's 1,096 final labels;
2. where the pick misses one of those labels, #84's keep-one search; elsewhere #89's keep-two search; both with #85's
   fitter, keeping the triples that fit every label;
3. #88's refit duel on the pick and those triples;
4. #82's flat-line step on the duel's winner only where the duel keeps the pick (#89's reading, fixed here).

The result was fixed in advance as the network after step 4. Beside it the score reports the pick, the duel's winner
and the arm with the line step on every winner (#89's order).

    python experiments/locate-repair/fresh_end.py --output runs/e90 --part 0 --parts 4    # step 1, parts 0 to 3
    python experiments/locate-repair/fresh_end.py --repair runs/e90 --workers 4           # steps 2 to 4
    python experiments/locate-repair/fresh_end.py --score runs/e90
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import shutil
import time

import numpy as np

import early_values as ev
import fresh_line as fl
import keep_one as k
import perfect_fit as pfit
import refit_duel as rd
import saved_65
from keep_two import triples as keep_two_triples
from value_labels import line_arm
from wide_compare import EVIDENCE

OUT = EVIDENCE / "fresh-end"
SEEDS = tuple(range(700, 750))
LINE = k.LINE


def first_part(seed, out):
    saved_65.run_seed(seed, out)


def pick_of(row):
    p = fl.gated(row)
    c = row["candidates"][p]
    return {"pick": p, "winner": c["winner"], "edits": c["edits"], "disagreement": c["disagreement"],
            "faults_found": c["faults_found"]}


def prepare(folder):
    """Copy each case's final #65 labels into the evidence, where ``keep_one.load`` reads them."""
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        last = json.loads((folder / f"case-{seed}.json").read_text())["stages_run"] - 1
        shutil.copy(folder / f"case-{seed}.stage{last}.npz", OUT / "labels" / f"case-{seed}.npz")


def values(net):
    by = {(n, i): v for n, i, v in net["edits"]}
    return [by[(n, i)] for n, i in net["triple"]]


def repair(args):
    seed, pick, out = args
    path = out / f"case-{seed}.repair.json"
    if path.exists():
        return seed
    start = time.perf_counter()
    case, x, y = k.load(seed)
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    missed = k.wrong(case, pick["edits"], x, y)
    ts = k.triples(case, pick["winner"]) if missed else keep_two_triples(case, pick["winner"])
    found = [[list(p) for p in t] for t in ts if k.wt.fit(case, x, y, t)[0][0] == 1.0]
    search_seconds = time.perf_counter() - start
    duel_dir = out / "duel"
    duel_dir.mkdir(exist_ok=True)
    duel = None
    net = {"triple": [list(w) for w in pick["winner"]], "edits": pick["edits"],
           "disagreement": pick["disagreement"], "faults_found": pick["faults_found"]}
    if found:
        rd.run_seed((seed, {"pick": pick, "triples": found, "before": [pick["disagreement"], pick["faults_found"]]},
                     duel_dir))
        duel = json.loads((duel_dir / f"case-{seed}.json").read_text())
        if duel["winner"] != "pick":
            net = {"triple": found[int(duel["winner"][1:])], "edits": duel["edits"],
                   "disagreement": duel["result"][0], "faults_found": duel["result"][1]}
    t = tuple(tuple(w) for w in net["triple"])
    line = line_arm(case, x, y, t, values(net), probe, seed)
    after_line = line["rounds"][-1]["disagreement"] if line["flat"] else net["disagreement"]
    kept = duel is None or duel["winner"] == "pick"
    result = after_line if kept else net["disagreement"]
    path.write_text(json.dumps({
        "seed": seed, "faults": case["faults"], "pick": {key: pick[key] for key in pick if key != "edits"},
        "pick_wrong": missed, "search": "keep-one" if missed else "keep-two", "searched": len(ts), "perfect": found,
        "true_perfect": any(len(ev.t3.hits(tuple(tuple(w) for w in p), case["faults"], case["width"])) == 3
                            for p in found),
        "duel": None if duel is None else {key: duel[key] for key in ("winner", "asked", "correct", "rounds")},
        "chosen": net, "line": {"flat": line["flat"], "after": after_line, "applied": kept and line["flat"]},
        "result": [result, net["faults_found"]], "every_winner": [after_line, net["faults_found"]],
        "search_seconds": round(search_seconds, 1), "seconds": round(time.perf_counter() - start, 1)}) + "\n")
    return seed


def tally(d):
    return {"under_0.01%": sum(v[0] < LINE for v in d.values()), "under_0.1%": sum(v[0] < 10 * LINE for v in d.values()),
            "exact": sum(v[1] == 3 for v in d.values()), "median": float(np.median([v[0] for v in d.values()])),
            "over_line": {str(s): v[0] for s, v in d.items() if v[0] >= LINE}}


def moves(seeds, a, b):
    return {"closer": sorted(s for s in seeds if b[s][0] < a[s][0]),
            "further": sorted(s for s in seeds if b[s][0] > a[s][0]),
            "gained": sorted(s for s in seeds if b[s][0] < LINE <= a[s][0]),
            "lost": sorted(s for s in seeds if a[s][0] < LINE <= b[s][0]),
            "gained_exact": sorted(s for s in seeds if b[s][1] == 3 != a[s][1]),
            "lost_exact": sorted(s for s in seeds if a[s][1] == 3 != b[s][1])}


def score(folder):
    seeds = list(SEEDS)
    first = {s: json.loads((folder / f"case-{s}.json").read_text()) for s in seeds}
    rows = {s: json.loads((folder / f"case-{s}.repair.json").read_text()) for s in seeds}
    arms = {
        "repair65": {s: (first[s]["candidates"]["repair65"]["disagreement"],
                         first[s]["candidates"]["repair65"]["faults_found"]) for s in seeds},
        "pick": {s: (rows[s]["pick"]["disagreement"], rows[s]["pick"]["faults_found"]) for s in seeds},
        "duel": {s: (rows[s]["chosen"]["disagreement"], rows[s]["chosen"]["faults_found"]) for s in seeds},
        "every_winner": {s: tuple(rows[s]["every_winner"]) for s in seeds},
        "result": {s: tuple(rows[s]["result"]) for s in seeds}}
    out = {"seeds": seeds, **{name: tally(d) for name, d in arms.items()},
           "result_vs_pick": moves(seeds, arms["pick"], arms["result"]),
           "result_vs_every_winner": moves(seeds, arms["every_winner"], arms["result"]),
           "keep_one": sorted(s for s in seeds if rows[s]["search"] == "keep-one"),
           "with_candidates": sorted(s for s in seeds if rows[s]["perfect"]),
           "true_perfect": sorted(s for s in seeds if rows[s]["true_perfect"]),
           "took_other": sorted(s for s in seeds if rows[s]["duel"] and rows[s]["duel"]["winner"] != "pick"),
           "flat": sorted(s for s in seeds if rows[s]["line"]["flat"]),
           "line_applied": sorted(s for s in seeds if rows[s]["line"]["applied"]),
           "labels_asked": {"first_part": sum(r["without"]["asked"] for r in first.values())
                            + sum(r["two_round"]["asked"] for r in first.values()
                                  if r["two_round"]["split_inputs"] >= fl.K),
                            "duel": sum(r["duel"]["asked"] for r in rows.values() if r["duel"]),
                            "line": sum(32 for r in rows.values() if r["line"]["applied"])},
           "triples_fitted": sum(r["searched"] for r in rows.values()),
           "core_hours": {"first_part": sum(r["seconds_65"] + r["seconds_71"] for r in first.values()) / 3600,
                          "keep_one_search": sum(r["search_seconds"] for r in rows.values()
                                                 if r["search"] == "keep-one") / 3600,
                          "keep_two_search": sum(r["search_seconds"] for r in rows.values()
                                                 if r["search"] == "keep-two") / 3600,
                          "duel_and_line": sum(r["seconds"] - r["search_seconds"] for r in rows.values()) / 3600},
           "per_case": {str(s): {name: arms[name][s] for name in arms} | {
               "search": rows[s]["search"], "candidates": len(rows[s]["perfect"]),
               "winner": rows[s]["duel"]["winner"] if rows[s]["duel"] else None} for s in seeds}}
    return rows, out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--repair", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    k.SOURCES[700] = OUT
    if a.score:
        rows, result = score(a.score)
        (OUT / "fresh-end.json").write_text(json.dumps([rows[s] for s in SEEDS]) + "\n")
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({key: v for key, v in result.items() if key not in ("per_case", "seeds")}, indent=1))
        return
    k.cr.install("columns")
    k.br.install()
    if a.repair:
        prepare(a.repair)
        pfit.install()
        seeds = a.only or list(SEEDS)
        jobs = [(s, pick_of(json.loads((a.repair / f"case-{s}.json").read_text())), a.repair) for s in seeds]
        def gated_first(job):  # the long keep-one searches start first
            case, x, y = k.load(job[0])
            return 0 if k.wrong(case, job[1]["edits"], x, y) else 1
        jobs.sort(key=gated_first)
        with Pool(a.workers) as pool:  # forked after the installs above
            for s in pool.imap_unordered(repair, jobs):
                print("done", s, time.strftime("%H:%M:%S"), flush=True)
        print("finished", flush=True)
        return
    ev.SAVE_EDITS = True
    a.output.mkdir(parents=True, exist_ok=True)
    for seed in list(a.only or SEEDS)[a.part::a.parts]:
        start = time.time()
        first_part(seed, a.output)
        print("done", seed, round(time.time() - start), time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
