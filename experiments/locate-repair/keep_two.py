"""Search the triples that keep two of the pick's weights on every case (roadmap #89).

#88's refit duel fixed 611, but only because #87 ran #84's search there by hand: 611's pick fits every one of #65's
final labels, so the gate (a missed label) never fires, and #84's search costs about 0.8 core-hours a case. This runs
a search cheap enough for every case, then the duel and the flat-line step, on all 200 of #79's, #80's and #83's
cases:

- where the pick misses a label (#84's 27 cases), the candidates are #85's perfect fits from #84's search, as in #88;
- elsewhere, fit every triple that keeps two of the pick's three weights and changes any third parameter (238 a case)
  with #85's fitter on #65's final labels, and keep those that fit every label;
- run #88's refit duel on the pick and the candidates (nothing to do where there are none);
- run #82's flat-line step on the duel's winner, as #86 did.

The result was fixed in advance as the network after both steps.

    python experiments/locate-repair/keep_two.py --output runs/t89 --workers 4
    python experiments/locate-repair/keep_two.py --score runs/t89
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import flat_stack as fs
import keep_one as k
import perfect_fit as pfit
import refit_duel as rd
from value_labels import line_arm
from wide_compare import EVIDENCE

OUT = EVIDENCE / "keep-two"
LINE = k.LINE


def triples(case, winner):
    """Every triple that keeps two of the winner's weights, in a fixed order."""
    params = [(n, i) for n in k.NAMES for i in range(case["arrays"][n].size)]
    order = {p: j for j, p in enumerate(params)}
    w = [tuple(p) for p in winner]
    seen = set()
    for a in range(3):
        for b in range(a + 1, 3):
            for p in params:
                if p not in (w[a], w[b]):
                    seen.add(tuple(sorted((w[a], w[b], p), key=order.get)))
    return sorted(seen, key=lambda t: [order[p] for p in t])


def searched():
    return {r["seed"]: [p["triple"] for p in r["perfect"]]
            for r in json.loads((EVIDENCE / "perfect-fit" / "perfect-fit.json").read_text())}


def run_seed(args):
    seed, pick, earlier, out = args
    path = out / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.perf_counter()
    case, x, y = k.load(seed)
    if earlier is not None:
        found, search, count = earlier, "keep-one (#85)", None
    else:
        ts = triples(case, pick["winner"])
        found = [[list(p) for p in t] for t in ts if k.wt.fit(case, x, y, t)[0][0] == 1.0]
        search, count = "keep-two", len(ts)
    search_seconds = time.perf_counter() - start
    duel_dir = out / "duel"
    duel_dir.mkdir(exist_ok=True)
    if found:
        rd.run_seed((seed, {"pick": pick, "triples": found, "before": [pick["disagreement"], pick["faults_found"]]},
                     duel_dir))
        duel = json.loads((duel_dir / f"case-{seed}.json").read_text())
        name = duel["winner"]
        triple = pick["winner"] if name == "pick" else found[int(name[1:])]
        net = {"triple": [list(w) for w in triple], "edits": pick["edits"] if name == "pick" else duel["edits"],
               "disagreement": duel["result"][0], "faults_found": duel["result"][1]}
    else:
        duel = None
        net = {"triple": [list(w) for w in pick["winner"]], "edits": pick["edits"],
               "disagreement": pick["disagreement"], "faults_found": pick["faults_found"]}
    probe = k.ev.bs.make_probe(case, k.ev.bs.PROBE)
    t = tuple(tuple(w) for w in net["triple"])
    line = line_arm(case, x, y, t, fs.values(net), probe, seed)
    after = line["rounds"][-1]["disagreement"] if line["flat"] else net["disagreement"]
    path.write_text(json.dumps({
        "seed": seed, "search": search, "searched": count, "perfect": found,
        "true_perfect": any(len(k.ev.t3.hits(tuple(tuple(w) for w in p), case["faults"], case["width"])) == 3
                            for p in found),
        "pick": {key: pick[key] for key in ("winner", "disagreement", "faults_found")},
        "duel": None if duel is None else {key: duel[key] for key in ("winner", "asked", "correct", "rounds")},
        "chosen": net, "line": {"flat": line["flat"], "after": after}, "result": [after, net["faults_found"]],
        "search_seconds": round(search_seconds, 1), "seconds": round(time.perf_counter() - start, 1)}) + "\n")
    return seed


def score(folder):
    rows = {r["seed"]: r for r in (json.loads(p.read_text()) for p in sorted(folder.glob("case-*.json")))}
    seeds = sorted(rows)
    flat = {r["seed"]: r for r in json.loads((EVIDENCE / "flat-stack" / "flat-stack.json").read_text())}
    stack = {s: (fs.result(flat[s]), flat[s]["chosen"]["faults_found"]) for s in seeds}
    pick = {s: (rows[s]["pick"]["disagreement"], rows[s]["pick"]["faults_found"]) for s in seeds}
    duel = {s: (rows[s]["chosen"]["disagreement"], rows[s]["chosen"]["faults_found"]) for s in seeds}
    after = {s: tuple(rows[s]["result"]) for s in seeds}
    tally = lambda d: {"under_0.01%": sum(v[0] < LINE for v in d.values()),
                       "under_0.1%": sum(v[0] < 10 * LINE for v in d.values()),
                       "exact": sum(v[1] == 3 for v in d.values()), "median": float(np.median([v[0] for v in d.values()])),
                       "over_line": {str(s): v[0] for s, v in d.items() if v[0] >= LINE}}
    moves = lambda a, b: {"closer": sorted(s for s in seeds if b[s][0] < a[s][0]),
                          "further": sorted(s for s in seeds if b[s][0] > a[s][0]),
                          "gained": sorted(s for s in seeds if b[s][0] < LINE <= a[s][0]),
                          "lost": sorted(s for s in seeds if a[s][0] < LINE <= b[s][0]),
                          "gained_exact": sorted(s for s in seeds if b[s][1] == 3 != a[s][1]),
                          "lost_exact": sorted(s for s in seeds if a[s][1] == 3 != b[s][1])}
    two = [s for s in seeds if rows[s]["search"] == "keep-two"]
    out = {"seeds": seeds, "pick": tally(pick), "duel": tally(duel), "result": tally(after), "after_86": tally(stack),
           "vs_pick": moves(pick, after), "vs_86": moves(stack, after),
           "keep_two_cases": len(two),
           "keep_two_with_candidates": sorted(s for s in two if rows[s]["perfect"]),
           "keep_two_true_perfect": sorted(s for s in two if rows[s]["true_perfect"]),
           "pick_two_broken": sorted(s for s in two if rows[s]["pick"]["faults_found"] == 2),
           "took_other": sorted(s for s in seeds if rows[s]["duel"] and rows[s]["duel"]["winner"] != "pick"),
           "flat": sorted(s for s in seeds if rows[s]["line"]["flat"]),
           "labels_asked": {"duel": sum(r["duel"]["asked"] for r in rows.values() if r["duel"]),
                            "line": sum(32 for r in rows.values() if r["line"]["flat"])},
           "triples_fitted": sum(r["searched"] or 0 for r in rows.values()),
           "core_hours": sum(r["seconds"] for r in rows.values()) / 3600,
           "core_hours_search": sum(r["search_seconds"] for r in rows.values() if r["search"] == "keep-two") / 3600,
           "per_case": {str(s): {"pick": pick[s], "after_86": stack[s], "result": after[s],
                                 "candidates": len(rows[s]["perfect"]),
                                 "winner": rows[s]["duel"]["winner"] if rows[s]["duel"] else None} for s in seeds}}
    return rows, out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.score:
        rows, result = score(a.score)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "keep-two.json").write_text(json.dumps([rows[s] for s in sorted(rows)]) + "\n")
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k_: v for k_, v in result.items() if k_ not in ("per_case", "seeds")}, indent=1))
        return
    k.cr.install("columns")
    k.br.install()
    pfit.install()
    a.output.mkdir(parents=True, exist_ok=True)
    picks, earlier = k.picks(), searched()
    seeds = a.only or sorted(picks, key=lambda s: (s not in (611, 645), s))
    jobs = [(s, picks[s], earlier.get(s), a.output) for s in seeds]
    with Pool(a.workers) as pool:  # forked after the installs above
        for s in pool.imap_unordered(run_seed, jobs):
            print("done", s, time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
