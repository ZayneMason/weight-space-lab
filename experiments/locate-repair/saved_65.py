"""Save #65's own repair and choose among the saved networks (roadmap #80).

#79 added #65's repair to #71's disagree chooser as a fifth candidate, but #65's stage saved only its winning triple
and labels, so #79 rebuilt the repair by refitting that triple. The rebuild landed within 0.001% of #65's recorded
disagreement in 83 of 100 cases, and far off in two (415 and 418). This script does two things:

1. ``--rescore``: on #79's 100 cases, from the networks #79 saved, scores a two-round chooser. Round one is #71's
   disagree chooser on #71's own candidates, as it is. Round two is a duel between its pick and #65's repair: up to
   32 labels where the two disagree, from a fresh pool, and #65's repair is taken only when it gets more of them
   right. The point is seed 319, where #79's one-round chooser spent all its labels ruling out #65's repair.
2. The whole cheap repair (columns and the batched committee fits, as #78) on 50 fresh seeds (500 to 549), with
   ``early_values.SAVE_EDITS`` on, so #65's stage keeps the network it ends with. #71's stage then runs with both
   the saved repair and #79's rebuild as candidates, and every chooser is scored on the same pool.

    python experiments/locate-repair/saved_65.py --rescore
    python experiments/locate-repair/saved_65.py --output runs/s80 --part 0 --parts 4   # one per core
    python experiments/locate-repair/saved_65.py --score runs/s80
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import batched_refine as br
import cut_refine as cr
import early_values as ev
import pick_fixes as pf
import tied_labels
from keep_65 import FIRST, LAST, choose
from wide_compare import EVIDENCE

OUT = EVIDENCE / "saved-65"
KEEP = EVIDENCE / "keep-65" / "keep-65.json"
SEEDS = tuple(range(500, 550))
LINE = pf.LINE
SEVENTY_ONE = ("pick68", "restarts", "boundary", "bigger")


def as_edits(rows):
    return [(n, int(i), float(v)) for n, i, v in rows]


def duel(case, first, edits_first, edits_65, seed):
    """Round two: the first round's pick against #65's repair, on up to 32 labels where the two disagree."""
    rng = np.random.default_rng([seed, 80])
    pool = rng.uniform(-2, 2, size=(pf.POOL, 2))
    a = ev.tf.predict(ev.tf.apply(case["arrays"], edits_first), pool)
    b = ev.tf.predict(ev.tf.apply(case["arrays"], edits_65), pool)
    split = np.flatnonzero(a != b)
    asked = rng.choice(split, size=min(pf.DUEL, len(split)), replace=False) if len(split) else split
    y = ev.tf.predict(case["teacher"], pool[asked])
    correct = {first: int((a[asked] == y).sum()), "repair65": int((b[asked] == y).sum())}
    return {"split_inputs": int(len(split)), "asked": int(len(asked)), "correct": correct,
            "pick": "repair65" if correct["repair65"] > correct[first] else first}


def with_65(case, networks, seed):
    """#79's one-round chooser with ``repair65`` among the candidates, both tie rules."""
    order, pf.ORDER = pf.ORDER, LAST  # pf.choose looks names up in ORDER (see keep_65.py)
    try:
        out = pf.choosers(case, networks, seed)["disagree"]
    finally:
        pf.ORDER = order
    out["pick_last"], out["pick_first"] = choose(out["correct"], LAST), choose(out["correct"], FIRST)
    del out["pick"]
    return out


def rescore():
    """The two-round chooser on #79's cases, from the networks #79 saved."""
    rows = json.loads(KEEP.read_text())
    out = []
    for row in rows:
        seed = row["seed"]
        case = ev.rs.resize(ev.t3.corrupt(16, seed))
        first = row["without"]["pick"]
        c = row["candidates"]
        d = duel(case, first, as_edits(c[first]["edits"]), as_edits(c["repair65"]["edits"]), seed)
        out.append({"seed": seed, "first_round": first, "duel": d,
                    "disagreement": {n: c[n]["disagreement"] for n in c},
                    "faults_found": {n: c[n]["faults_found"] for n in c},
                    "picks": {"without": first, "with_last": row["with"]["pick_last"], "two_round": d["pick"]},
                    "asked": {"without": row["without"]["asked"], "with_last": row["with"]["asked"],
                              "two_round": row["without"]["asked"] + d["asked"]}})
    return out


def final_65(row):
    """#65's repair as #72 and #78 scored it (the region arm, with its middle when the middle fits as many labels),
    with the edits this run saved for it."""
    e = row["extra"]["region"]
    m = e["middle"]
    chosen = m if m["accuracy"] >= e["practice_accuracy"] else e
    return chosen["edits"], chosen["disagreement"], e["faults_found"], tuple(tuple(t) for t in e["winner"])


def run_seed(seed, out):
    path = out / f"case-{seed}.json"
    if path.exists():
        return
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    start = time.perf_counter()
    r65 = ev.two_step(case, 2, saved=lambda k: out / f"case-{seed}.stage{k}.json", middle=True, extra=True)
    seconds_65 = r65["seconds"]  # summed from each stage's own clock, so a resumed case keeps its first run's time
    clock = time.perf_counter()
    last = r65["stages_run"] - 1
    z = np.load(out / f"case-{seed}.stage{last}.npz")
    x0 = np.concatenate([z["x"], z["added_region"]])
    y0 = np.concatenate([z["y"], ev.tf.predict(case["teacher"], z["added_region"])])
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    saved, recorded, found_65, t65 = final_65(r65)
    options = ev.TWO_STEP[last]
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
    built["rebuilt65"] = (t65, pf.wt.fit(case, x0, y0, t65), x0, y0)
    rows, networks = {}, {}
    for n, (t, f, cx, cy) in built.items():
        edits, used = pf.final_edits(case, cx, cy, t, f)
        networks[n] = edits
        rows[n] = {"winner": [list(e) for e in t], "middle_used": used, "practice_accuracy": float(f[0][0]),
                   "faults_found": len(ev.t3.hits(t, case["faults"], case["width"])),
                   "disagreement": ev.aq.disagreement(case, edits, probe), "edits": ev.edit_list(edits)}
    networks["repair65"] = as_edits(saved)
    rows["repair65"] = {"winner": [list(e) for e in t65], "faults_found": found_65, "recorded_disagreement": recorded,
                        "disagreement": ev.aq.disagreement(case, networks["repair65"], probe), "edits": saved}
    own = {n: networks[n] for n in SEVENTY_ONE if n in networks}
    without = pf.choosers(case, own, seed)["disagree"]
    first = without["pick"]
    row = {"seed": seed, "gate": gate, "stages_run": r65["stages_run"], "candidates": rows, "without": without,
           "with_saved": with_65(case, {**own, "repair65": networks["repair65"]}, seed),
           "with_rebuilt": with_65(case, {**own, "repair65": networks["rebuilt65"]}, seed),
           "two_round": duel(case, first, networks[first], networks["repair65"], seed),
           "seconds_65": seconds_65, "seconds_71": round(time.perf_counter() - clock, 1),
           "wall_seconds": round(time.perf_counter() - start, 1)}
    path.write_text(json.dumps(row) + "\n")


def tally(rows, picks, line=LINE):
    """Under the line, exact and gained/lost/worse against ``without`` for each set of picks."""
    d = lambda s, n: rows[s]["disagreement"][n]
    exact = lambda s, n: rows[s]["faults_found"][n] == 3
    seeds = sorted(rows)
    base = picks["without"]
    out = {}
    for name, p in picks.items():
        out[name] = {"under_0.01%": sum(d(s, p[s]) < line for s in seeds),
                     "under_0.1%": sum(d(s, p[s]) < 10 * line for s in seeds),
                     "exact": sum(exact(s, p[s]) for s in seeds),
                     "took_65": sorted(s for s in seeds if p[s] == "repair65"),
                     "gained": sorted(s for s in seeds if d(s, p[s]) < line <= d(s, base[s])),
                     "lost": sorted(s for s in seeds if d(s, base[s]) < line <= d(s, p[s])),
                     "better": sorted(s for s in seeds if d(s, p[s]) < d(s, base[s])),
                     "worse": sorted(s for s in seeds if d(s, p[s]) > d(s, base[s])),
                     "over_line": {str(s): d(s, p[s]) for s in seeds if d(s, p[s]) >= line}}
    return out


def score_rescore(rows):
    by = {r["seed"]: r for r in rows}
    picks = {k: {s: by[s]["picks"][k] for s in by} for k in ("without", "with_last", "two_round")}
    out = {"seeds": sorted(by), **tally(by, picks)}
    out["labels_asked"] = {k: sum(by[s]["asked"][k] for s in by) for k in picks}
    out["duel_asked"] = sum(by[s]["duel"]["asked"] for s in by)
    out["duel_no_split"] = sorted(s for s in by if by[s]["duel"]["split_inputs"] == 0)
    return out


def score(folder):
    raw = {s: json.loads((folder / f"case-{s}.json").read_text()) for s in SEEDS}
    rows = {s: {"disagreement": {n: c["disagreement"] for n, c in r["candidates"].items()},
                "faults_found": {n: c["faults_found"] for n, c in r["candidates"].items()}} for s, r in raw.items()}
    for s, r in raw.items():  # #79's chooser with the rebuild scores the rebuild's network
        rows[s]["disagreement"]["rebuilt"] = rows[s]["disagreement"]["rebuilt65"]
        rows[s]["faults_found"]["rebuilt"] = rows[s]["faults_found"]["rebuilt65"]
    pick = lambda s, k: raw[s][k]["pick_last"]
    picks = {"without": {s: r["without"]["pick"] for s, r in raw.items()},
             "with_saved": {s: pick(s, "with_saved") for s in raw},
             "with_saved_first": {s: raw[s]["with_saved"]["pick_first"] for s in raw},
             "with_rebuilt": {s: "rebuilt" if pick(s, "with_rebuilt") == "repair65" else pick(s, "with_rebuilt")
                              for s in raw},
             "two_round": {s: r["two_round"]["pick"] for s, r in raw.items()},
             "repair65": {s: "repair65" for s in raw},
             "probe_best": {s: min((n for n in rows[s]["disagreement"] if n not in ("rebuilt65", "rebuilt")),
                                   key=lambda n: (rows[s]["disagreement"][n], LAST.index(n))) for s in raw}}
    out = {"seeds": list(SEEDS), **tally(rows, picks)}
    rep = {s: raw[s]["candidates"]["repair65"] for s in raw}
    out["saved_65"] = {"matches_recorded": sum(abs(r["disagreement"] - r["recorded_disagreement"]) < 1e-12
                                               for r in rep.values()),
                       "under_0.01%": sum(r["disagreement"] < LINE for r in rep.values()),
                       "exact": sum(r["faults_found"] == 3 for r in rep.values())}
    reb = {s: raw[s]["candidates"]["rebuilt65"] for s in raw}
    out["rebuilt_65"] = {"within_0.001%": sum(abs(reb[s]["disagreement"] - rep[s]["disagreement"]) <= 1e-5 for s in raw),
                         "same_side_of_line": sum((reb[s]["disagreement"] < LINE) == (rep[s]["disagreement"] < LINE)
                                                  for s in raw),
                         "differs": {str(s): [rep[s]["disagreement"], reb[s]["disagreement"]] for s in raw
                                     if abs(reb[s]["disagreement"] - rep[s]["disagreement"]) > 1e-5}}
    out["labels_asked"] = {"without": sum(r["without"]["asked"] for r in raw.values()),
                           "with_saved": sum(r["with_saved"]["asked"] for r in raw.values()),
                           "two_round": sum(r["without"]["asked"] + r["two_round"]["asked"] for r in raw.values())}
    out["core_hours"] = sum(r["seconds_65"] + r["seconds_71"] for r in raw.values()) / 3600
    out["core_hours_stage_65"] = sum(r["seconds_65"] for r in raw.values()) / 3600
    out["stages_run"] = {str(k): sum(r["stages_run"] == k for r in raw.values()) for k in (1, 2, 3)}
    out["bigger_ran"] = sorted(s for s, r in raw.items() if r["gate"])
    return raw, out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.rescore:
        rows = rescore()
        (OUT / "rescore-79.json").write_text(json.dumps(rows) + "\n")
        result = score_rescore(rows)
        (OUT / "rescore-79-check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    if a.score:
        raw, result = score(a.score)
        (OUT / "saved-65.json").write_text(json.dumps([raw[s] for s in sorted(raw)]) + "\n")
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    cr.install("columns")
    br.install()
    ev.SAVE_EDITS = True
    a.output.mkdir(parents=True, exist_ok=True)
    for seed in list(a.only or SEEDS)[a.part::a.parts]:
        start = time.time()
        run_seed(seed, a.output)
        print("done", seed, round(time.time() - start), time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
