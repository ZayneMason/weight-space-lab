"""When the chosen fit misses a label, search the triples that keep one of its weights (roadmap #84).

In #83's seed 641 the true broken weights fit all 1,096 of #65's final labels and land at 0.0039%, while every one of
#71's candidates edits one wrong triple that gets 7 of those labels wrong. This script takes #81's pick on #79's 100
cases and #80's 50, and the gated pick on #83's 50, counts the labels of #65's final set its network gets wrong, and
in every case where that count is at least one:

- fits every triple that keeps one of the pick's three weights and adds any two of the other 81 parameters (each
  fitted on #65's final labels, as #71's candidates were not), about 9,700 triples per case;
- keeps the triples that fit every label, and scores each one's fitted network on the probe;
- runs #71's disagree chooser (up to 32 labels where the networks disagree, from #71's pool) on the pick and every
  perfect fit, with the pick first so a tie keeps it.

    python experiments/locate-repair/keep_one.py --prepare runs/l83        # copy #83's final #65 labels to the evidence
    python experiments/locate-repair/keep_one.py --output runs/k84 --part 0 --parts 4   # parts 0 to 3, one per core
    python experiments/locate-repair/keep_one.py --score runs/k84
    python experiments/locate-repair/keep_one.py --gap   # where the true triple is not among the perfect fits
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
import shutil
import time

import numpy as np

import batched_refine as br
import cut_refine as cr
import early_values as ev
import pick_fixes as pf
import value_labels as vl
from rebuild_committee import labels
from wide_compare import EVIDENCE

OUT = EVIDENCE / "keep-one"
FRESH = EVIDENCE / "fresh-line"
SOURCES = {300: EVIDENCE / "fresh-repair", 400: EVIDENCE / "fresh-50", 500: EVIDENCE / "saved-65", 600: FRESH}
LINE = pf.LINE
NAMES = ("hidden_weight", "hidden_bias", "weight", "bias")
wt = pf.wt


def prepare(folder):
    """Copy #83's final #65 labels (from its run folder) into the evidence, as #80's are."""
    (FRESH / "labels").mkdir(parents=True, exist_ok=True)
    for seed in range(600, 650):
        last = json.loads((folder / f"case-{seed}.json").read_text())["stages_run"] - 1
        shutil.copy(folder / f"case-{seed}.stage{last}.npz", FRESH / "labels" / f"case-{seed}.npz")


def picks():
    """The chosen network of every case: #81's pick on #79's and #80's cases, the gated pick on #83's."""
    out = {s: {k: v[k] for k in ("pick", "winner", "edits", "disagreement", "faults_found")}
           for s, v in vl.cases().items()}
    for r in json.loads((FRESH / "fresh-line.json").read_text()):
        p = r["line_step"]["gated_pick"]
        c = r["candidates"][p]
        out[r["seed"]] = {"pick": p, "winner": c["winner"], "edits": c["edits"], "disagreement": c["disagreement"],
                          "faults_found": c["faults_found"]}
    return out


def load(seed):
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, y = labels(case, seed, SOURCES[seed // 100 * 100])
    return case, x, y


def wrong(case, edits, x, y):
    return int((ev.tf.predict(ev.tf.apply(case["arrays"], [tuple(e) for e in edits]), x) != y).sum())


def triples(case, winner):
    """Every triple that keeps one of the winner's weights, in a fixed order."""
    params = [(n, i) for n in NAMES for i in range(case["arrays"][n].size)]
    order = {p: k for k, p in enumerate(params)}
    seen = set()
    for kept in (tuple(w) for w in winner):
        for a, b in combinations([p for p in params if p != kept], 2):
            seen.add(tuple(sorted((kept, a, b), key=order.get)))
    return sorted(seen, key=lambda t: [order[p] for p in t])


def chooser(case, networks, seed):
    """#71's disagree chooser (``pick_fixes.choosers``) for any number of networks: the same pool and
    duel draw; the most correct on the duel wins, and a tie keeps the first network (the pick)."""
    rng = np.random.default_rng([seed, 71])
    rng.uniform(-2, 2, size=(pf.UNIFORM, 2))  # the uniform labels, drawn as #71 draws them and not used here
    pool = rng.uniform(-2, 2, size=(pf.POOL, 2))
    names = list(networks)
    votes = np.stack([ev.tf.predict(ev.tf.apply(case["arrays"], networks[n]), pool).astype(np.int8) for n in names])
    split = np.flatnonzero((votes != votes[0]).any(axis=0))
    duel = rng.choice(split, size=min(pf.DUEL, len(split)), replace=False) if len(split) else split
    yd = ev.tf.predict(case["teacher"], pool[duel])
    correct = {n: int((votes[k][duel] == yd).sum()) for k, n in enumerate(names)}
    best = max(correct.values())
    return {"split_inputs": int(len(split)), "asked": int(len(duel)), "correct": correct,
            "pick": next(n for n in names if correct[n] == best)}


def run_seed(seed, pick, out):
    path = out / f"case-{seed}.json"
    if path.exists():
        return
    start = time.perf_counter()
    case, x, y = load(seed)
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    found = []
    for t in triples(case, pick["winner"]):
        f = wt.fit(case, x, y, t)
        if f[0][0] == 1.0:
            found.append((t, f))
    fit_seconds = time.perf_counter() - start
    perfect = [{"triple": [list(p) for p in t], "cost": -f[0][1],
                "faults_found": len(ev.t3.hits(t, case["faults"], case["width"])),
                "disagreement": ev.aq.disagreement(case, f[1], probe), "edits": ev.edit_list(f[1])} for t, f in found]
    networks = {"pick": [tuple(e) for e in pick["edits"]], **{f"t{k}": f[1] for k, (_, f) in enumerate(found)}}
    chosen = chooser(case, networks, seed)
    path.write_text(json.dumps({
        "seed": seed, "pick": pick, "pick_wrong": wrong(case, pick["edits"], x, y), "labels": len(y),
        "searched": len(triples(case, pick["winner"])), "faults": case["faults"],
        "true_searched": any(tuple(w) in {tuple(p) for p in case["faults"]} for w in pick["winner"]),
        "perfect": perfect, "chooser": chosen, "true_perfect": any(p["faults_found"] == 3 for p in perfect),
        "fit_seconds": round(fit_seconds, 1), "seconds": round(time.perf_counter() - start, 1)}) + "\n")


def selected(all_picks):
    """Seeds whose chosen network gets at least one of #65's final labels wrong, 641 first."""
    out = []
    for s, p in sorted(all_picks.items()):
        case, x, y = load(s)
        if wrong(case, p["edits"], x, y):
            out.append(s)
    return sorted(out, key=lambda s: (s != 641, s))


def result_of(row, name):
    if name == "pick":
        return row["pick"]["disagreement"], row["pick"]["faults_found"]
    p = row["perfect"][int(name[1:])]
    return p["disagreement"], p["faults_found"]


def score(folder):
    rows = {r["seed"]: r for r in (json.loads(p.read_text()) for p in sorted(folder.glob("case-*.json")))}
    seeds = sorted(rows)
    chosen = {s: result_of(rows[s], rows[s]["chooser"]["pick"]) for s in seeds}
    before = {s: (rows[s]["pick"]["disagreement"], rows[s]["pick"]["faults_found"]) for s in seeds}
    best = {s: min([before[s]] + [(p["disagreement"], p["faults_found"]) for p in rows[s]["perfect"]]) for s in seeds}
    tally = lambda d: {"under_0.01%": sum(v[0] < LINE for v in d.values()), "exact": sum(v[1] == 3 for v in d.values()),
                       "over_line": {str(s): v[0] for s, v in d.items() if v[0] >= LINE}}
    out = {"seeds": seeds, "pick": tally(before), "search": tally(chosen), "probe_best": tally(best)}
    out["better"] = [s for s in seeds if chosen[s][0] < before[s][0]]
    out["worse"] = [s for s in seeds if chosen[s][0] > before[s][0]]
    out["took_other"] = [s for s in seeds if rows[s]["chooser"]["pick"] != "pick"]
    out["per_case"] = {str(s): {"pick_wrong": rows[s]["pick_wrong"], "perfect": len(rows[s]["perfect"]),
                                "true_searched": rows[s]["true_searched"], "true_perfect": rows[s]["true_perfect"],
                                "pick": before[s], "search": chosen[s], "split_inputs":
                                rows[s]["chooser"]["split_inputs"]} for s in seeds}
    out["true_searched"] = sum(r["true_searched"] for r in rows.values())
    out["true_perfect"] = sum(r["true_perfect"] for r in rows.values())
    out["perfect_counts"] = {"median": float(np.median([len(r["perfect"]) for r in rows.values()])),
                             "max": max(len(r["perfect"]) for r in rows.values())}
    out["labels_asked"] = sum(r["chooser"]["asked"] for r in rows.values())
    out["triples_fitted"] = sum(r["searched"] for r in rows.values())
    out["core_hours"] = sum(r["seconds"] for r in rows.values()) / 3600
    return rows, out


def gap():
    """In each case where the true triple is not among the perfect fits: how many of #65's final labels its original
    values get wrong, and how many the fitter's best values for it get wrong."""
    cr.install("columns")
    br.install()
    out = {}
    for r in json.loads((OUT / "keep-one.json").read_text()):
        if r["true_perfect"]:
            continue
        case, x, y = load(r["seed"])
        t = tuple(tuple(e) for e in case["faults"])
        original = [(n, i, float(case["teacher"][n].flat[i])) for n, i in t]
        f = wt.fit(case, x, y, t)
        out[str(r["seed"])] = {"labels": len(y), "original_values_wrong": wrong(case, original, x, y),
                               "fitted_wrong": int(round((1 - f[0][0]) * len(y))),
                               "pick_wrong": r["pick_wrong"], "pick_faults_found": r["pick"]["faults_found"]}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prepare", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    p.add_argument("--gap", action="store_true")
    a = p.parse_args()
    if a.gap:
        result = gap()
        (OUT / "gap.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    if a.prepare:
        prepare(a.prepare)
        return
    if a.score:
        rows, result = score(a.score)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "keep-one.json").write_text(json.dumps([rows[s] for s in sorted(rows)]) + "\n")
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=1))
        return
    cr.install("columns")
    br.install()
    a.output.mkdir(parents=True, exist_ok=True)
    all_picks = picks()
    seeds = a.only or selected(all_picks)
    (a.output / "selected.json").write_text(json.dumps(seeds) + "\n")
    for seed in seeds[a.part::a.parts]:
        start = time.time()
        run_seed(seed, all_picks[seed], a.output)
        print("done", seed, round(time.time() - start), time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
