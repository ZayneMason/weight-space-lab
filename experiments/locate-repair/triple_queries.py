"""Targeted queries on three broken weights (roadmap #28)

Roadmap #27's triple search reached the true triple in 24 of 25 cases at H = 16 but picked it in
only 5, because several triples fit the practice set perfectly. This experiment takes #27's
coarse top 40 triples as the committee and runs #22's eight rounds of 8 targeted queries
(20,000 fresh inputs per round), re-fitting every triple finely each round.

    python experiments/locate-repair/triple_queries.py --seeds 25 --output runs/triple-queries
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


t3 = _load("three_faults", "three_faults.py")
js, sc, tf, lr, cr = t3.js, t3.sc, t3.tf, t3.lr, t3.cr
aq = cr.aq

COMMITTEE = 40


def committee(case):
    px, py, _ = lr.split(case["practice"])
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    order = js.joint_order(js.joint_scores(case, px, py, pairs))
    triples = t3.triples_from([pairs[k] for k in order], lr.entries(case["arrays"]))
    coarse = {t: t3.fit_triple(case, px, py, t, t3.COARSE)[0] for t in triples}
    return sorted(triples, key=coarse.get, reverse=True)[:COMMITTEE]


def run(case, candidates, silent):
    width, faults = case["width"], case["faults"]
    visible = [k for k in range(len(faults)) if k not in silent]
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    rng = np.random.default_rng([width, case["seed"], 404])
    x, y, history = px, py, []
    for round_ in range(aq.ROUNDS + 1):
        fits = {t: t3.fine_fit(case, x, y, t) for t in candidates}
        winner = max(fits, key=lambda t: fits[t][0])
        score, edits = fits[winner]
        found = t3.hits(winner, faults, width)
        true_scores = [s[0] for t, (s, _) in fits.items() if len(t3.hits(t, faults, width)) == 3]
        history.append(dict(tf.outcome(case, edits, fx, fy, frepair, inputs, teacher_labels),
                            queries=len(x) - len(px), faults_found=len(found),
                            found_visible=all(k in found for k in visible), practice_accuracy=score[0],
                            tied_best=sum(s[0] == score[0] for s, _ in fits.values()),
                            true_triple_accuracy=max(true_scores) if true_scores else None,
                            winner=[list(e) for e in winner]))
        if round_ == aq.ROUNDS:
            break
        pool = rng.uniform(-2, 2, size=(aq.POOL, 2))
        picked, history[-1]["split_of_queries"] = aq.choose_queries(case, fits, pool, rng, "targeted")
        x = np.concatenate([x, picked])
        y = np.concatenate([y, tf.predict(case["teacher"], picked)])
    return history


def analyse(case):
    candidates = committee(case)
    silent = cr.silent_faults(case)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "silent_faults": silent,
            "committee_has_true_triple": any(len(t3.hits(t, case["faults"], case["width"])) == 3 for t in candidates),
            "targeted": run(case, candidates, silent)}


def summarize(rows):
    out = {"cases": len(rows), "silent_cases": sum(bool(r["silent_faults"]) for r in rows),
           "committee_has_true_triple": sum(r["committee_has_true_triple"] for r in rows), "targeted": []}
    for k in range(len(rows[0]["targeted"])):
        steps = [r["targeted"][k] for r in rows]
        out["targeted"].append({"queries": steps[0]["queries"],
                                "found_all_three": sum(s["faults_found"] == 3 for s in steps),
                                "found_visible": sum(s["found_visible"] for s in steps),
                                "success": sum(s["success"] for s in steps),
                                "damage_undone_median": float(np.median([s["damage_undone"] for s in steps]))})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            case = t3.corrupt(args.width, seed)
            if case is not None:
                rows.append(analyse(case))
                last = rows[-1]["targeted"][-1]
                print(seed, rows[-1]["committee_has_true_triple"], last["faults_found"], last["found_visible"],
                      last["success"], flush=True)
    report = {"schema": "weight-space-triple-queries-v1", "committee": COMMITTEE, "pool": aq.POOL,
              "rounds": aq.ROUNDS, "per_round": aq.PER_ROUND, "summary": summarize(rows), "cases": rows}
    (args.output / "triple-queries.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]["targeted"][-1]))


if __name__ == "__main__":
    main()
