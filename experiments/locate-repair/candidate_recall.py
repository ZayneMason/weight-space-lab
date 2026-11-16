"""Which candidate rule keeps the broken pair? (roadmap #25)

Roadmap #24 kept 40 candidates per case: #23's coverage screen, a coarse 61 x 61 joint fit on its
top 1,000 pairs, and the best 40 by practice accuracy, then smallest change. The broken pair was
among them in 13 of 25 cases at H = 32. This script coarse-fits the screen's top 3,000 pairs and
records each pair's coarse accuracy and change, so that candidate rules can be compared offline:
how many pairs to fit, how many to keep, and how to order them.

    python experiments/locate-repair/candidate_recall.py --seeds 25 --output runs/candidate-recall

Phase 2 (``--queries``) reads that file, keeps 100 candidates by the best rule found there, and
runs #22's targeted queries on them. It also marks *silent* faults: corrupting the entry alone
changes no practice label, so no behavioural search can see it.

    python experiments/locate-repair/candidate_recall.py --queries runs/candidate-recall/candidate-recall.json \
        --output runs/candidate-queries
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


sc = _load("pair_screen", "pair_screen.py")
aq = _load("active_queries", "active_queries.py")
aq.POOL = 20_000
ps, tf, lr = sc.ps, sc.tf, sc.lr

FIT = 3000
TOPS = (1000, 3000)
KEEPS = (40, 100, 200)
TOLERANCES = (0.0, 0.002, 0.005)


def order(acc, cost, screen_rank, rule, tolerance=0.0):
    """Indices of fitted pairs, best first.

    ``min-change``: accuracy, then smallest change (#24's rule). ``screen``: accuracy, then screen
    rank. A tolerance treats coarse accuracies within it of the best as equal.
    """
    acc = np.asarray(acc, float)
    level = np.where(acc >= acc.max() - tolerance - 1e-12, acc.max(), acc)
    second = np.asarray(cost, float) if rule == "min-change" else np.asarray(screen_rank, float)
    return np.lexsort((second, -level))


def recall(row, top, keep, rule, tolerance):
    n = min(top, len(row["accuracy"]))
    best = order(row["accuracy"][:n], row["cost"][:n], range(n), rule, tolerance)[:keep]
    return bool(set(best.tolist()) & set(row["true_indices"]))


def analyse(case, fit=FIT):
    px, py, _ = lr.split(case["practice"])
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    true = [k for k, p in enumerate(pairs) if ps.found(p, case["faults"], case["width"]) == 2]
    accuracy, cost = [], []
    for pair in pairs[:fit]:
        a, c = ps.pair_fit(case, px, py, pair)
        accuracy.append(round(float(a), 6))
        cost.append(round(-float(c), 4))
    fine = {}
    for k in true[:2]:
        (fa, fc), _ = ps.fine_fit(case, px, py, pairs[k])
        fine[k] = [round(float(fa), 6), round(-float(fc), 4)]
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "true_pair_rank": true[0],
            "true_indices": [k for k in true if k < fit], "true_fine_fit": fine,
            "accuracy": accuracy, "cost": cost}


QUERY_FIT, QUERY_KEEP, QUERY_RULE = 3000, 100, "min-change"


def silent_faults(case):
    """Faults that change no practice label when corrupted alone."""
    px, py, _ = lr.split(case["practice"])
    out = []
    for k, (name, index) in enumerate(case["faults"]):
        arrays = {n: a.copy() for n, a in case["teacher"].items()}
        arrays[name].flat[index] = case["arrays"][name].flat[index]
        if (tf.predict(arrays, px) == py).all():
            out.append(k)
    return out


def found_visible(winner, faults, width, silent):
    hit = {k for e in winner for k in tf.matches(e, faults, width)}
    return all(k in hit for k in range(len(faults)) if k not in silent)


def query_case(row, fit=QUERY_FIT, keep=QUERY_KEEP, rule=QUERY_RULE):
    case = tf.corrupt(row["width"], row["seed"])
    px, py, _ = lr.split(case["practice"])
    pairs = sc.ranked_pairs(case["arrays"], px, py)[:fit]
    n = min(fit, len(row["accuracy"]))
    pool = [pairs[k] for k in order(row["accuracy"][:n], row["cost"][:n], range(n), rule)[:keep]]
    silent = silent_faults(case)
    history = aq.run(case, pool, "targeted")
    for step in history:
        step["found_visible"] = found_visible(step["winner"], case["faults"], case["width"], silent)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "silent_faults": silent,
            "candidates_have_true_pair": any(ps.found(p, case["faults"], case["width"]) == 2 for p in pool),
            "targeted": history}


def summarize_queries(rows):
    out = {"cases": len(rows), "silent_cases": sum(bool(r["silent_faults"]) for r in rows),
           "candidates_have_true_pair": sum(r["candidates_have_true_pair"] for r in rows), "targeted": []}
    for k in range(len(rows[0]["targeted"])):
        steps = [r["targeted"][k] for r in rows]
        out["targeted"].append({"queries": steps[0]["queries"],
                                "found_pair": sum(s["faults_found"] == 2 for s in steps),
                                "found_visible": sum(s["found_visible"] for s in steps),
                                "success": sum(s["success"] for s in steps),
                                "damage_undone_median": float(np.median([s["damage_undone"] for s in steps]))})
    return out


def summarize(rows):
    out = {"cases": len(rows), "in_screen": {str(t): sum(r["true_pair_rank"] < t for r in rows) for t in TOPS}}
    table = []
    for top in TOPS:
        for keep in KEEPS:
            for rule in ("min-change", "screen"):
                for tol in TOLERANCES:
                    table.append({"fit": top, "keep": keep, "rule": rule, "tolerance": tol,
                                  "recall": sum(recall(r, top, keep, rule, tol) for r in rows)})
    out["rules"] = table
    best = [max(r["accuracy"][:1000]) for r in rows]
    out["coarse_perfect_pairs_in_top_1000_median"] = float(np.median([sum(a == 1.0 for a in r["accuracy"][:1000])
                                                                       for r in rows]))
    out["true_pair_coarse_shortfall"] = [round(b - max((r["accuracy"][k] for k in r["true_indices"]), default=0), 4)
                                         if r["true_indices"] else None for b, r in zip(best, rows)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--queries-merge", type=Path, nargs="+")
    parser.add_argument("--queries", type=Path, help="phase 2: candidate-recall.json to run targeted queries on")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.queries:
        source = [r for r in json.loads(args.queries.read_text())["cases"]
                  if args.seed_start <= r["seed"] < args.seed_start + args.seeds]
        rows = []
        for row in source:
            rows.append(query_case(row))
            print(row["seed"], rows[-1]["candidates_have_true_pair"], rows[-1]["targeted"][-1]["faults_found"],
                  rows[-1]["targeted"][-1]["found_visible"], flush=True)
        report = {"schema": "weight-space-candidate-queries-v1", "fit": QUERY_FIT, "keep": QUERY_KEEP,
                  "rule": QUERY_RULE, "pool": aq.POOL, "summary": summarize_queries(rows), "cases": rows}
        (args.output / "candidate-queries.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report["summary"]["targeted"][-1]))
        return
    if args.queries_merge:
        rows = sorted((r for p in args.queries_merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
        report = {"schema": "weight-space-candidate-queries-v1", "fit": QUERY_FIT, "keep": QUERY_KEEP,
                  "rule": QUERY_RULE, "pool": aq.POOL, "summary": summarize_queries(rows), "cases": rows}
        (args.output / "candidate-queries.json").write_text(json.dumps(report, indent=1) + "\n")
        return
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            case = tf.corrupt(args.width, seed)
            if case is not None:
                rows.append(analyse(case))
                print(seed, rows[-1]["true_pair_rank"], flush=True)
    report = {"schema": "weight-space-candidate-recall-v1", "fit": FIT, "summary": summarize(rows), "cases": rows}
    (args.output / "candidate-recall.json").write_text(json.dumps(report) + "\n")
    print(json.dumps(report["summary"]["in_screen"]))


if __name__ == "__main__":
    main()
