"""A joint pair screen for broken weights (roadmap #26)

#23's coverage screen scores a pair from its two single edits. In #25, every case it ranked past
3,000 of 13,041 pairs was a case no candidate rule could recover. This screen fits each pair
jointly on a small grid (13 x 13 values within 2.5 tensor spreads) over the whole practice set,
and ranks pairs by that accuracy, with the coverage rank breaking ties.

    python experiments/locate-repair/joint_screen.py --seeds 25 --output runs/joint-screen

``--queries`` keeps the screen's best 40 pairs as candidates and runs #22's targeted queries on
them, scoring #25's visible faults.

    python experiments/locate-repair/joint_screen.py --seeds 25 --queries --output runs/joint-queries
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sc = _load("pair_screen", "pair_screen.py")
ps, tf, lr = sc.ps, sc.tf, sc.lr
cr = _load("candidate_recall", "candidate_recall.py")
aq = cr.aq

POINTS = 13
KEEP = 40
TOPS = (1000, 3000)


def joint_scores(case, px, py, pairs, points=POINTS):
    """Best practice accuracy of each pair over a points x points joint grid."""
    return np.array([tf.joint(case, px, py, list(pair), points=points)[1] for pair in pairs])


def joint_order(accuracy):
    """Pair indices by joint accuracy, best first; the incoming (coverage) order breaks ties."""
    return np.lexsort((np.arange(len(accuracy)), -np.asarray(accuracy)))


def first_true(order, true):
    true = set(true)
    return next(r for r, k in enumerate(order) if k in true)


def analyse(case):
    px, py, _ = lr.split(case["practice"])
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    true = [k for k, p in enumerate(pairs) if ps.found(p, case["faults"], case["width"]) == 2]
    start = time.time()
    accuracy = joint_scores(case, px, py, pairs)
    seconds = time.time() - start
    order = joint_order(accuracy)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"],
            "coverage_rank": true[0], "joint_rank": first_true(order, true),
            "true_pair_joint_accuracy": float(max(accuracy[k] for k in true)),
            "best_joint_accuracy": float(accuracy.max()), "pairs_at_best": int((accuracy == accuracy.max()).sum()),
            "seconds": round(seconds, 1)}


def query_case(case, keep=KEEP):
    px, py, _ = lr.split(case["practice"])
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    order = joint_order(joint_scores(case, px, py, pairs))
    pool = [pairs[k] for k in order[:keep]]
    silent = cr.silent_faults(case)
    history = aq.run(case, pool, "targeted")
    for step in history:
        step["found_visible"] = cr.found_visible(step["winner"], case["faults"], case["width"], silent)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "silent_faults": silent,
            "candidates_have_true_pair": any(ps.found(p, case["faults"], case["width"]) == 2 for p in pool),
            "targeted": history}


def summarize(rows):
    out = {"cases": len(rows)}
    for key in ("coverage_rank", "joint_rank"):
        ranks = [r[key] for r in rows]
        out[key] = {"median": float(np.median(ranks)), **{f"in_top_{t}": sum(x < t for x in ranks) for t in TOPS}}
    best = [min(r["coverage_rank"], r["joint_rank"]) for r in rows]
    out["either_in_top_1500"] = sum(x < 1500 for x in best)
    out["seconds_median"] = float(np.median([r["seconds"] for r in rows]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--queries", action="store_true", help="run targeted queries on the best 40 pairs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.queries:
        if args.merge:
            rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
        else:
            rows = []
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                case = tf.corrupt(args.width, seed)
                if case is not None:
                    rows.append(query_case(case))
                    last = rows[-1]["targeted"][-1]
                    print(seed, rows[-1]["candidates_have_true_pair"], last["faults_found"], last["found_visible"], flush=True)
        report = {"schema": "weight-space-joint-queries-v1", "points": POINTS, "keep": KEEP, "pool": aq.POOL,
                  "summary": cr.summarize_queries(rows), "cases": rows}
        (args.output / "joint-queries.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report["summary"]["targeted"][-1]))
        return
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            case = tf.corrupt(args.width, seed)
            if case is not None:
                rows.append(analyse(case))
                print(seed, rows[-1]["coverage_rank"], rows[-1]["joint_rank"], rows[-1]["seconds"], flush=True)
    report = {"schema": "weight-space-joint-screen-v1", "points": POINTS, "summary": summarize(rows), "cases": rows}
    (args.output / "joint-screen.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
