"""Three broken weights (roadmap #27)

Roadmap #26 located two broken weights at H = 32 with a joint pair screen and targeted queries.
This experiment breaks three entries in three different hidden units and asks whether the pair
screen still ranks a pair of true faults near the top, and whether extending the best pairs by a
third entry finds all three from practice data alone.

1. Every pair gets #26's 13 x 13 joint fit on the practice set, ranked by accuracy (coverage
   rank breaks ties).
2. The best 200 pairs are each extended by every other entry, and each triple gets a coarse
   7 x 7 x 7 joint fit. The best 20 triples are re-fitted on a 21 x 21 x 21 grid, two finer local
   grids around its best point, and three passes of one-entry-at-a-time fits on #10's fine grid
   (the 21-point grid does not contain the true shifts). The winner has the best practice accuracy, then the smallest
   total change.

    python experiments/locate-repair/three_faults.py --width 16 --seeds 25 --output runs/three-faults
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


js = _load("joint_screen", "joint_screen.py")
sc, ps, tf, lr, cr = js.sc, js.ps, js.tf, js.lr, js.cr
sub = tf.sub

PAIRS, COARSE, FINE, REFINE, SPAN = 200, 7, 21, 20, 2.5


def corrupt(width, seed, max_tries=200):
    """#20's two-fault case plus a third corrupted entry in yet another hidden unit, shifted by
    1.6 tensor spreads; never the mirror or the same unit as either earlier fault."""
    base = tf.corrupt(width, seed)
    if base is None:
        return None
    rng = np.random.default_rng([width, seed, 3])
    taken = [tuple(f) for f in base["faults"]]
    candidates = lr.entries(base["arrays"])
    for _ in range(max_tries):
        third = candidates[rng.integers(len(candidates))]
        if any(third == f or sub.relation(third, f, width) in ("mirror", "same unit") for f in taken):
            continue
        source = {n: v.copy() for n, v in base["arrays"].items()}
        name, index = third
        source[name].flat[index] += np.float32(rng.choice([-1, 1]) * 1.6 * base["scale"][name])
        try:
            practice = lr.tasks._dataset(np.random.default_rng([width, seed, 7, tf.PER_GROUP]), source,
                                         base["teacher"], lr.ARCHITECTURE, tf.PER_GROUP, "practice", 2)
            heldout = lr.tasks._dataset(np.random.default_rng([width, seed, 10_000_000]), source, base["teacher"],
                                        lr.ARCHITECTURE, 128, "final", 2)
        except ValueError:
            continue
        return dict(base, arrays=source, faults=base["faults"] + [[name, int(index)]], practice=practice,
                    heldout=heldout)
    return None


def fit_triple(case, px, py, triple, points, span=SPAN, center=None):
    """Best (practice accuracy, -relative change) and edits over a points^3 grid of the three values,
    within ``span`` tensor spreads of ``center`` (default: the current values)."""
    arrays, scale = case["arrays"], case["scale"]
    current = [float(arrays[n].flat[i]) for n, i in triple]
    middle = current if center is None else center
    offsets = np.linspace(-span, span, points)
    grids = [c + offsets * scale[n] for c, (n, _) in zip(middle, triple)]
    (n1, i1), (n2, i2), (n3, i3) = triple
    best = None
    for v1 in grids[0]:
        for v2 in grids[1]:
            trial = {n: a.copy() for n, a in arrays.items()}
            trial[n1].flat[i1] = np.float32(v1)
            trial[n2].flat[i2] = np.float32(v2)
            accuracy = (lr.Evaluator(trial, px).predict(n3, i3, grids[2]) == py).mean(axis=1)
            cost = (abs(v1 - current[0]) / scale[n1] + abs(v2 - current[1]) / scale[n2]
                    + np.abs(grids[2] - current[2]) / scale[n3])
            j = np.lexsort((cost, -accuracy))[0]
            key = (float(accuracy[j]), -float(cost[j]))
            if best is None or key > best[0]:
                best = key, [(n1, i1, float(v1)), (n2, i2, float(v2)), (n3, i3, float(grids[2][j]))]
    return best


def polish(case, px, py, best, passes=3):
    """Re-fit each entry in turn on #10's fine 1-D grid, starting from a grid fit."""
    (accuracy, _), edits = best
    edits = [list(e) for e in edits]
    for _ in range(passes):
        for k, (name, index, value) in enumerate(edits):
            others = [tuple(e) for j, e in enumerate(edits) if j != k]
            evaluator = lr.Evaluator(tf.apply(case["arrays"], others), px)
            accuracy, edits[k][2] = tf.best_edit(evaluator, py, name, index, value)
    cost = sum(abs(v - float(case["arrays"][n].flat[i])) / case["scale"][n] for n, i, v in edits)
    return (float(accuracy), -float(cost)), [tuple(e) for e in edits]


def fine_fit(case, px, py, triple):
    """A 21^3 grid, two finer local grids around its best point (steps of 0.06 and 0.012
    spreads), then a polish."""
    best = fit_triple(case, px, py, triple, FINE)
    for points, span in ((11, 0.3), (11, 0.06)):
        best = max(best, fit_triple(case, px, py, triple, points, span, center=[v for _, _, v in best[1]]),
                   key=lambda b: b[0])
    return max(best, polish(case, px, py, best), key=lambda b: b[0])


def hits(group, faults, width):
    return {k for e in group for k in tf.matches(e, faults, width)}


def triples_from(pairs, entries, limit=PAIRS):
    seen, out = set(), []
    for pair in pairs[:limit]:
        for e in entries:
            if e in pair:
                continue
            key = tuple(sorted([*pair, e]))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def analyse(case):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    silent = cr.silent_faults(case)
    visible = [k for k in range(len(faults)) if k not in silent]
    start = time.time()
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    order = js.joint_order(js.joint_scores(case, px, py, pairs))
    ranked = [pairs[k] for k in order]
    pair_rank = next(r for r, p in enumerate(ranked) if len(hits(p, faults, width)) >= 2)
    triples = triples_from(ranked, lr.entries(case["arrays"]))
    coarse = {t: fit_triple(case, px, py, t, COARSE)[0] for t in triples}
    top = sorted(triples, key=coarse.get, reverse=True)
    true_triples = [r for r, t in enumerate(top) if len(hits(t, faults, width)) == 3]
    fine = {t: fine_fit(case, px, py, t) for t in top[:REFINE]}
    winner = max(fine, key=lambda t: fine[t][0])
    told = fine_fit(case, px, py, [tuple(f) for f in faults])
    seconds = time.time() - start
    found = hits(winner, faults, width)
    return {"width": width, "seed": case["seed"], "faults": faults, "silent_faults": silent,
            "best_pair_rank": pair_rank, "triples": len(triples),
            "true_triple_rank": true_triples[0] if true_triples else None,
            "winner": [list(e) for e in winner], "faults_found": len(found),
            "found_visible": all(k in found for k in visible),
            "perfect_fine_triples": sum(v[0][0] == 1.0 for v in fine.values()),
            "search": dict(tf.outcome(case, fine[winner][1], fx, fy, frepair, inputs, teacher_labels),
                           practice_accuracy=fine[winner][0][0]),
            "told": dict(tf.outcome(case, told[1], fx, fy, frepair, inputs, teacher_labels),
                         practice_accuracy=told[0][0]),
            "seconds": round(seconds, 1)}


def summarize(rows):
    n = len(rows)
    return {"cases": n, "silent_cases": sum(bool(r["silent_faults"]) for r in rows),
            "best_pair_rank_median": float(np.median([r["best_pair_rank"] for r in rows])),
            "true_triple_among_candidates": sum(r["true_triple_rank"] is not None for r in rows),
            "true_triple_in_top_20": sum(r["true_triple_rank"] is not None and r["true_triple_rank"] < REFINE
                                         for r in rows),
            "found_all_three": sum(r["faults_found"] == 3 for r in rows),
            "found_visible": sum(r["found_visible"] for r in rows),
            "search_success": sum(r["search"]["success"] for r in rows),
            "told_success": sum(r["told"]["success"] for r in rows),
            "search_damage_undone_median": float(np.median([r["search"]["damage_undone"] for r in rows])),
            "told_damage_undone_median": float(np.median([r["told"]["damage_undone"] for r in rows])),
            "seconds_median": float(np.median([r["seconds"] for r in rows]))}


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
            case = corrupt(args.width, seed)
            if case is not None:
                rows.append(analyse(case))
                r = rows[-1]
                print(seed, r["best_pair_rank"], r["true_triple_rank"], r["faults_found"], r["found_visible"],
                      r["search"]["success"], r["seconds"], flush=True)
    report = {"schema": "weight-space-three-faults-v1", "pairs": PAIRS, "coarse": COARSE, "fine": FINE,
              "refine": REFINE, "summary": summarize(rows), "cases": rows}
    (args.output / "three-faults.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
