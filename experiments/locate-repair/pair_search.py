"""Can a search over pairs of weights find two broken ones from practice data? (roadmap #21)

Roadmap #20 broke two entries in different hidden units. Told which two, a joint search over
both values passed the benchmark in 96% of cases; untold, two greedy edits found both broken
entries in 4-8% of cases. This experiment ranks *pairs* of entries instead of one entry at a time.

- **shortlist:** every entry's best single-edit practice accuracy; keep the top K entries
  (K = 8, 16);
- **pair search:** every pair in the shortlist gets a joint 61 x 61 grid (each value within 2.5
  tensor spreads of its current value); the pair with the best practice accuracy wins,
  smallest total relative change among ties. The coarse grid misses the true values by up to
  half a step, so the 20 best pairs are re-fitted on #20's 200 x 200 grid and the winner is
  picked again on that fit;
- **all pairs** (H = 8 only): the same search over every pair of the 42 entries, which also
  counts how many pairs reach a practice-perfect fit, i.e. how ambiguous the practice data is.

Same cases, practice data and success rule as #20.

    python experiments/locate-repair/pair_search.py --widths 8 32 --seeds 50 --output runs/pair-search
    python experiments/locate-repair/pair_search.py --widths 8 --seeds 25 --all-pairs --output runs/pair-search-all
"""
import argparse
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("two_faults", HERE / "two_faults.py")
tf = importlib.util.module_from_spec(_spec)
sys.modules["two_faults"] = tf
_spec.loader.exec_module(tf)
lr, sub = tf.lr, tf.sub

SHORTLISTS = (8, 16)
COARSE = 61
REFINE = 20


def single_scores(case, px, py):
    """Best single-edit practice accuracy per entry."""
    evaluator, arrays = lr.Evaluator(case["arrays"], px), case["arrays"]
    return {(n, i): tf.best_edit(evaluator, py, n, i, float(arrays[n].flat[i]))[0] for n, i in lr.entries(arrays)}


def pair_fit(case, px, py, pair, points=COARSE, span=2.5):
    """Best (accuracy, -relative change) over a joint grid of the pair's two values."""
    edits, accuracy = tf.joint(case, px, py, list(pair), points=points, span=span)
    cost = sum(abs(v - float(case["arrays"][n].flat[i])) / case["scale"][n] for n, i, v in edits)
    return accuracy, -cost


def found(pair, faults, width):
    return len({k for e in pair for k in tf.matches(e, faults, width)})


def fine_fit(case, px, py, pair):
    edits, accuracy = tf.joint(case, px, py, list(pair))
    cost = sum(abs(v - float(case["arrays"][n].flat[i])) / case["scale"][n] for n, i, v in edits)
    return (accuracy, -cost), edits


def search(case, px, py, pool, refine=REFINE):
    """Coarse-fit every pair in the pool, re-fit the best ``refine`` finely, pick the best fine fit."""
    fits = {pair: pair_fit(case, px, py, pair) for pair in combinations(pool, 2)}
    shortlist = sorted(fits, key=fits.get, reverse=True)[:refine]
    fine = {pair: fine_fit(case, px, py, pair) for pair in shortlist}
    winner = max(fine, key=lambda p: fine[p][0])
    return winner, fits, fine


def analyse(case, all_pairs=False):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    scores = single_scores(case, px, py)
    ranked = sorted(scores, key=lambda e: -scores[e])  # stable: ties keep tensor order
    rank = lambda fault: min(k for k, e in enumerate(ranked) if tf.matches(e, [fault], width))
    row = {"width": width, "seed": case["seed"], "faults": faults, "fault_ranks": [rank(f) for f in faults]}
    pools = {f"top{k}": ranked[:k] for k in SHORTLISTS}
    if all_pairs:
        pools["all"] = ranked
    for label, pool in pools.items():
        winner, fits, fine = search(case, px, py, pool)
        (accuracy, _), edits = fine[winner]
        perfect = [p for p, ((a, _), _) in fine.items() if a == 1.0]
        true = tuple(tuple(f) for f in faults)
        true_fine = fine_fit(case, px, py, true)[0][0]
        row[label] = dict(tf.outcome(case, edits, fx, fy, frepair, inputs, teacher_labels),
                          shortlist_has_both=found(pool, faults, width) == 2,
                          faults_found=found(winner, faults, width), practice_accuracy=accuracy,
                          pairs=len(fits), refined=len(fine), perfect_refined_pairs=len(perfect),
                          refined_has_both=any(found(p, faults, width) == 2 for p in fine),
                          true_pair_fine_accuracy=true_fine,
                          perfect_pairs_with_both=sum(found(p, faults, width) == 2 for p in perfect),
                          true_pair_coarse_accuracy=fits[true][0] if true in fits else
                          (fits[true[::-1]][0] if true[::-1] in fits else None),
                          winner=[list(e) for e in winner])
    return row


def summarize(rows):
    out = {}
    for width in sorted({r["width"] for r in rows}):
        group = [r for r in rows if r["width"] == width]
        entry = {"cases": len(group), "fault_rank_median": float(np.median([r["fault_ranks"] for r in group])),
                 "both_faults_in_top": {k: float(np.mean([max(r["fault_ranks"]) < k for r in group])) for k in (2, 4, 8, 16)}}
        for label in [k for k in group[0] if k.startswith("top") or k == "all"]:
            rows_ = [r[label] for r in group]
            entry[label] = {"success": float(np.mean([r["success"] for r in rows_])),
                            "shortlist_has_both": float(np.mean([r["shortlist_has_both"] for r in rows_])),
                            "found_both": float(np.mean([r["faults_found"] == 2 for r in rows_])),
                            "found_one": float(np.mean([r["faults_found"] == 1 for r in rows_])),
                            "damage_undone_median": float(np.median([r["damage_undone"] for r in rows_])),
                            "refined_has_both": float(np.mean([r["refined_has_both"] for r in rows_])),
                            "true_pair_fine_perfect": float(np.mean([r["true_pair_fine_accuracy"] == 1.0 for r in rows_])),
                            "perfect_refined_pairs_median": float(np.median([r["perfect_refined_pairs"] for r in rows_])),
                            "pairs": rows_[0]["pairs"]}
        out[str(width)] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--all-pairs", action="store_true", help="also search every pair of entries (slow beyond H = 8)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for width in args.widths:
        for seed in range(args.seeds):
            case = tf.corrupt(width, seed)
            if case is not None:
                rows.append(analyse(case, args.all_pairs))
        print(width, json.dumps(summarize([r for r in rows if r["width"] == width])[str(width)]), flush=True)
        report = {"schema": "weight-space-pair-search-v2", "coarse_grid": COARSE, "refined": REFINE, "summary": summarize(rows), "cases": rows}
        (args.output / "pair-search.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
