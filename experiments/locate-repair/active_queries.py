"""Can a few targeted examples single out two broken weights? (roadmap #22)

Roadmap #21 searched every pair of entries at H = 8 and found that 1,024 practice examples
often fit several pairs perfectly; the smallest-change rule then picks a look-alike pair. This
experiment asks the teacher about inputs where the candidate repairs disagree.

Per case (#20's cases, H = 8):

1. #21's all-pairs search; its 20 finely re-fitted pairs are the candidates.
2. Each round, every candidate is re-fitted (joint 200 x 200 grid) on the practice set plus the
   queries so far; the winner has the best accuracy, smallest total change among ties.
3. **targeted** queries: of 5,000 uniform inputs, the 8 where the candidates within 0.5% of the
   best accuracy split their votes most evenly, labelled by the teacher. **random** queries:
   8 uniform inputs. Eight rounds, so up to 64 queries.

    python experiments/locate-repair/active_queries.py --seeds 25 --output runs/active-queries
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pair_search", HERE / "pair_search.py")
ps = importlib.util.module_from_spec(_spec)
sys.modules["pair_search"] = ps
_spec.loader.exec_module(ps)
tf, lr = ps.tf, ps.lr

ROUNDS, PER_ROUND, POOL, MARGIN = 8, 8, 5000, 0.005


def committee_split(predictions):
    """Per input, the share of the minority vote (0 = all agree, 0.5 = an even split)."""
    ones = predictions.mean(axis=0)
    return np.minimum(ones, 1 - ones)


def choose_queries(case, fits, pool, rng, mode):
    if mode == "random":
        return pool[rng.choice(len(pool), PER_ROUND, replace=False)]
    best = max(score[0] for score, _ in fits.values())
    committee = [edits for score, edits in fits.values() if score[0] >= best - MARGIN]
    split = committee_split(np.stack([tf.predict(tf.apply(case["arrays"], e), pool) for e in committee]))
    order = np.lexsort((rng.random(len(pool)), -split))  # most even split first, random among ties
    return pool[order[:PER_ROUND]], float(split[order[:PER_ROUND]].mean())


def disagreement(case, edits, probe):
    """Share of ``probe`` = (inputs, teacher labels) where the edited network disagrees (roadmap #37)."""
    x, labels = probe
    return float((tf.predict(tf.apply(case["arrays"], edits), x) != labels).mean())


def run(case, candidates, mode, probe=None):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    rng = np.random.default_rng([width, case["seed"], 404])
    x, y, history = px, py, []
    for round_ in range(ROUNDS + 1):
        fits = {pair: ps.fine_fit(case, x, y, pair) for pair in candidates}
        winner = max(fits, key=lambda p: fits[p][0])
        score, edits = fits[winner]
        history.append(dict(tf.outcome(case, edits, fx, fy, frepair, inputs, teacher_labels),
                            queries=len(x) - len(px), faults_found=ps.found(winner, faults, width),
                            practice_accuracy=score[0], tied_best=sum(s[0] == score[0] for s, _ in fits.values()),
                            winner=[list(e) for e in winner]))
        if probe is not None:
            history[-1]["disagreement"] = disagreement(case, edits, probe)
        if round_ == ROUNDS:
            break
        pool = rng.uniform(-2, 2, size=(POOL, 2))
        picked = choose_queries(case, fits, pool, rng, mode)
        if mode == "targeted":
            picked, history[-1]["split_of_queries"] = picked
        x = np.concatenate([x, picked])
        y = np.concatenate([y, tf.predict(case["teacher"], picked)])
    return history


def analyse(case):
    px, py, _ = lr.split(case["practice"])
    _, _, fine = ps.search(case, px, py, lr.entries(case["arrays"]))
    candidates = list(fine)
    has_true = any(ps.found(p, case["faults"], case["width"]) == 2 for p in candidates)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "candidates_have_true_pair": has_true,
            "targeted": run(case, candidates, "targeted"), "random": run(case, candidates, "random")}


def summarize(rows):
    out = {"cases": len(rows), "candidates_have_true_pair": float(np.mean([r["candidates_have_true_pair"] for r in rows]))}
    for mode in ("targeted", "random"):
        out[mode] = [{"queries": rows[0][mode][k]["queries"],
                      "found_both": float(np.mean([r[mode][k]["faults_found"] == 2 for r in rows])),
                      "success": float(np.mean([r[mode][k]["success"] for r in rows])),
                      "damage_undone_median": float(np.median([r[mode][k]["damage_undone"] for r in rows])),
                      "tied_best_median": float(np.median([r[mode][k]["tied_best"] for r in rows]))}
                     for k in range(len(rows[0][mode]))]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--merge", type=Path, nargs="+", help="merge active-queries.json files into --output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            case = tf.corrupt(args.width, seed)
            if case is not None:
                rows.append(analyse(case))
                print(seed, rows[-1]["targeted"][-1]["faults_found"], rows[-1]["random"][-1]["faults_found"], flush=True)
    report = {"schema": "weight-space-active-queries-v1", "rounds": ROUNDS, "per_round": PER_ROUND,
              "summary": summarize(rows), "cases": rows}
    (args.output / "active-queries.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
