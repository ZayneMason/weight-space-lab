"""Targeted examples on screened candidates at H = 32 (roadmap #24)

Roadmap #22 showed that at H = 8, 64 teacher-labelled examples chosen where the candidate pairs
disagree let the broken pair win in 20 of 25 cases, against 14 without them. #23's coverage screen
made candidate finding cheap at H = 32, where fits on its top 1,000 pairs still found the broken
pair in only 10% of cases. This experiment combines the two:

1. #23's coverage screen; #21's coarse joint fit of the top 1,000 pairs; the best 40 (not 20)
   re-fitted finely are the candidates;
2. #22's eight rounds of 8 targeted or 8 random queries, re-fitting every candidate each round,
   with a query pool of 20,000 fresh uniform inputs per round (#22 used 5,000).

    python experiments/locate-repair/active_screen.py --seeds 25 --output runs/active-screen
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
ps, tf, lr = sc.ps, sc.tf, sc.lr

TOP, CANDIDATES = 1000, 40
aq.POOL = 20_000


def candidates(case, top=TOP, keep=CANDIDATES):
    px, py, _ = lr.split(case["practice"])
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    rank = sc.true_rank(pairs, case["faults"], case["width"])
    fits = {pair: ps.pair_fit(case, px, py, pair) for pair in pairs[:top]}
    return sorted(fits, key=fits.get, reverse=True)[:keep], rank


def analyse(case):
    pool, rank = candidates(case)
    has_true = any(ps.found(p, case["faults"], case["width"]) == 2 for p in pool)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "true_pair_rank": rank,
            "candidates_have_true_pair": has_true,
            "targeted": aq.run(case, pool, "targeted"), "random": aq.run(case, pool, "random")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--merge", type=Path, nargs="+", help="merge active-screen.json files into --output")
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
                print(seed, rows[-1]["candidates_have_true_pair"], rows[-1]["targeted"][-1]["faults_found"],
                      rows[-1]["random"][-1]["faults_found"], flush=True)
    report = {"schema": "weight-space-active-screen-v1", "top": TOP, "candidates": CANDIDATES, "pool": aq.POOL,
              "rounds": aq.ROUNDS, "per_round": aq.PER_ROUND, "summary": aq.summarize(rows), "cases": rows}
    (args.output / "active-screen.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
