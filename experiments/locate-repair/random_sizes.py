"""Do the screens only work because the grid matches the bug? (roadmap #31)

Every fault since #7 shifts its entry by exactly 1.6 tensor spreads, and every joint grid since
#20 (61, 13, 7 and 200 points over +-2.5 spreads) has a point at or within 0.01 of +-1.6 or
+-1.67. #30a found that the 7-point triple grid loses most of its recall when its points move
away from 1.6. This experiment redraws each fault's size uniformly between 1.0 and 2.2 spreads
(same entries, same signs, same seeds) and re-runs:

- **pairs:** #26's pipeline at H = 32: joint 13 x 13 pair screen, best 40 pairs, #22's targeted
  examples;
- **triples:** #30a's rankings at H = 16 (coarse 7, coarse 9, refined).

    python experiments/locate-repair/random_sizes.py --mode pairs --seeds 25 --output runs/random-sizes
    python experiments/locate-repair/random_sizes.py --mode triples --seeds 25 --output runs/random-sizes

Control: ``--fixed 1.6 --points 14`` keeps #26's cases exactly and moves only the pair screen's grid
points off 1.6.
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


tr = _load("triple_recall", "triple_recall.py")
t3, js, lr, tf = tr.t3, tr.js, tr.lr, tr.tf
sc, ps, cr, aq = js.sc, js.ps, js.cr, js.aq

LOW, HIGH = 1.0, 2.2


def resize(case, low=LOW, high=HIGH, max_tries=50):
    """The same faults with the same signs, each shifted by a size drawn uniformly from [low, high]
    spreads instead of 1.6; practice and held-out sets rebuilt as #20 builds them."""
    width, seed, teacher = case["width"], case["seed"], case["teacher"]
    rng = np.random.default_rng([width, seed, 31])
    for _ in range(max_tries):
        sizes = rng.uniform(low, high, len(case["faults"]))
        source = {n: v.copy() for n, v in teacher.items()}
        for (name, index), size in zip(case["faults"], sizes):
            sign = np.sign(float(case["arrays"][name].flat[index]) - float(teacher[name].flat[index]))
            source[name].flat[index] = np.float32(teacher[name].flat[index] + sign * size * case["scale"][name])
        try:
            practice = lr.tasks._dataset(np.random.default_rng([width, seed, 7, tf.PER_GROUP]), source, teacher,
                                         lr.ARCHITECTURE, tf.PER_GROUP, "practice", 2)
            heldout = lr.tasks._dataset(np.random.default_rng([width, seed, 10_000_000]), source, teacher,
                                        lr.ARCHITECTURE, 128, "final", 2)
        except ValueError:
            continue
        return dict(case, arrays=source, practice=practice, heldout=heldout, sizes=[float(s) for s in sizes])
    return None


def pair_case(case, points=js.POINTS):
    px, py, _ = lr.split(case["practice"])
    start = time.time()
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    order = js.joint_order(js.joint_scores(case, px, py, pairs, points=points))
    true = [k for k, p in enumerate(pairs) if ps.found(p, case["faults"], case["width"]) == 2]
    pool = [pairs[k] for k in order[:js.KEEP]]
    silent = cr.silent_faults(case)
    history = aq.run(case, pool, "targeted")
    for step in history:
        step["found_visible"] = cr.found_visible(step["winner"], case["faults"], case["width"], silent)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "sizes": case["sizes"],
            "silent_faults": silent, "joint_rank": js.first_true(order, true),
            "candidates_have_true_pair": any(ps.found(p, case["faults"], case["width"]) == 2 for p in pool),
            "targeted": history, "seconds": round(time.time() - start, 1)}


def summarize_pairs(rows):
    ranks = [r["joint_rank"] for r in rows]
    out = {"cases": len(rows), "silent_cases": sum(bool(r["silent_faults"]) for r in rows),
           "joint_rank_median_position": float(np.median(ranks)) + 1,
           "joint_rank_in_top_40": sum(r < 40 for r in ranks),
           "candidates_have_true_pair": sum(r["candidates_have_true_pair"] for r in rows), "targeted": []}
    for k in range(len(rows[0]["targeted"])):
        steps = [r["targeted"][k] for r in rows]
        out["targeted"].append({"queries": steps[0]["queries"], "found_pair": sum(s["faults_found"] == 2 for s in steps),
                                "found_visible": sum(s["found_visible"] for s in steps),
                                "success": sum(s["success"] for s in steps)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("pairs", "triples"), required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--fixed", type=float, help="every fault this size instead of a random one (a control)")
    parser.add_argument("--points", type=int, default=js.POINTS, help="points per axis of the joint pair screen")
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    width = args.width or (32 if args.mode == "pairs" else 16)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            base = (tf.corrupt if args.mode == "pairs" else t3.corrupt)(width, seed)
            bounds = (args.fixed, args.fixed) if args.fixed is not None else (LOW, HIGH)
            case = resize(base, *bounds) if base is not None else None
            if case is None:
                continue
            row = pair_case(case, args.points) if args.mode == "pairs" else dict(tr.analyse(case), sizes=case["sizes"])
            rows.append(row)
            brief = (row["joint_rank"], row["targeted"][-1]["faults_found"], row["targeted"][-1]["found_visible"]) \
                if args.mode == "pairs" else row["positions"]
            print(seed, json.dumps(brief), [round(s, 2) for s in case["sizes"]], flush=True)
    summary = summarize_pairs(rows) if args.mode == "pairs" else tr.summarize(rows)
    sizes = [args.fixed, args.fixed] if args.fixed is not None else [LOW, HIGH]
    report = {"schema": f"weight-space-random-sizes-{args.mode}-v1", "sizes": sizes, "points": args.points, "width": width,
              "summary": summary, "cases": rows}
    (args.output / f"random-sizes-{args.mode}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
