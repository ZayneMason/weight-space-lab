"""Denser screens on bug sizes the grid cannot match (roadmap #32)

#31 showed that #26's 13-point pair screen and #27's 7-point triple ranking lose recall when the
faults are not 1.6 spreads, the size their grids happen to contain. This experiment keeps #31's
random-size cases (sizes uniform in 1.0-2.2 spreads) and makes the screens denser, so that every
size sits near a grid point:

- **pairs:** #26's pipeline with a 25-point joint screen (``random_sizes.py --points 25``);
- **triples:** a 25-point pair screen feeding #27's extended triples, ranked by a coarse 7^3 fit,
  a coarse 11^3 fit, and the coarse-11 top 1,000 refined with #29's margin fit.

    python experiments/locate-repair/random_sizes.py --mode pairs --points 25 --seeds 25 --output runs/dense-screen
    python experiments/locate-repair/dense_screen.py --seeds 25 --output runs/dense-screen
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


rs = _load("random_sizes", "random_sizes.py")
tr, t3, js, sc, lr, wt = rs.tr, rs.t3, rs.js, rs.sc, rs.lr, rs.tr.wt

PAIR_POINTS, COARSE, DENSE = 25, 7, 11


def triples_case(case, pair_points=PAIR_POINTS, coarse=COARSE, dense=DENSE, refine_top=tr.REFINE_TOP):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    start = time.time()
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    ranked = [pairs[k] for k in js.joint_order(js.joint_scores(case, px, py, pairs, points=pair_points))]
    triples = t3.triples_from(ranked, lr.entries(case["arrays"]))
    is_true = np.array([len(t3.hits(t, faults, width)) == 3 for t in triples])
    fits = {}
    for points in (coarse, dense):
        fit = [t3.fit_triple(case, px, py, t, points) for t in triples]
        fits[points] = (fit, np.array([b[0][0] for b in fit]), np.array([-b[0][1] for b in fit]))
    base = tr.order(fits[coarse][1], fits[coarse][2], None)
    dense_order = tr.order(fits[dense][1], fits[dense][2], None)
    rank = np.empty(len(triples), int)
    rank[dense_order] = np.arange(len(triples))
    top = dense_order[:refine_top]
    refined = [wt.refine(case, px, py, triples[k], fits[dense][0][k]) for k in top]
    accr, costr = np.array([b[0][0] for b in refined]), np.array([-b[0][1] for b in refined])
    positions = {
        f"coarse{coarse}": tr.position(base, is_true),
        f"coarse{dense}": tr.position(dense_order, is_true),
        f"refined{dense}": tr.position(top[tr.order(accr, costr, None)], is_true),
        f"refined{dense}_coarse{dense}_order": tr.position(top[tr.order(accr, costr, rank[top])], is_true),
    }
    return {"width": width, "seed": case["seed"], "faults": faults, "sizes": case["sizes"],
            "triples": len(triples), "true_in_triples": bool(is_true.any()), "positions": positions,
            "seconds": round(time.time() - start, 1)}


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
            base = t3.corrupt(args.width, seed)
            case = rs.resize(base) if base is not None else None
            if case is None:
                continue
            rows.append(triples_case(case))
            print(seed, json.dumps(rows[-1]["positions"]), [round(s, 2) for s in case["sizes"]], flush=True)
    report = {"schema": "weight-space-dense-screen-triples-v1", "sizes": [rs.LOW, rs.HIGH], "pair_points": PAIR_POINTS,
              "coarse": [COARSE, DENSE], "width": args.width, "summary": tr.summarize(rows), "cases": rows}
    (args.output / "dense-screen-triples.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
