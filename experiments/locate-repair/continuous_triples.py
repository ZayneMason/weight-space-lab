"""Rank every extended triple by a continuous fit (roadmap #33)

#32 left the three-fault suspect list limited by the coarse grid: on #31's random-size cases the
true triple ranked below 1,000 on an 11^3 grid in 8 of 25 cases, so refinement never saw it. This
experiment drops the grid as a filter. Every triple that #27 extends from the 25-point pair screen's
best 200 pairs gets #29's Nelder-Mead margin fit, started from the current values and from the coarse
7^3 optimum, and the better of the two is kept. Triples are then ranked by practice accuracy, with
ties going to the smallest change or to the coarse-7 order.

    python experiments/locate-repair/continuous_triples.py --seeds 25 --output runs/continuous-triples
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


ds = _load("dense_screen", "dense_screen.py")
rs, tr, t3, js, sc, lr, wt = ds.rs, ds.tr, ds.t3, ds.js, ds.sc, ds.lr, ds.wt


def current_values(case, triple):
    return [float(case["arrays"][n].flat[i]) for n, i in triple]


def continuous_fits(case, px, py, triple, coarse):
    """#29's margin fit from the current values and from the coarse optimum; both results."""
    here = wt.refine(case, px, py, triple, wt.score(case, px, py, triple, current_values(case, triple)))
    there = wt.refine(case, px, py, triple, coarse)
    return here, there


def best_of(a, b):
    return max(a, b, key=lambda fit: fit[0])


def triples_case(case, pair_points=ds.PAIR_POINTS):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    start = time.time()
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    ranked = [pairs[k] for k in js.joint_order(js.joint_scores(case, px, py, pairs, points=pair_points))]
    triples = t3.triples_from(ranked, lr.entries(case["arrays"]))
    is_true = np.array([len(t3.hits(t, faults, width)) == 3 for t in triples])
    coarse = [t3.fit_triple(case, px, py, t, 7) for t in triples]
    acc7, cost7 = np.array([b[0][0] for b in coarse]), np.array([-b[0][1] for b in coarse])
    base = tr.order(acc7, cost7, None)
    rank7 = np.empty(len(triples), int)
    rank7[base] = np.arange(len(triples))
    here, there = zip(*(continuous_fits(case, px, py, t, c) for t, c in zip(triples, coarse)))
    both = [best_of(a, b) for a, b in zip(here, there)]
    out = {}
    for name, fits in (("from_current", here), ("from_coarse7", there), ("best_of_both", both)):
        accuracy, cost = np.array([b[0][0] for b in fits]), np.array([-b[0][1] for b in fits])
        out[name] = (accuracy, cost)
    positions = {"coarse7": tr.position(base, is_true)}
    for name, (accuracy, cost) in out.items():
        positions[name] = tr.position(tr.order(accuracy, cost, None), is_true)
        positions[f"{name}_coarse7_order"] = tr.position(tr.order(accuracy, cost, rank7), is_true)
    accuracy, cost = out["best_of_both"]
    true_k = np.flatnonzero(is_true)
    return {"width": width, "seed": case["seed"], "faults": faults, "sizes": case["sizes"],
            "triples": len(triples), "true_in_triples": bool(true_k.size), "positions": positions,
            "perfect": int((accuracy == 1).sum()),
            "true_accuracy": float(accuracy[true_k[0]]) if true_k.size else None,
            "true_cost": float(cost[true_k[0]]) if true_k.size else None,
            "true_size_total": float(sum(case["sizes"])),
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
            print(seed, json.dumps(rows[-1]["positions"]), rows[-1]["perfect"], [round(s, 2) for s in case["sizes"]],
                  flush=True)
    summary = tr.summarize(rows)
    summary["perfect_median"] = float(np.median([r["perfect"] for r in rows])) if rows else None
    summary["true_fits_all"] = sum(r["true_accuracy"] == 1.0 for r in rows)
    report = {"schema": "weight-space-continuous-triples-v1", "sizes": [rs.LOW, rs.HIGH], "pair_points": ds.PAIR_POINTS,
              "width": args.width, "summary": summary, "cases": rows}
    (args.output / "continuous-triples.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
