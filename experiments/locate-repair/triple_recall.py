"""Getting the true triple onto the suspect list (roadmap #30, part a)

Roadmap #29 found all three broken weights whenever the fit could track them, but in 8 of 25
cases the true triple never made #27's coarse top 40. This experiment measures where the true
triple ranks under three ways of scoring #27's extended triples on the practice set:

- **coarse 7:** #27's 7^3 grid fit (what #28 and #29 used);
- **coarse 9:** a 9^3 grid fit;
- **refined:** the coarse-7 top 1,000, each refined with #29's Nelder-Mead margin fit.

Each is ordered by practice accuracy, with smallest change breaking ties (as #28 did) or, for
comparison, an earlier order: the pair screen's extension order for coarse 7, the coarse-7 order
for the others. It reports how often the true triple is in the top 40, 100 and 200.

    python experiments/locate-repair/triple_recall.py --seeds 25 --output runs/triple-recall
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


wt = _load("warm_triples", "warm_triples.py")
t3, js, sc, lr, tf = wt.t3, wt.t3.js, wt.t3.sc, wt.lr, wt.tf

REFINE_TOP, KEEPS, SPAN_CHECK = 1000, (40, 100, 200), (2.0, 3.0)


def order(accuracy, cost, tiebreak):
    """Indices by best accuracy; ties by smallest change or by the given earlier order."""
    second = cost if tiebreak is None else tiebreak
    return np.lexsort((second, -np.asarray(accuracy)))


def position(ordering, is_true):
    hits = [p for p, k in enumerate(ordering) if is_true[k]]
    return hits[0] if hits else None


def analyse(case):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    start = time.time()
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    ranked = [pairs[k] for k in js.joint_order(js.joint_scores(case, px, py, pairs))]
    triples = t3.triples_from(ranked, lr.entries(case["arrays"]))
    is_true = np.array([len(t3.hits(t, faults, width)) == 3 for t in triples])
    c7 = [t3.fit_triple(case, px, py, t, 7) for t in triples]
    c9 = [t3.fit_triple(case, px, py, t, 9) for t in triples]
    acc7, cost7 = np.array([b[0][0] for b in c7]), np.array([-b[0][1] for b in c7])
    acc9, cost9 = np.array([b[0][0] for b in c9]), np.array([-b[0][1] for b in c9])
    base = order(acc7, cost7, None)  # #28's committee order: accuracy, then smallest change
    rank7 = np.empty(len(triples), int)
    rank7[base] = np.arange(len(triples))
    top = base[:REFINE_TOP]
    refined = [wt.refine(case, px, py, triples[k], c7[k]) for k in top]
    accr, costr = np.array([b[0][0] for b in refined]), np.array([-b[0][1] for b in refined])
    positions = {
        "coarse7": position(base, is_true),
        "coarse7_extension_order": position(order(acc7, cost7, np.arange(len(triples))), is_true),
        "coarse9": position(order(acc9, cost9, None), is_true),
        "coarse9_coarse7_order": position(order(acc9, cost9, rank7), is_true),
        "refined": position(top[order(accr, costr, None)], is_true),
        "refined_coarse7_order": position(top[order(accr, costr, rank7[top])], is_true),
    }
    true_k = [k for k in np.flatnonzero(is_true)]
    return {"width": width, "seed": case["seed"], "faults": faults, "triples": len(triples),
            "true_in_triples": bool(true_k), "positions": positions,
            "true_coarse7_accuracy": float(acc7[true_k[0]]) if true_k else None,
            "true_coarse9_accuracy": float(acc9[true_k[0]]) if true_k else None,
            "perfect_coarse7": int((acc7 == 1).sum()), "perfect_coarse9": int((acc9 == 1).sum()),
            "perfect_refined": int((accr == 1).sum()), "seconds": round(time.time() - start, 1)}


def span_check(case, spans=SPAN_CHECK):
    """The true triple's coarse-7 position when the grid spans other widths (offsets then miss
    the 1.6-spread corruption by different amounts)."""
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    ranked = [pairs[k] for k in js.joint_order(js.joint_scores(case, px, py, pairs))]
    triples = t3.triples_from(ranked, lr.entries(case["arrays"]))
    is_true = np.array([len(t3.hits(t, faults, width)) == 3 for t in triples])
    out = {"seed": case["seed"], "positions": {}}
    for span in spans:
        fits = [t3.fit_triple(case, px, py, t, 7, span) for t in triples]
        accuracy, cost = np.array([b[0][0] for b in fits]), np.array([-b[0][1] for b in fits])
        out["positions"][f"span{span}"] = position(order(accuracy, cost, None), is_true)
    return out


def summarize(rows):
    out = {"cases": len(rows), "true_in_triples": sum(r.get("true_in_triples", True) for r in rows)}
    for name in rows[0]["positions"]:
        values = [r["positions"][name] for r in rows]
        out[name] = {f"top{k}": sum(v is not None and v < k for v in values) for k in KEEPS}
        found = [v + 1 for v in values if v is not None]
        out[name]["median_position"] = float(np.median(found)) if found else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--span-check", action="store_true", help="only the coarse-7 grid at other spans")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.span_check:
        if args.merge:
            rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
        else:
            rows = []
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                case = t3.corrupt(args.width, seed)
                if case is not None:
                    rows.append(span_check(case))
                    print(seed, json.dumps(rows[-1]["positions"]), flush=True)
        report = {"schema": "weight-space-triple-span-check-v1", "spans": SPAN_CHECK, "summary": summarize(rows),
                  "cases": rows}
        (args.output / "span-check.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report["summary"]))
        return
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            case = t3.corrupt(args.width, seed)
            if case is not None:
                rows.append(analyse(case))
                print(seed, json.dumps(rows[-1]["positions"]), rows[-1]["seconds"], flush=True)
    report = {"schema": "weight-space-triple-recall-v1", "refine_top": REFINE_TOP, "keeps": KEEPS,
              "summary": summarize(rows), "cases": rows}
    (args.output / "triple-recall.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
