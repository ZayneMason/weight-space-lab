"""Targeted examples on a 200-triple committee (roadmap #34)

#33's continuous fit puts the true triple in the top 200 in 21 of 25 random-size cases, but a median
of 160 triples fit every practice label, so ranking on practice data alone can't single it out. This
experiment builds #33's ranking, takes the best 200 triples (ties by coarse-7 order) as the
committee, and runs #22's targeted examples: each round, 8 inputs from 20,000 fresh ones on which
the committee members within 0.5% of the best accuracy split their votes most evenly, labelled by the
teacher, up to 64. Every round each member is re-fitted with #29's margin fit, started from its last
values and from its current values, keeping the better.

    python experiments/locate-repair/triple_committee.py --seeds 25 --output runs/triple-committee
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


ct = _load("continuous_triples", "continuous_triples.py")
rs, tr, t3, js, sc, lr, wt, ds = ct.rs, ct.tr, ct.t3, ct.js, ct.sc, ct.lr, ct.wt, ct.ds
tf, cr, aq = rs.tf, rs.cr, rs.aq

COMMITTEE = 200
DRAW, CAP = 10, 20


def boundary_pool(case, fits, rng, size=None, draw=DRAW, cap=CAP):
    """Roadmap #35: of ``draw`` x ``size`` uniform inputs, the ``size`` nearest any decision boundary
    of the (at most ``cap``) members within the query margin of the best accuracy."""
    size = size or aq.POOL
    candidates = rng.uniform(-2, 2, size=(draw * size, 2))
    best = max(score[0] for score, _ in fits.values())
    tied = [(t, edits) for t, (score, edits) in fits.items() if score[0] >= best - aq.MARGIN][:cap]
    ones = np.ones(len(candidates))
    closeness = np.min([np.abs(wt.signed_margins(case, t, [v for _, _, v in edits], candidates, ones))
                        for t, edits in tied], axis=0)
    return candidates[np.argsort(closeness)[:size]]


def committee(case, px, py, size=COMMITTEE, pairs_kept=None):
    """#33's ranking: every extended triple fitted continuously, best accuracy first, coarse-7 order
    breaking ties. Returns the best ``size`` triples with their fits. ``pairs_kept`` is how many of the
    best pairs are extended to triples (default t3.PAIRS, 200; roadmap #47 tries more)."""
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    ranked = [pairs[k] for k in js.joint_order(js.joint_scores(case, px, py, pairs, points=ds.PAIR_POINTS))]
    triples = t3.triples_from(ranked, lr.entries(case["arrays"]), pairs_kept or t3.PAIRS)
    coarse = [t3.fit_triple(case, px, py, t, 7) for t in triples]
    acc7, cost7 = np.array([b[0][0] for b in coarse]), np.array([-b[0][1] for b in coarse])
    rank7 = np.empty(len(triples), int)
    rank7[tr.order(acc7, cost7, None)] = np.arange(len(triples))
    fits = [ct.best_of(*ct.continuous_fits(case, px, py, t, c)) for t, c in zip(triples, coarse)]
    accuracy, cost = np.array([b[0][0] for b in fits]), np.array([-b[0][1] for b in fits])
    top = tr.order(accuracy, cost, rank7)[:size]
    return [triples[k] for k in top], {triples[k]: fits[k] for k in top}, len(triples)


def refit(case, x, y, triple, previous):
    """The better of #29's margin fit from the last values and from the current values."""
    last = wt.refine(case, x, y, triple, wt.score(case, x, y, triple, [v for _, _, v in previous[1]]))
    here = wt.refine(case, x, y, triple, wt.score(case, x, y, triple, ct.current_values(case, triple)))
    return ct.best_of(last, here)


def query_pool(case, fits, rng, pool_mode):
    """Uniform (#34), boundary (#35), or mixed: half of each (#36)."""
    if pool_mode == "uniform":
        return rng.uniform(-2, 2, size=(aq.POOL, 2))
    if pool_mode == "boundary":
        return boundary_pool(case, fits, rng)
    half = aq.POOL // 2
    return np.concatenate([rng.uniform(-2, 2, size=(half, 2)), boundary_pool(case, fits, rng, size=aq.POOL - half)])


def run(case, members, fits, silent, pool_mode="uniform", return_fits=False, probe=None):
    width, faults = case["width"], case["faults"]
    told_triple = tuple(tuple(f[:2]) for f in faults)
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    rng = np.random.default_rng([width, case["seed"], 404])
    x, y, history = px, py, []
    told = wt.refine(case, x, y, told_triple, wt.score(case, x, y, told_triple, ct.current_values(case, told_triple)))
    for round_ in range(aq.ROUNDS + 1):
        if round_:
            fits = {t: refit(case, x, y, t, fits[t]) for t in members}
            told = refit(case, x, y, told_triple, told)
        winner = max(members, key=lambda t: fits[t][0])
        (accuracy, _), edits = fits[winner]
        found = t3.hits(winner, faults, width)
        true_scores = [fits[t][0][0] for t in members if len(t3.hits(t, faults, width)) == 3]
        history.append(dict(tf.outcome(case, edits, fx, fy, frepair, inputs, teacher_labels),
                            queries=len(x) - len(px), faults_found=len(found),
                            found_visible=cr.found_visible(winner, faults, width, silent), practice_accuracy=accuracy,
                            tied_best=sum(fits[t][0][0] == accuracy for t in members),
                            true_triple_accuracy=max(true_scores) if true_scores else None,
                            told_accuracy=told[0][0], winner=[list(e) for e in winner]))
        if probe is not None:
            history[-1]["disagreement"] = aq.disagreement(case, edits, probe)
            history[-1]["told_disagreement"] = aq.disagreement(case, told[1], probe)
        if round_ == aq.ROUNDS:
            break
        pool = query_pool(case, fits, rng, pool_mode)
        picked, history[-1]["split_of_queries"] = aq.choose_queries(case, fits, pool, rng, "targeted")
        x = np.concatenate([x, picked])
        y = np.concatenate([y, tf.predict(case["teacher"], picked)])
    return (history, fits, told) if return_fits else history


def analyse(case, size=COMMITTEE, pool_mode="uniform"):
    px, py, _ = lr.split(case["practice"])
    start = time.time()
    members, fits, count = committee(case, px, py, size)
    built = time.time() - start
    silent = cr.silent_faults(case)
    history = run(case, members, fits, silent, pool_mode)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "sizes": case["sizes"],
            "silent_faults": silent, "triples": count,
            "committee_has_true_triple": any(len(t3.hits(t, case["faults"], case["width"])) == 3 for t in members),
            "targeted": history, "seconds_committee": round(built, 1), "seconds": round(time.time() - start, 1)}


def summarize(rows):
    out = {"cases": len(rows), "silent_cases": sum(bool(r["silent_faults"]) for r in rows),
           "committee_has_true_triple": sum(r["committee_has_true_triple"] for r in rows), "targeted": []}
    for k in range(len(rows[0]["targeted"])):
        steps = [r["targeted"][k] for r in rows]
        out["targeted"].append({
            "queries": steps[0]["queries"], "all_three": sum(s["faults_found"] == 3 for s in steps),
            "found_visible": sum(s["found_visible"] for s in steps), "success": sum(s["success"] for s in steps),
            "winner_fits_all": sum(s["practice_accuracy"] == 1.0 for s in steps),
            "told_fits_all": sum(s["told_accuracy"] == 1.0 for s in steps),
            "true_triple_fits_all": sum(s["true_triple_accuracy"] == 1.0 for s in steps
                                        if s["true_triple_accuracy"] is not None),
            "tied_best_median": float(np.median([s["tied_best"] for s in steps]))})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--committee", type=int, default=COMMITTEE)
    parser.add_argument("--pool", choices=("uniform", "boundary", "mixed"), default="uniform",
                        help="where each round's candidate inputs come from (boundary: roadmap #35)")
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
            rows.append(analyse(case, args.committee, args.pool))
            last = rows[-1]["targeted"][-1]
            print(seed, rows[-1]["committee_has_true_triple"], last["faults_found"], last["found_visible"],
                  last["success"], last["practice_accuracy"], last["tied_best"], rows[-1]["seconds"], flush=True)
    report = {"schema": "weight-space-triple-committee-v1", "committee": args.committee, "pool_mode": args.pool, "sizes": [rs.LOW, rs.HIGH],
              "pool": aq.POOL, "rounds": aq.ROUNDS, "per_round": aq.PER_ROUND, "summary": summarize(rows), "cases": rows}
    (args.output / "triple-committee.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]["targeted"][-1]))


if __name__ == "__main__":
    main()
