"""A fitter that keeps the true triple perfect (roadmap #29)

Roadmap #28's targeted examples land where candidate repairs disagree, and there #27's grid fit
of three values falls short: after 64 examples no candidate, the true triple included, fit every
label in 15 of the 17 cases that had the true triple. This experiment re-runs #28 with a better
fit for every triple in every round:

- the round's full grid fit (#27's fine fit), as before;
- a warm start: the triple's previous best values, and local grids and coordinate passes around them;
- a continuous refinement of the best of these, minimising a hinge loss on the output margin
  with Nelder-Mead in units of each tensor's spread.

The best by practice accuracy, then smallest change, is kept. Everything else is #28's.

    python experiments/locate-repair/warm_triples.py --seeds 25 --output runs/warm-triples
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tq = _load("triple_queries", "triple_queries.py")
t3, tf, lr, cr, aq = tq.t3, tq.tf, tq.lr, tq.cr, tq.aq

EPS = 1e-4


def signed_margins(case, triple, values, x, y):
    arrays = tf.apply(case["arrays"], [(n, i, v) for (n, i), v in zip(triple, values)])
    a = {n: v.astype(np.float64) for n, v in arrays.items()}
    logits = np.maximum(x @ a["hidden_weight"].T + a["hidden_bias"], 0) @ a["weight"].T + a["bias"]
    return (2 * y - 1) * (logits[:, 1] - logits[:, 0])


def score(case, x, y, triple, values):
    edits = [(n, i, float(np.float32(v))) for (n, i), v in zip(triple, values)]
    accuracy = float((tf.predict(tf.apply(case["arrays"], edits), x) == y).mean())
    cost = sum(abs(v - float(case["arrays"][n].flat[i])) / case["scale"][n] for n, i, v in edits)
    return (accuracy, -float(cost)), edits


def refine(case, x, y, triple, best):
    """Nelder-Mead on mean(relu(EPS - signed margin)), from the best grid point."""
    scale = np.array([case["scale"][n] for n, _ in triple])
    start = np.array([v for _, _, v in best[1]])
    loss = lambda z: float(np.maximum(EPS - signed_margins(case, triple, start + z * scale, x, y), 0).mean())
    result = minimize(loss, np.zeros(3), method="Nelder-Mead",
                      options={"initial_simplex": np.vstack([np.zeros(3), 0.02 * np.eye(3)]),
                               "xatol": 1e-6, "fatol": 1e-9, "maxiter": 800})
    return score(case, x, y, triple, start + result.x * scale)


def local(case, x, y, triple, values):
    best = score(case, x, y, triple, values)
    for points, span in ((11, 0.3), (11, 0.06)):
        best = max(best, t3.fit_triple(case, x, y, triple, points, span, center=[v for _, _, v in best[1]]),
                   key=lambda b: b[0])
    return max(best, t3.polish(case, x, y, best), key=lambda b: b[0])


def fit(case, x, y, triple, previous=None):
    best = t3.fine_fit(case, x, y, triple)
    if previous is not None:
        best = max(best, local(case, x, y, triple, [v for _, _, v in previous[1]]), key=lambda b: b[0])
    return max(best, refine(case, x, y, triple, best), key=lambda b: b[0])


def run(case, candidates, silent):
    width, faults = case["width"], case["faults"]
    told_triple = tuple(tuple(f[:2]) for f in faults)
    visible = [k for k in range(len(faults)) if k not in silent]
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    rng = np.random.default_rng([width, case["seed"], 404])
    x, y, history, fits, told = px, py, [], {}, None
    for round_ in range(aq.ROUNDS + 1):
        fits = {t: fit(case, x, y, t, fits.get(t)) for t in candidates}
        told = fit(case, x, y, told_triple, told)
        winner = max(fits, key=lambda t: fits[t][0])
        (accuracy, _), edits = fits[winner]
        found = t3.hits(winner, faults, width)
        true_scores = [s[0] for t, (s, _) in fits.items() if len(t3.hits(t, faults, width)) == 3]
        history.append(dict(tf.outcome(case, edits, fx, fy, frepair, inputs, teacher_labels),
                            queries=len(x) - len(px), faults_found=len(found),
                            found_visible=all(k in found for k in visible), practice_accuracy=accuracy,
                            tied_best=sum(s[0] == accuracy for s, _ in fits.values()),
                            true_triple_accuracy=max(true_scores) if true_scores else None,
                            told_accuracy=told[0][0], winner=[list(e) for e in winner]))
        if round_ == aq.ROUNDS:
            break
        pool = rng.uniform(-2, 2, size=(aq.POOL, 2))
        picked, history[-1]["split_of_queries"] = aq.choose_queries(case, fits, pool, rng, "targeted")
        x = np.concatenate([x, picked])
        y = np.concatenate([y, tf.predict(case["teacher"], picked)])
    return history


def analyse(case):
    candidates = tq.committee(case)
    silent = cr.silent_faults(case)
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "silent_faults": silent,
            "committee_has_true_triple": any(len(t3.hits(t, case["faults"], case["width"])) == 3 for t in candidates),
            "targeted": run(case, candidates, silent)}


def summarize(rows):
    out = tq.summarize(rows)
    for k, step in enumerate(out["targeted"]):
        steps = [r["targeted"][k] for r in rows]
        step["winner_fits_all"] = sum(s["practice_accuracy"] == 1.0 for s in steps)
        step["told_fits_all"] = sum(s["told_accuracy"] == 1.0 for s in steps)
        step["true_triple_fits_all"] = sum(s["true_triple_accuracy"] == 1.0 for s in steps
                                           if s["true_triple_accuracy"] is not None)
    return out


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
            case = t3.corrupt(args.width, seed)
            if case is not None:
                rows.append(analyse(case))
                last = rows[-1]["targeted"][-1]
                print(seed, rows[-1]["committee_has_true_triple"], last["faults_found"], last["found_visible"],
                      last["success"], last["practice_accuracy"], last["told_accuracy"], flush=True)
    report = {"schema": "weight-space-warm-triples-v1", "committee": tq.COMMITTEE, "pool": aq.POOL,
              "rounds": aq.ROUNDS, "per_round": aq.PER_ROUND, "eps": EPS, "summary": summarize(rows), "cases": rows}
    (args.output / "warm-triples.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]["targeted"][-1]))


if __name__ == "__main__":
    main()
