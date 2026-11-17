"""Roadmap #38: queries that pin down the winning pair repair's values.

#37 found two-fault repairs that edit the exact broken weights and still disagree with the original
network on 0.13% to 0.17% of inputs, because the values are a few hundredths of a spread off. #22's
targeted labels split the committee over *which* weights to edit, so once the winner is right they
stop helping. Two changes, alone and together, on #35's random-size two-fault cases:

- **Value queries.** Each round, half the labels still go to the committee's most even splits; the
  other half go to inputs where the winner and a nudged copy of it disagree (each of its two values
  moved by +/- DELTA spreads, taken in turn), which says which way each value should move.
- **Middle-of-the-tie fit.** The pair fit takes the best practice accuracy on a 200 x 200 grid and
  breaks ties toward the corrupted values, which puts the repair at the edge of the region that fits
  every label. Instead take the tied grid point nearest the middle of that region.
- **Refined fit.** The grid's step is 0.025 spreads, and a one-trial check on seed 5 showed the
  nearest grid point to the truth still disagrees on 0.17%. So refine: a 61 x 61 grid within 1.5
  coarse steps of the grid winner, then the middle of the tie there.

Every round's winner is probed on a million inputs, as in #37.

    python experiments/locate-repair/value_queries.py --output runs/value-queries
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("behaviour_scores", HERE / "behaviour_scores.py")
bs = importlib.util.module_from_spec(_spec)
sys.modules["behaviour_scores"] = bs
_spec.loader.exec_module(bs)
rs, lr, tf, aq, ds, cr = bs.rs, bs.lr, bs.tf, bs.aq, bs.ds, bs.cr
js, sc, ps = rs.js, rs.sc, rs.ps

DELTA = 0.01  # spreads; #37's right-weight misses were 0.01 to 0.03 spreads off
STEP = 5 / 199  # the 200-point grid's step, in spreads
FINE, FINE_SPAN = 61, 1.5  # local refinement: 61 x 61 points within 1.5 coarse steps of the grid winner
ARMS = {"value": ("value", "edge"), "midtie": ("committee", "middle"), "refine": ("committee", "refine"),
        "refine_value": ("value", "refine")}
LINES = bs.LINES


def joint_grid(case, px, py, pair, points=200, span=2.5):
    """tf.joint's grid, returned whole: accuracy[i, j] for value offsets[i] on pair[0], offsets[j] on pair[1]."""
    arrays, scale = case["arrays"], case["scale"]
    (n1, i1), (n2, i2) = pair
    c1, c2 = float(arrays[n1].flat[i1]), float(arrays[n2].flat[i2])
    offsets = np.linspace(-span, span, points)
    g1, g2 = c1 + offsets * scale[n1], c2 + offsets * scale[n2]
    accuracy = np.empty((points, points))
    for k, v in enumerate(g1):
        trial = {n: a.copy() for n, a in arrays.items()}
        trial[n1].flat[i1] = np.float32(v)
        accuracy[k] = (lr.Evaluator(trial, px).predict(n2, i2, g2) == py).mean(axis=1)
    return accuracy, g1, g2


def local_grid(case, px, py, edits, points=FINE, span=FINE_SPAN * STEP):
    """Accuracy on a fine grid centred on ``edits`` (two entries), within ``span`` spreads either way."""
    arrays, scale = case["arrays"], case["scale"]
    (n1, i1, v1), (n2, i2, v2) = edits
    offsets = np.linspace(-span, span, points)
    g1, g2 = v1 + offsets * scale[n1], v2 + offsets * scale[n2]
    accuracy = np.empty((points, points))
    for k, v in enumerate(g1):
        trial = {n: a.copy() for n, a in arrays.items()}
        trial[n1].flat[i1] = np.float32(v)
        accuracy[k] = (lr.Evaluator(trial, px).predict(n2, i2, g2) == py).mean(axis=1)
    return accuracy, g1, g2


def middle_of_tie(accuracy):
    """The grid point at the best accuracy that is nearest the centroid of all such points."""
    tied = np.argwhere(accuracy == accuracy.max())
    centre = tied.mean(axis=0)
    return tuple(tied[np.argmin(((tied - centre) ** 2).sum(axis=1))])


def fit(case, x, y, pair, how):
    if how == "edge" or len(pair) == 1:
        return ps.fine_fit(case, x, y, pair)
    (n1, i1), (n2, i2) = pair
    if how == "refine":
        _, coarse = ps.fine_fit(case, x, y, pair)
        accuracy, g1, g2 = local_grid(case, x, y, coarse)
    else:
        accuracy, g1, g2 = joint_grid(case, x, y, list(pair))
    i, j = middle_of_tie(accuracy)
    edits = [(n1, i1, float(g1[i])), (n2, i2, float(g2[j]))]
    cost = sum(abs(v - float(case["arrays"][n].flat[k])) / case["scale"][n] for n, k, v in edits)
    return (float(accuracy[i, j]), -cost), edits


def neighbours(case, edits, delta=DELTA):
    """The winner with one of its values moved by +/- delta spreads, for each value in turn."""
    out = []
    for k, (name, index, value) in enumerate(edits):
        for sign in (1, -1):
            moved = list(edits)
            moved[k] = (name, index, value + sign * delta * case["scale"][name])
            out.append(moved)
    return out


def value_queries(case, edits, pool, rng, count):
    """Round-robin over the nudged copies: for each, a random pool input it labels differently from the winner."""
    winner = tf.predict(tf.apply(case["arrays"], edits), pool)
    differs = [np.flatnonzero(tf.predict(tf.apply(case["arrays"], n), pool) != winner) for n in neighbours(case, edits)]
    picked, k = [], 0
    while len(picked) < count and any(len(d) for d in differs):
        d = differs[k % len(differs)]
        d = d[~np.isin(d, picked)]
        if len(d):
            picked.append(int(rng.choice(d)))
        differs[k % len(differs)] = d
        k += 1
    return [int(i) for i in picked]


def run(case, candidates, queries, how, probe):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    rng = np.random.default_rng([width, case["seed"], 404])
    silent = cr.silent_faults(case)
    x, y, history = px, py, []
    for round_ in range(aq.ROUNDS + 1):
        fits = {pair: fit(case, x, y, pair, how) for pair in candidates}
        winner = max(fits, key=lambda p: fits[p][0])
        score, edits = fits[winner]
        history.append(dict(tf.outcome(case, edits, fx, fy, frepair, inputs, teacher_labels),
                            queries=len(x) - len(px), faults_found=ps.found(winner, faults, width),
                            found_visible=cr.found_visible([list(e) for e in winner], faults, width, silent),
                            practice_accuracy=score[0], tied_best=sum(s[0] == score[0] for s, _ in fits.values()),
                            winner=[list(e) for e in winner], disagreement=aq.disagreement(case, edits, probe)))
        if round_ == aq.ROUNDS:
            break
        pool = rng.uniform(-2, 2, size=(aq.POOL, 2))
        if queries == "committee":
            picked, _ = aq.choose_queries(case, fits, pool, rng, "targeted")
        else:
            chosen = value_queries(case, edits, pool, rng, aq.PER_ROUND // 2)
            rest = np.delete(pool, chosen, axis=0)
            more, _ = aq.choose_queries(case, fits, rest, rng, "targeted")
            history[-1]["value_queries"] = len(chosen)
            picked = np.concatenate([pool[chosen], more[:aq.PER_ROUND - len(chosen)]])
        x = np.concatenate([x, picked])
        y = np.concatenate([y, tf.predict(case["teacher"], picked)])
    return history


def analyse(case, arms=tuple(ARMS)):
    px, py, _ = lr.split(case["practice"])
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    order = js.joint_order(js.joint_scores(case, px, py, pairs, points=ds.PAIR_POINTS))
    candidates = [pairs[k] for k in order[:js.KEEP]]
    probe = bs.make_probe(case, bs.PROBE)
    row = {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "sizes": case["sizes"],
           "candidates_have_true_pair": any(ps.found(p, case["faults"], case["width"]) == 2 for p in candidates)}
    for arm in arms:
        start = time.time()
        row[arm] = run(case, candidates, *ARMS[arm], probe)
        row[f"{arm}_seconds"] = round(time.time() - start, 1)
    return row


def summarize(rows, arms=tuple(ARMS)):
    out = {"cases": len(rows)}
    for arm in arms:
        out[arm] = []
        for k in range(len(rows[0][arm])):
            steps = [r[arm][k] for r in rows]
            d = [s["disagreement"] for s in steps]
            entry = {"queries": steps[0]["queries"], "all_entries": sum(s["faults_found"] == 2 for s in steps),
                     "found_visible": sum(s["found_visible"] for s in steps), "success": sum(s["success"] for s in steps),
                     "disagreement_median": round(float(np.median(d)), 6)}
            entry.update({f"under_{line:g}": sum(v < line for v in d) for line in LINES})
            out[arm].append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            saved = args.output / f"case-{seed}.json"
            if saved.exists():
                rows.append(json.loads(saved.read_text()))
                continue
            base = tf.corrupt(args.width, seed)
            case = rs.resize(base) if base is not None else None
            if case is None:
                continue
            rows.append(analyse(case, tuple(args.arms)))
            saved.write_text(json.dumps(rows[-1]) + "\n")
            print(seed, {a: (rows[-1][a][-1]["faults_found"], round(rows[-1][a][-1]["disagreement"], 6)) for a in args.arms},
                  flush=True)
    report = {"schema": "weight-space-value-queries-v1", "width": args.width, "sizes": [rs.LOW, rs.HIGH], "delta": DELTA,
              "probe": bs.PROBE, "pool": aq.POOL, "arms": {a: list(ARMS[a]) for a in args.arms},
              "summary": summarize(rows, tuple(args.arms)), "cases": rows}
    (args.output / "value-queries.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({a: report["summary"][a][-1] for a in args.arms}))


if __name__ == "__main__":
    main()
