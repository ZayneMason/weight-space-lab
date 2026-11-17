"""Roadmap #39: switch to value queries once the committee agrees.

#38's best arm (refined pair fit plus value queries) gives half of every round's labels to value
queries, which cost one case of weight-finding (22 of 25 every visible fault, against 23). #37's
logs show why a switch could keep both: after a few rounds the committee (the pairs within 0.5% of
the best practice accuracy) labels every input in the pool the same way, so its "targeted" labels
after that are effectively random. Two rules, both on #38's refined fit:

- **switch:** while any pool input splits the committee, all 8 labels are targeted; once none does,
  all 8 are value queries.
- **fill:** each round, targeted labels only on inputs that split the committee (most even first),
  and value queries for the rest of the 8.

Every round's winner is probed on a million inputs, as in #37 and #38.

    python experiments/locate-repair/switch_queries.py --output runs/switch-queries
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("value_queries", HERE / "value_queries.py")
vq = importlib.util.module_from_spec(_spec)
sys.modules["value_queries"] = vq
_spec.loader.exec_module(vq)
bs, rs, lr, tf, aq, ds, cr, js, sc, ps = vq.bs, vq.rs, vq.lr, vq.tf, vq.aq, vq.ds, vq.cr, vq.js, vq.sc, vq.ps

ARMS = ("switch", "fill")
LINES = bs.LINES


def committee_split(case, fits, pool):
    """Per pool input, the committee's minority share (0 = every pair within the margin agrees)."""
    best = max(score[0] for score, _ in fits.values())
    committee = [edits for score, edits in fits.values() if score[0] >= best - aq.MARGIN]
    return aq.committee_split(np.stack([tf.predict(tf.apply(case["arrays"], e), pool) for e in committee]))


def choose(case, fits, edits, pool, rng, rule, per_round=None):
    """Indices into ``pool`` for this round, and how many of them are value queries."""
    per_round = per_round or aq.PER_ROUND
    split = committee_split(case, fits, pool)
    order = np.lexsort((rng.random(len(pool)), -split))
    splitting = [int(i) for i in order if split[i] > 0][:per_round]
    if rule == "switch":
        if splitting:
            return [int(i) for i in order[:per_round]], 0
        targeted = []
    else:
        targeted = splitting
    rest = np.setdiff1d(np.arange(len(pool)), targeted)
    value = [int(rest[i]) for i in vq.value_queries(case, edits, pool[rest], rng, per_round - len(targeted))]
    filler = [int(i) for i in order if int(i) not in set(targeted) | set(value)][:per_round - len(targeted) - len(value)]
    return targeted + value + filler, len(value)


def run(case, candidates, rule, probe):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    rng = np.random.default_rng([width, case["seed"], 404])
    silent = cr.silent_faults(case)
    x, y, history = px, py, []
    for round_ in range(aq.ROUNDS + 1):
        fits = {pair: vq.fit(case, x, y, pair, "refine") for pair in candidates}
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
        chosen, history[-1]["value_queries"] = choose(case, fits, edits, pool, rng, rule)
        picked = pool[chosen]
        x = np.concatenate([x, picked])
        y = np.concatenate([y, tf.predict(case["teacher"], picked)])
    return history


def analyse(case, arms=ARMS):
    px, py, _ = lr.split(case["practice"])
    pairs = sc.ranked_pairs(case["arrays"], px, py)
    order = js.joint_order(js.joint_scores(case, px, py, pairs, points=ds.PAIR_POINTS))
    candidates = [pairs[k] for k in order[:js.KEEP]]
    probe = bs.make_probe(case, bs.PROBE)
    row = {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "sizes": case["sizes"]}
    for arm in arms:
        start = time.time()
        row[arm] = run(case, candidates, arm, probe)
        row[f"{arm}_seconds"] = round(time.time() - start, 1)
    return row


def summarize(rows, arms=ARMS):
    out = {"cases": len(rows)}
    for arm in arms:
        out[arm] = []
        for k in range(len(rows[0][arm])):
            steps = [r[arm][k] for r in rows]
            d = [s["disagreement"] for s in steps]
            entry = {"queries": steps[0]["queries"], "all_entries": sum(s["faults_found"] == 2 for s in steps),
                     "found_visible": sum(s["found_visible"] for s in steps), "success": sum(s["success"] for s in steps),
                     "disagreement_median": round(float(np.median(d)), 6),
                     "value_queries": sum(s.get("value_queries", 0) for s in steps)}
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
    report = {"schema": "weight-space-switch-queries-v1", "width": args.width, "sizes": [rs.LOW, rs.HIGH],
              "delta": vq.DELTA, "probe": bs.PROBE, "pool": aq.POOL, "arms": list(args.arms),
              "summary": summarize(rows, tuple(args.arms)), "cases": rows}
    (args.output / "switch-queries.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({a: report["summary"][a][-1] for a in args.arms}))


if __name__ == "__main__":
    main()
