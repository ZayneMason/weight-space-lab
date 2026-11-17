"""Roadmap #40: value queries for three broken weights.

#38 closed most of the two-fault value gap with a refined fit plus labels aimed at the winner's
values. #34's three-fault repairs already refit each triple with #29's continuous margin fit, so this
experiment adds only the queries: each round, 4 of the 8 labels go to inputs where the winning triple
and a copy of it with one of its three values nudged by 0.01 spreads disagree (the six copies in turn),
and the other 4 to the committee's most even splits, as in #34. Everything else is #37's three-fault
setup (H = 16, random sizes, committee of 200 triples, uniform pool, a million-input probe after every
round), so #37 is the baseline. The true triple is refitted on the same labels alongside, as a floor.

    python experiments/locate-repair/triple_values.py --output runs/triple-values
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
bs = vq.bs
tc = bs.tc
rs, t3, lr, cr, tf, aq, ct, wt = tc.rs, tc.t3, tc.lr, tc.cr, tc.tf, tc.aq, tc.ct, tc.wt

WIDTH = 16
LINES = bs.LINES


def choose(case, fits, edits, pool, rng, value_share=0.5):
    """Value queries around the winner's ``edits`` for ``value_share`` of the round, targeted for the rest."""
    count = int(round(aq.PER_ROUND * value_share))
    chosen = vq.value_queries(case, edits, pool, rng, count)
    rest = np.delete(pool, chosen, axis=0)
    more, _ = aq.choose_queries(case, fits, rest, rng, "targeted")
    return np.concatenate([pool[chosen], more[:aq.PER_ROUND - len(chosen)]]), len(chosen)


def run(case, members, fits, silent, probe):
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
            fits = {t: tc.refit(case, x, y, t, fits[t]) for t in members}
            told = tc.refit(case, x, y, told_triple, told)
        winner = max(members, key=lambda t: fits[t][0])
        (accuracy, _), edits = fits[winner]
        history.append(dict(tf.outcome(case, edits, fx, fy, frepair, inputs, teacher_labels),
                            queries=len(x) - len(px), faults_found=len(t3.hits(winner, faults, width)),
                            found_visible=cr.found_visible(winner, faults, width, silent), practice_accuracy=accuracy,
                            tied_best=sum(fits[t][0][0] == accuracy for t in members), winner=[list(e) for e in winner],
                            disagreement=aq.disagreement(case, edits, probe),
                            told_disagreement=aq.disagreement(case, told[1], probe)))
        if round_ == aq.ROUNDS:
            break
        pool = rng.uniform(-2, 2, size=(aq.POOL, 2))
        picked, history[-1]["value_queries"] = choose(case, fits, edits, pool, rng)
        x = np.concatenate([x, picked])
        y = np.concatenate([y, tf.predict(case["teacher"], picked)])
    return history


def analyse(case):
    px, py, _ = lr.split(case["practice"])
    start = time.time()
    members, fits, count = tc.committee(case, px, py)
    silent = cr.silent_faults(case)
    history = run(case, members, fits, silent, bs.make_probe(case, bs.PROBE))
    return {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "sizes": case["sizes"],
            "silent_faults": silent, "triples": count,
            "committee_has_true_triple": any(len(t3.hits(t, case["faults"], case["width"])) == 3 for t in members),
            "values": history, "seconds": round(time.time() - start, 1)}


def summarize(rows):
    out = []
    for k in range(len(rows[0]["values"])):
        steps = [r["values"][k] for r in rows]
        d, told = [s["disagreement"] for s in steps], [s["told_disagreement"] for s in steps]
        entry = {"queries": steps[0]["queries"], "all_three": sum(s["faults_found"] == 3 for s in steps),
                 "found_visible": sum(s["found_visible"] for s in steps), "success": sum(s["success"] for s in steps),
                 "disagreement_median": round(float(np.median(d)), 6), "told_median": round(float(np.median(told)), 6),
                 "value_queries": sum(s.get("value_queries", 0) for s in steps)}
        entry.update({f"under_{line:g}": sum(v < line for v in d) for line in LINES})
        entry.update({f"told_under_{line:g}": sum(v < line for v in told) for line in LINES})
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=WIDTH)
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
            saved = args.output / f"case-{seed}.json"
            if saved.exists():
                rows.append(json.loads(saved.read_text()))
                continue
            base = t3.corrupt(args.width, seed)
            case = rs.resize(base) if base is not None else None
            if case is None:
                continue
            rows.append(analyse(case))
            saved.write_text(json.dumps(rows[-1]) + "\n")
            print(seed, rows[-1]["values"][-1]["faults_found"], round(rows[-1]["values"][-1]["disagreement"], 6), flush=True)
    report = {"schema": "weight-space-triple-values-v1", "width": args.width, "sizes": [rs.LOW, rs.HIGH],
              "delta": vq.DELTA, "probe": bs.PROBE, "pool": aq.POOL, "summary": summarize(rows), "cases": rows}
    (args.output / "triple-values.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"][-1]))


if __name__ == "__main__":
    main()
