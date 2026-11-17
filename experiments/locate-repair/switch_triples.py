"""Roadmap #41: #39's switch rule for three broken weights.

#40 gave half of every round's labels to value queries around the winning triple: 24 of 25 three-fault
repairs then behaved like the original network (22 in #37), but every visible fault dropped from 19
to 17. For two faults, #39's switch rule kept both: all 8 labels go to the committee's most even
splits while any pool input splits it (the triples within 0.5% of the best practice accuracy), and
all 8 go to value queries once none does. This runs that rule on #40's cases.

#40's reference fit of the true triple stuck at about 2% in seed 2 while the winner, the same triple,
ended at 0.0015%. So the reference is now started the way the committee starts each of its members
(a coarse 7-point fit, then #29's continuous fits from several starts), then refit each round as they
are. #40's reference is kept alongside for comparison.

    python experiments/locate-repair/switch_triples.py --output runs/switch-triples
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


tv = _load("triple_values", "triple_values.py")
ss = _load("switch_queries", "switch_queries.py")
vq, bs, tc = tv.vq, tv.bs, tv.tc
rs, t3, lr, cr, tf, aq, ct, wt = tv.rs, tv.t3, tv.lr, tv.cr, tv.tf, tv.aq, tv.ct, tv.wt
WIDTH, LINES = tv.WIDTH, tv.LINES


def told_start(case, x, y, triple):
    """The true triple fitted exactly as the committee fits each of its members."""
    return ct.best_of(*ct.continuous_fits(case, x, y, triple, t3.fit_triple(case, x, y, triple, 7)))


def run(case, members, fits, silent, probe, chooser=None):
    """``chooser(case, fits, edits, pool, rng)`` returns (picked, info); the default is #39's switch rule."""
    width, faults = case["width"], case["faults"]
    told_triple = tuple(tuple(f[:2]) for f in faults)
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = tf.predict(case["teacher"], inputs)
    rng = np.random.default_rng([width, case["seed"], 404])
    x, y, history = px, py, []
    told = told_start(case, x, y, told_triple)
    chained = wt.refine(case, x, y, told_triple, wt.score(case, x, y, told_triple, ct.current_values(case, told_triple)))
    for round_ in range(aq.ROUNDS + 1):
        if round_:
            fits = {t: tc.refit(case, x, y, t, fits[t]) for t in members}
            told, chained = tc.refit(case, x, y, told_triple, told), tc.refit(case, x, y, told_triple, chained)
        winner = max(members, key=lambda t: fits[t][0])
        (accuracy, _), edits = fits[winner]
        history.append(dict(tf.outcome(case, edits, fx, fy, frepair, inputs, teacher_labels),
                            queries=len(x) - len(px), faults_found=len(t3.hits(winner, faults, width)),
                            found_visible=cr.found_visible(winner, faults, width, silent), practice_accuracy=accuracy,
                            tied_best=sum(fits[t][0][0] == accuracy for t in members), winner=[list(e) for e in winner],
                            disagreement=aq.disagreement(case, edits, probe),
                            told_disagreement=aq.disagreement(case, told[1], probe),
                            chained_told_disagreement=aq.disagreement(case, chained[1], probe)))
        if round_ == aq.ROUNDS:
            break
        pool = rng.uniform(-2, 2, size=(aq.POOL, 2))
        if chooser is None:
            picked, history[-1]["value_queries"] = ss.choose(case, fits, edits, pool, rng, "switch")
        else:
            picked, info = chooser(case, fits, edits, pool, rng)
            history[-1].update(info)
        x = np.concatenate([x, pool[picked]])
        y = np.concatenate([y, tf.predict(case["teacher"], pool[picked])])
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
            "switch": history, "seconds": round(time.time() - start, 1)}


def summarize(rows):
    out = []
    for k in range(len(rows[0]["switch"])):
        steps = [r["switch"][k] for r in rows]
        d = [s["disagreement"] for s in steps]
        told, chained = [s["told_disagreement"] for s in steps], [s["chained_told_disagreement"] for s in steps]
        entry = {"queries": steps[0]["queries"], "all_three": sum(s["faults_found"] == 3 for s in steps),
                 "found_visible": sum(s["found_visible"] for s in steps), "success": sum(s["success"] for s in steps),
                 "disagreement_median": round(float(np.median(d)), 6), "told_median": round(float(np.median(told)), 6),
                 "value_queries": sum(s.get("value_queries", 0) for s in steps)}
        for name, values in (("", d), ("told_", told), ("chained_told_", chained)):
            entry.update({f"{name}under_{line:g}": sum(v < line for v in values) for line in LINES})
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
            last = rows[-1]["switch"][-1]
            print(seed, last["faults_found"], round(last["disagreement"], 6), round(last["told_disagreement"], 6), flush=True)
    report = {"schema": "weight-space-switch-triples-v1", "width": args.width, "sizes": [rs.LOW, rs.HIGH],
              "delta": vq.DELTA, "probe": bs.PROBE, "pool": aq.POOL, "summary": summarize(rows), "cases": rows}
    (args.output / "switch-triples.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"][-1]))


if __name__ == "__main__":
    main()
