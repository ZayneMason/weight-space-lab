"""Score every repair by behaviour, round by round (roadmap #37)

#36 found that scoring a three-fault repair by its entries undercounts: 22 of 25 winners disagree
with the original network on under 0.1% of inputs, against 19 that find every visible fault. This
experiment re-runs the two-fault pipeline (#32's 25-point pair screen and #22's targeted examples),
on random sizes and on the fixed 1.6-spread control, and the three-fault pipeline (#34), and after
every round measures the winner's disagreement with the teacher on a million fresh inputs.

    python experiments/locate-repair/behaviour_scores.py --mode pairs --seeds 25 --output runs/behaviour
    python experiments/locate-repair/behaviour_scores.py --mode pairs --fixed 1.6 --seeds 25 --output runs/behaviour
    python experiments/locate-repair/behaviour_scores.py --mode triples --seeds 25 --output runs/behaviour
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


tc = _load("triple_committee", "triple_committee.py")
rs, t3, lr, cr, tf, aq, ds = tc.rs, tc.t3, tc.lr, tc.cr, tc.tf, tc.aq, tc.ds

PROBE = 1_000_000
LINES = (1e-2, 1e-3, 1e-4)


def make_probe(case, size=PROBE):
    x = np.random.default_rng([case["width"], case["seed"], 36]).uniform(-2, 2, size=(size, 2))
    return x, tf.predict(case["teacher"], x)


def analyse(case, mode, probe_size=PROBE):
    probe = make_probe(case, probe_size)
    corrupted = float((tf.predict(case["arrays"], probe[0]) != probe[1]).mean())
    start = time.time()
    if mode == "pairs":
        row = rs.pair_case(case, ds.PAIR_POINTS, probe)
    else:
        px, py, _ = lr.split(case["practice"])
        members, fits, count = tc.committee(case, px, py)
        silent = cr.silent_faults(case)
        row = {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "sizes": case["sizes"],
               "silent_faults": silent, "triples": count,
               "committee_has_true_triple": any(len(t3.hits(t, case["faults"], case["width"])) == 3 for t in members),
               "targeted": tc.run(case, members, fits, silent, "uniform", probe=probe)}
    row["corrupted_disagreement"] = corrupted
    row["seconds"] = round(time.time() - start, 1)
    return row


def summarize(rows, mode):
    entries = 2 if mode == "pairs" else 3
    out = {"cases": len(rows), "corrupted_disagreement_median": float(np.median([r["corrupted_disagreement"] for r in rows])),
           "targeted": []}
    for k in range(len(rows[0]["targeted"])):
        steps = [r["targeted"][k] for r in rows]
        d = np.array([s["disagreement"] for s in steps])
        step = {"queries": steps[0]["queries"], "all_entries": sum(s["faults_found"] == entries for s in steps),
                "found_visible": sum(s["found_visible"] for s in steps), "success": sum(s["success"] for s in steps),
                "disagreement_median": float(np.median(d))}
        for line in LINES:
            step[f"under_{line:g}"] = int((d < line).sum())
        out["targeted"].append(step)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("pairs", "triples"), required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--fixed", type=float, help="every fault this size instead of a random one (a control)")
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    width = args.width or (32 if args.mode == "pairs" else 16)
    bounds = (args.fixed, args.fixed) if args.fixed is not None else (rs.LOW, rs.HIGH)
    tag = f"{args.mode}-fixed{args.fixed:g}" if args.fixed is not None else args.mode
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = sorted((r for p in args.merge for r in json.loads(p.read_text())["cases"]), key=lambda r: r["seed"])
    else:
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            saved = args.output / f"{tag}-case-{seed}.json"
            if saved.exists():
                rows.append(json.loads(saved.read_text()))
                continue
            base = (tf.corrupt if args.mode == "pairs" else t3.corrupt)(width, seed)
            case = rs.resize(base, *bounds) if base is not None else None
            if case is None:
                continue
            rows.append(analyse(case, args.mode))
            saved.write_text(json.dumps(rows[-1]) + "\n")
            last = rows[-1]["targeted"][-1]
            print(seed, last["faults_found"], last["found_visible"], round(last["disagreement"], 6), rows[-1]["seconds"],
                  flush=True)
    report = {"schema": "weight-space-behaviour-scores-v1", "mode": args.mode, "width": width, "sizes": list(bounds),
              "probe": PROBE, "summary": summarize(rows, args.mode), "cases": rows}
    (args.output / f"behaviour-{tag}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]["targeted"][-1]))


if __name__ == "__main__":
    main()
