"""Are the last lookalikes wrong at all? (roadmap #36)

#34 and #35 ended some three-fault cases on a lookalike tied with the true triple on every label
after 64 targeted examples. This experiment builds #34's 200-triple committee once per case, runs
the targeted rounds with the uniform (#34), boundary (#35) and mixed (half of each) query pools, and
then measures, on a million fresh uniform inputs, how often each member tied at the best accuracy
disagrees with the teacher. The same is measured for the true triple's fit, the told triple's fit
and the corrupted network.

    python experiments/locate-repair/lookalikes.py --seeds 25 --output runs/lookalikes
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
rs, t3, lr, cr, tf = tc.rs, tc.t3, tc.lr, tc.cr, tc.tf

POOLS = ("uniform", "boundary", "mixed")
PROBE, CAP = 1_000_000, 20


def disagreement(case, edits, probe, teacher_labels):
    """Share of probe inputs where the edited network's label differs from the teacher's."""
    return float((tf.predict(tf.apply(case["arrays"], edits), probe) != teacher_labels).mean())


def lookalike_report(case, members, fits, told, probe, teacher_labels, cap=CAP):
    width, faults = case["width"], case["faults"]
    best = max(fits[t][0][0] for t in members)
    tied = [t for t in members if fits[t][0][0] == best][:cap]
    rows = [{"triple": [list(e) for e in t], "faults_found": len(t3.hits(t, faults, width)),
             "disagreement": disagreement(case, fits[t][1], probe, teacher_labels)} for t in tied]
    true = [t for t in members if len(t3.hits(t, faults, width)) == 3]
    return {"tied": rows, "tied_count": sum(fits[t][0][0] == best for t in members),
            "true_fit": ({"practice_accuracy": fits[true[0]][0][0],
                          "disagreement": disagreement(case, fits[true[0]][1], probe, teacher_labels)}
                         if true else None),
            "told_fit": {"practice_accuracy": told[0][0],
                         "disagreement": disagreement(case, told[1], probe, teacher_labels)}}


def analyse(case, pools=POOLS, size=tc.COMMITTEE):
    px, py, _ = lr.split(case["practice"])
    start = time.time()
    members, fits, count = tc.committee(case, px, py, size)
    built = time.time() - start
    silent = cr.silent_faults(case)
    probe = np.random.default_rng([case["width"], case["seed"], 36]).uniform(-2, 2, size=(PROBE, 2))
    teacher_labels = tf.predict(case["teacher"], probe)
    out = {"width": case["width"], "seed": case["seed"], "faults": case["faults"], "sizes": case["sizes"],
           "silent_faults": silent, "triples": count,
           "committee_has_true_triple": any(len(t3.hits(t, case["faults"], case["width"])) == 3 for t in members),
           "corrupted_disagreement": float((tf.predict(case["arrays"], probe) != teacher_labels).mean()),
           "seconds_committee": round(built, 1), "pools": {}}
    for pool in pools:
        history, final, told = tc.run(case, members, dict(fits), silent, pool, return_fits=True)
        report = lookalike_report(case, members, final, told, probe, teacher_labels)
        out["pools"][pool] = {"final": history[-1], "history": history, **report}
    out["seconds"] = round(time.time() - start, 1)
    return out


def _median(values):
    return float(np.median(values)) if len(values) else None


def summarize(rows, pools=POOLS):
    out = {"cases": len(rows), "committee_has_true_triple": sum(r["committee_has_true_triple"] for r in rows)}
    for pool in pools:
        steps = [r["pools"][pool] for r in rows]
        winner = [s["tied"][0]["disagreement"] for s in steps]
        worst_tied = [max(t["disagreement"] for t in s["tied"]) for s in steps]
        out[pool] = {"all_three": sum(s["final"]["faults_found"] == 3 for s in steps),
                     "found_visible": sum(s["final"]["found_visible"] for s in steps),
                     "success": sum(s["final"]["success"] for s in steps),
                     "winner_disagreement_median": float(np.median(winner)),
                     "worst_tied_disagreement_median": float(np.median(worst_tied)),
                     "true_fit_disagreement_median": _median([s["true_fit"]["disagreement"] for s in steps
                                                                if s["true_fit"]]),
                     "told_fit_disagreement_median": float(np.median([s["told_fit"]["disagreement"] for s in steps]))}
    out["corrupted_disagreement_median"] = float(np.median([r["corrupted_disagreement"] for r in rows]))
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
            base = t3.corrupt(args.width, seed)
            case = rs.resize(base) if base is not None else None
            if case is None:
                continue
            saved = args.output / f"case-{seed}.json"
            if saved.exists():
                rows.append(json.loads(saved.read_text()))
                continue
            rows.append(analyse(case))
            saved.write_text(json.dumps(rows[-1]) + "\n")
            r = rows[-1]
            print(seed, {p: (r["pools"][p]["final"]["faults_found"], r["pools"][p]["tied_count"],
                             round(max(t["disagreement"] for t in r["pools"][p]["tied"]), 5)) for p in POOLS},
                  r["seconds"], flush=True)
    report = {"schema": "weight-space-lookalikes-v1", "probe": PROBE, "pools": list(POOLS),
              "summary": summarize(rows), "cases": rows}
    (args.output / "lookalikes.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
