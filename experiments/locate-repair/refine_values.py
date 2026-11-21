"""Refine the values once the repair has the right weights (roadmap #63).

17 of the 33 width-16 cases over 0.01% in seeds 100 to 199 (#62) edit exactly the three broken weights, so their
gap is in the values. The repair's margin fit stops at the first point that fits every label and keeps the
smallest change, which may sit at the edge of the region that fits. This script replays the kept stage of every
exact case to recover its final labelled set (practice plus 64 targeted labels), then compares the repair's values
with the middle of the region that fits every one of those labels. No new labels and no new search.

    python experiments/locate-repair/refine_values.py --replay 164 --output runs/refine-values   # one per case
    python experiments/locate-repair/refine_values.py --score runs/refine-values
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import early_values as ev
from wide_compare import EVIDENCE, cases

st, tc, t3, rs, lr, bs, aq, cr = ev.st, ev.tc, ev.t3, ev.rs, ev.lr, ev.bs, ev.aq, ev.cr
wt = st.wt
ARMS = ("two-step-unseen/two-step-unseen.json", "third-stage-rule/third-stage-rule.json")
OUT = EVIDENCE / "refine-values"
STEPS, BURN, REACH, BISECT = 400, 100, 0.5, 24
WALKS = {"short": (400, 100, 0), "short_again": (400, 100, 1000), "long": (1600, 400, 2000)}  # steps, burn, rng offset


def exact_cases():
    """Seed -> kept stage for every case whose last stage edits exactly the three broken weights."""
    out = {}
    for path in ARMS:
        for s, row in cases(EVIDENCE / path).items():
            if row["two_step"][-1]["faults_found"] == 3:
                out[s] = len(row["two_step"]) - 1
    return out


def build(seed, width=16):
    return rs.resize(t3.corrupt(width, seed))


def replay(case, stage, early=2):
    """Rerun one stage exactly as ``early_values.analyse`` does, keeping the final labelled set."""
    options = ev.TWO_STEP[stage]
    px, py, _ = lr.split(case["practice"])
    size = {"size": options["committee_size"]} if "committee_size" in options else {}
    members, fits, _ = tc.committee(case, px, py, pairs_kept=options.get("pairs_kept"), **size)
    keep = {}
    history = st.run(case, members, fits, cr.silent_faults(case), bs.make_probe(case, bs.PROBE),
                     chooser=ev.make_chooser(early), keep=keep)
    return history[-1], keep


def feasible_centre(case, x, y, triple, values, rng, steps=STEPS, burn=BURN, reach=REACH):
    """Mean of a hit-and-run walk inside the region of value triples that miss no more labels than the repair's
    values do (for nearly every case, the region that fits every label), started at those values. Each step picks
    a random direction in units of each tensor's spread, finds how far the region reaches both ways along it by
    bisection (at most ``reach`` spreads), and moves to a uniform point on that chord. Returns (mean, mean chord
    length in spreads)."""
    scale = np.array([case["scale"][n] for n, _ in triple])
    wrong = lambda z: int((wt.signed_margins(case, triple, z * scale, x, y) <= 0).sum())
    z = np.array(values) / scale
    allowed = wrong(z)
    fits = lambda z: wrong(z) <= allowed

    def edge(d):
        lo, hi = 0.0, reach
        if fits(z + hi * d):
            return hi
        for _ in range(BISECT):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fits(z + mid * d) else (lo, mid)
        return lo

    kept, chords = [], []
    for k in range(burn + steps):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        a, b = -edge(-d), edge(d)
        z = z + rng.uniform(a, b) * d
        if k >= burn:
            kept.append(z.copy())
            chords.append(b - a)
    return np.mean(kept, axis=0) * scale, float(np.mean(chords))


def change_ratios(case, edits):
    """Each edit's change from the broken value as a share of the true change (1 is exact, below 1 undershoots)."""
    out = []
    for n, i, v in edits:
        broken, true = float(case["arrays"][n].flat[i]), float(case["teacher"][n].flat[i])
        if abs(true - broken) > 1e-6:
            out.append((v - broken) / (true - broken))
    return out


def score_case(seed, saved):
    case = build(seed)
    data = np.load(saved / f"labels-{seed}.npz")
    x, y = data["x"], data["y"]
    meta = json.loads((saved / f"replay-{seed}.json").read_text())
    triple = tuple((n, int(i)) for n, i in meta["winner"])
    probe = bs.make_probe(case, bs.PROBE)
    repair = [(n, int(i), float(v)) for n, i, v in meta["edits"]]
    row = {"seed": seed, "stage": meta["stage"], "repair": aq.disagreement(case, repair, probe),
           "saved": meta["disagreement"], "labels": int(len(y)), "repair_ratios": change_ratios(case, repair)}
    for name, (steps, burn, offset) in WALKS.items():
        centre, chord = feasible_centre(case, x, y, triple, [v for _, _, v in repair],
                                        np.random.default_rng(seed + offset), steps=steps, burn=burn)
        (accuracy, _), edits = wt.score(case, x, y, triple, centre)
        row[name] = {"disagreement": aq.disagreement(case, edits, probe), "accuracy": accuracy, "mean_chord": chord,
                     "ratios": change_ratios(case, edits)}
    return row


def summarize(rows, walk):
    pairs = [(r["repair"], r[walk]["disagreement"]) for r in rows]
    line = lambda k, t: sum(p[k] < t for p in pairs)
    return {"cases": len(rows), "repair_under_0.01%": line(0, 1e-4), "centre_under_0.01%": line(1, 1e-4),
            "repair_under_0.1%": line(0, 1e-3), "centre_under_0.1%": line(1, 1e-3),
            "closer": sum(c < a for a, c in pairs), "further": sum(c > a for a, c in pairs),
            "geometric_mean_ratio": float(np.exp(np.mean([np.log(c / a) for a, c in pairs]))),
            "crossed_under_0.01%": sorted(r["seed"] for r, (a, c) in zip(rows, pairs) if a >= 1e-4 > c),
            "crossed_over_0.01%": sorted(r["seed"] for r, (a, c) in zip(rows, pairs) if a < 1e-4 <= c),
            "median_repair": float(np.median([p[0] for p in pairs])),
            "median_centre": float(np.median([p[1] for p in pairs])),
            "median_chord": float(np.median([r[walk]["mean_chord"] for r in rows])),
            "worst_repair": max(p[0] for p in pairs), "worst_centre": max(p[1] for p in pairs)}


CHORD_BANDS = ((0, 0.002), (0.002, 0.005), (0.005, float("inf")))


def by_chord(rows, walk="long"):
    """How the gain depends on how far the region reaches (the long walk's mean chord, in spreads)."""
    out = []
    for lo, hi in CHORD_BANDS:
        band = [r for r in rows if lo <= r[walk]["mean_chord"] < hi]
        ratios = [r[walk]["disagreement"] / r["repair"] for r in band]
        out.append({"from": lo, "to": hi if hi != float("inf") else None, "cases": len(band),
                    "geometric_mean_ratio": float(np.exp(np.mean(np.log(ratios)))) if band else None,
                    "closer": sum(q < 1 for q in ratios),
                    "closer_in_every_walk": sum(all(r[w]["disagreement"] < r["repair"] for w in WALKS) for r in band)})
    return out


def same_direction(rows, a="short", b="short_again"):
    """Cases where two walks with different random draws agree on whether the middle is closer than the repair."""
    return sum((r[a]["disagreement"] < r["repair"]) == (r[b]["disagreement"] < r["repair"]) for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", type=int, nargs="+")
    parser.add_argument("--score", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--list", action="store_true", help="print the exact cases and their kept stage")
    args = parser.parse_args()
    exact = exact_cases()
    if args.list:
        print(" ".join(str(s) for s in sorted(exact)))
        return
    if args.replay:
        args.output.mkdir(parents=True, exist_ok=True)
        for seed in args.replay:
            last, keep = replay(build(seed), exact[seed])
            np.savez(args.output / f"labels-{seed}.npz", x=keep["x"], y=keep["y"])
            meta = {"seed": seed, "stage": exact[seed], "winner": [list(e) for e in keep["winner"]],
                    "edits": [list(e) for e in keep["edits"]], "disagreement": last["disagreement"],
                    "faults_found": last["faults_found"]}
            (args.output / f"replay-{seed}.json").write_text(json.dumps(meta) + "\n")
            print(seed, meta["stage"], meta["disagreement"], flush=True)
        return
    seeds = [s for s in sorted(exact) if (args.score / f"replay-{s}.json").exists()]
    with Pool(args.workers) as pool:
        rows = pool.starmap(score_case, [(s, args.score) for s in seeds])
    result = {"walks": WALKS, "reach": REACH, "replays_match_saved": all(r["repair"] == r["saved"] for r in rows),
              "summary": {w: summarize(rows, w) for w in WALKS}, "same_direction": same_direction(rows),
              "by_chord": by_chord(rows),
              "cases": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
