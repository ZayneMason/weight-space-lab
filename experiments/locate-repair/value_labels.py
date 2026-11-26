"""Ask more value labels where the values are not settled (roadmap #82).

After #81 the two cases over the 0.01% line, 331 and 400, have every candidate editing the three broken weights (or
their mirrors): the weights are settled and the values are not. This script takes #81's pick (the gated two-round
chooser) on #79's 100 cases and #80's 50, keeps its weights, and tries two ways to move its values, both on #65's
final 1,096 labels plus any new ones:

- ``region``: the way #65's stage spends its last labels. Walk the region of values that fit the labels as well as the
  pick does (#63's hit-and-run walk), ask the 8 pool inputs where the walk's networks split most evenly, refit the
  values, four times (32 labels). Run only where every candidate the chooser saw edits the same weights, as the
  roadmap item says. A ``control`` arm refits on #65's labels alone.
- ``line``: find the direction in which the labels constrain the values least (the smallest singular vector of the
  margin gradients of the 24 labels nearest the boundary) and how far the fit reaches both ways along it. Where that
  direction is flat (its singular value is zero to 1e-6 of the largest), the fitting region is a line segment a random
  walk cannot travel along. Take the segment's middle (``line_middle``), then four rounds of 8 labels where 9 networks
  spaced along the segment split most evenly, each followed by a new segment and its middle (``line_after_8`` and on).
  Run on every case.

    python experiments/locate-repair/value_labels.py --prepare runs/s80
    python experiments/locate-repair/value_labels.py --output runs/v82 --workers 4
    python experiments/locate-repair/value_labels.py --score runs/v82
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import shutil
import time

import numpy as np

import cut_refine as cr
import early_values as ev
import pick_fixes as pf
from gated_duel import load as load_duels, pick as gated_pick
from rebuild_committee import labels
from refine_values import feasible_centre
from wide_compare import EVIDENCE

OUT = EVIDENCE / "value-labels"
KEEP = EVIDENCE / "keep-65" / "keep-65.json"
SAVED = EVIDENCE / "saved-65"
SOURCES = {300: EVIDENCE / "fresh-repair", 400: EVIDENCE / "fresh-50", 500: SAVED}
ROUNDS, PER_ROUND = 4, 8
ACTIVE, EPS, REACH, FLAT, POINTS = 24, 1e-4, 4.0, 1e-6, 9
LINE = pf.LINE
K, M = 32, 1  # #81's rule
wt = pf.wt


def prepare(folder):
    """Copy #80's final #65 labels (from its run folder) into the evidence, as #72's and #78's are."""
    (SAVED / "labels").mkdir(parents=True, exist_ok=True)
    for seed in range(500, 550):
        last = json.loads((folder / f"case-{seed}.json").read_text())["stages_run"] - 1
        shutil.copy(folder / f"case-{seed}.stage{last}.npz", SAVED / "labels" / f"case-{seed}.npz")


def weights(candidate):
    return tuple(sorted(tuple(e) for e in candidate["winner"]))


def cases():
    """#81's pick on every case of #79 and #80, and whether all the chooser's candidates edit the same weights."""
    duels = {r["seed"]: r for rows in load_duels() for r in rows}
    rows = json.loads(KEEP.read_text()) + json.loads((SAVED / "saved-65.json").read_text())
    out = {}
    for r in rows:
        cands = {n: c for n, c in r["candidates"].items() if n != "rebuilt65"}
        p = gated_pick(duels[r["seed"]], K, M)
        out[r["seed"]] = {"pick": p, "winner": cands[p]["winner"], "edits": cands[p]["edits"],
                          "disagreement": cands[p]["disagreement"], "faults_found": cands[p]["faults_found"],
                          "same_weights": len({weights(c) for c in cands.values()}) == 1}
    return out


def final(case, x, y, triple, fit, probe):
    edits, used = pf.final_edits(case, x, y, triple, fit)
    return {"disagreement": ev.aq.disagreement(case, edits, probe), "middle_used": used,
            "practice_accuracy": float(fit[0][0]), "edits": ev.edit_list(edits)}


def segment(case, x, y, triple, z, scale):
    """The least-constrained direction at ``z`` (in spreads) and how far the fit reaches both ways along it."""
    margins = lambda z: wt.signed_margins(case, triple, z * scale, x, y)
    base = margins(z)
    grad = np.stack([(margins(z + EPS * e) - margins(z - EPS * e)) / (2 * EPS) for e in np.eye(3)], axis=1)
    norm = np.linalg.norm(grad, axis=1) + 1e-12
    near = np.argsort(np.where(base > 0, base / norm, np.inf), kind="stable")[:ACTIVE]
    _, sv, vt = np.linalg.svd(grad[near] / norm[near, None])
    d = vt[-1]
    allowed = int((base <= 0).sum())
    fits = lambda t: int((margins(z + t * d) <= 0).sum()) <= allowed

    def edge(sign):
        lo, hi = 0.0, REACH
        if fits(sign * hi):
            return hi
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fits(sign * mid) else (lo, mid)
        return lo

    return d, -edge(-1), edge(1), [float(s) for s in sv]


def line_arm(case, x, y, triple, values, probe, seed):
    scale = np.array([case["scale"][n] for n, _ in triple])
    net = lambda z: [(n, i, float(v)) for (n, i), v in zip(triple, z * scale)]
    z = np.array(values) / scale
    d, a, b, sv = segment(case, x, y, triple, z, scale)
    flat = sv[-1] <= FLAT * sv[0]
    out = {"singular_values": sv, "flat": bool(flat), "segment": [float(a), float(b)]}
    if not flat:
        return out
    rng = np.random.default_rng([seed, 82, 1])
    mid = z + (a + b) / 2 * d
    out["middle"] = {"disagreement": ev.aq.disagreement(case, net(mid), probe), "edits": ev.edit_list(net(mid))}
    out["rounds"] = []
    for _ in range(ROUNDS):
        points = [z + t * d for t in np.linspace(a, b, POINTS)]
        pool, ties = rng.uniform(-2, 2, size=(ev.aq.POOL, 2)), rng.random(ev.aq.POOL)
        votes = np.stack([ev.tf.predict(ev.tf.apply(case["arrays"], net(p)), pool) for p in points])
        split = ev.aq.committee_split(votes)
        picked = np.lexsort((ties, -split))[:PER_ROUND]
        x = np.concatenate([x, pool[picked]])
        y = np.concatenate([y, ev.tf.predict(case["teacher"], pool[picked])])
        z = mid
        d, a, b, sv = segment(case, x, y, triple, z, scale)
        mid = z + (a + b) / 2 * d
        out["rounds"].append({"labels": int(len(x)), "splitting": int((split > 0).sum()), "segment": [float(a), float(b)],
                              "singular_values": sv, "disagreement": ev.aq.disagreement(case, net(mid), probe),
                              "edits": ev.edit_list(net(mid))})
    return out


def region_arm(case, x, y, triple, values, probe, seed):
    fit = ev.tc.refit(case, x, y, triple, wt.score(case, x, y, triple, values))
    out = {"control": final(case, x, y, triple, fit, probe), "rounds": []}
    rng = np.random.default_rng([seed, 82])
    for k in range(ROUNDS):
        samples = []
        feasible_centre(case, x, y, triple, [v for _, _, v in fit[1]], np.random.default_rng([seed, 82, k]),
                        steps=1600, burn=400, samples=samples)
        pool, ties = rng.uniform(-2, 2, size=(ev.aq.POOL, 2)), rng.random(ev.aq.POOL)
        split = ev.region_split(case, triple, samples, pool)
        picked = np.lexsort((ties, -split))[:PER_ROUND]
        x = np.concatenate([x, pool[picked]])
        y = np.concatenate([y, ev.tf.predict(case["teacher"], pool[picked])])
        fit = ev.tc.refit(case, x, y, triple, fit)
        out["rounds"].append({"labels": int(len(x)), "splitting": int((split > 0).sum()),
                              **final(case, x, y, triple, fit, probe)})
    return out


def run_seed(args):
    seed, chosen, out = args
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.perf_counter()
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, y = labels(case, seed, SOURCES[seed // 100 * 100])
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    triple = tuple(tuple(e) for e in chosen["winner"])
    values = [v for _, _, v in chosen["edits"]]
    row = {"seed": seed, "chosen": {k: chosen[k] for k in ("pick", "disagreement", "faults_found", "same_weights")},
           "winner": chosen["winner"], "line": line_arm(case, x, y, triple, values, probe, seed)}
    row["line_seconds"] = round(time.perf_counter() - start, 1)
    if chosen["same_weights"]:
        row["region"] = region_arm(case, x, y, triple, values, probe, seed)
    row["seconds"] = round(time.perf_counter() - start, 1)
    path.write_text(json.dumps(row) + "\n")
    return seed


def compare(seeds, d0, d):
    return {"under_0.01%": sum(d[s] < LINE for s in seeds), "median": float(np.median([d[s] for s in seeds])),
            "better": sorted(s for s in seeds if d[s] < d0[s]), "worse": sorted(s for s in seeds if d[s] > d0[s]),
            "gained": sorted(s for s in seeds if d[s] < LINE <= d0[s]),
            "lost": sorted(s for s in seeds if d0[s] < LINE <= d[s]),
            "over_line": {str(s): d[s] for s in seeds if d[s] >= LINE}}


def score(rows):
    seeds = sorted(rows)
    d0 = {s: rows[s]["chosen"]["disagreement"] for s in seeds}
    flat = [s for s in seeds if rows[s]["line"]["flat"]]
    same = [s for s in seeds if rows[s]["chosen"]["same_weights"]]
    out = {"seeds": seeds, "core_hours": sum(rows[s]["seconds"] for s in seeds) / 3600,
           "core_hours_line": sum(rows[s]["line_seconds"] for s in seeds) / 3600,
           "chosen": compare(seeds, d0, d0), "flat": flat, "same_weights": same,
           "flat_and_same": sorted(set(flat) & set(same)),
           "flat_by_pick_faults": {str(k): sum(rows[s]["chosen"]["faults_found"] == k for s in flat) for k in (1, 2, 3)}}
    line = {"line_middle": {s: rows[s]["line"]["middle"]["disagreement"] if s in flat else d0[s] for s in seeds}}
    for k in range(ROUNDS):
        line[f"line_after_{(k + 1) * PER_ROUND}"] = {
            s: rows[s]["line"]["rounds"][k]["disagreement"] if s in flat else d0[s] for s in seeds}
    for name, d in line.items():
        out[name] = compare(seeds, d0, d)
    region = {"control": {s: rows[s]["region"]["control"]["disagreement"] for s in same}}
    for k in range(ROUNDS):
        region[f"region_after_{(k + 1) * PER_ROUND}"] = {s: rows[s]["region"]["rounds"][k]["disagreement"] for s in same}
    for name, d in region.items():
        out[name] = compare(same, d0, d)
    out["region_splitting"] = {str(s): [r["splitting"] for r in rows[s]["region"]["rounds"]] for s in same}
    out["segment"] = {str(s): rows[s]["line"]["segment"] for s in seeds}
    out["per_case_flat"] = {str(s): [d0[s], line["line_middle"][s]] + [line[f"line_after_{(k + 1) * PER_ROUND}"][s]
                                                                       for k in range(ROUNDS)] for s in flat}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prepare", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare)
        return
    chosen = cases()
    if a.score:
        rows = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in (a.only or sorted(chosen))}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "value-labels.json").write_text(json.dumps([rows[s] for s in sorted(rows)]) + "\n")
        result = score(rows)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("per_case_flat", "segment", "region_splitting")},
                         indent=1))
        return
    cr.install("columns")
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = [(s, chosen[s], str(a.output)) for s in (a.only or sorted(chosen))]
    jobs.sort(key=lambda j: not j[1]["same_weights"])  # the long region arm first
    print("jobs", len(jobs), flush=True)
    with Pool(a.workers) as pool:  # forked after the install above, so every worker runs the cheap refine
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
