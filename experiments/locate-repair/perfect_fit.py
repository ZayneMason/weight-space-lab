"""Make the fitter reach a perfect fit when one exists (roadmap #85).

In 10 of #84's 27 cases the broken weights' original values get every one of #65's final labels right, but the fitter
(``warm_triples.fit``: a grid, two local grids, a polish and a Nelder-Mead hinge refine) leaves 1 or 2 wrong. This adds
one step after it: when a triple's best fit misses between 1 and ``NEAR`` labels, linearise every label's margin in the
three values and solve a small linear program for the smallest change (in tensor spreads) that puts every margin above
a small target, inside a trust region; repeat while it gets fewer labels wrong. The fit with more labels right is kept,
as before.

- ``--true``: on #84's 10 cases, the true triple's fit with and without the step, and the pick's own weights moved by
  the step onto #65's final labels (a control: the picks were fitted on other labels, as #82 found).
- the run: #84's search again on its 27 cases with the step installed in every fit.

    python experiments/locate-repair/perfect_fit.py --true
    python experiments/locate-repair/perfect_fit.py --output runs/p85 --part 0 --parts 4   # parts 0 to 3, one per core
    python experiments/locate-repair/perfect_fit.py --score runs/p85
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog

import keep_one as k
from wide_compare import EVIDENCE

OUT = EVIDENCE / "perfect-fit"
NEAR = 3
TARGET = 2e-4
wt = k.wt
ORIGINAL = wt.fit
CALLS = {"fits": 0, "nudged": 0, "fixed": 0}


def jacobian(case, triple, v, scale, m, x, y, h=1e-5):
    return np.stack([(wt.signed_margins(case, triple, v + h * scale * np.eye(3)[j], x, y) - m) / h
                     for j in range(3)], axis=1)


def nudge(case, x, y, triple, best, rounds=40, radius=0.05):
    """The smallest L1 change, in spreads, that puts every linearised margin above TARGET within ``radius``; where
    that is infeasible, the step that raises the smallest margin most. A step is kept when it leaves fewer labels
    wrong or raises the smallest margin; otherwise the radius halves."""
    scale = np.array([case["scale"][n] for n, _ in triple])
    v = np.array([e[2] for e in best[1]], dtype=np.float64)
    m = wt.signed_margins(case, triple, v, x, y)
    for _ in range(rounds):
        if m.min() > wt.EPS:
            break
        J = jacobian(case, triple, v, scale, m, x, y)
        res = linprog(np.ones(6), A_ub=np.hstack([-J, J]), b_ub=m - TARGET, bounds=[(0, radius)] * 6, method="highs")
        if res.status == 0:
            dz = res.x[:3] - res.x[3:]
        else:
            res = linprog([0, 0, 0, -1], A_ub=np.hstack([-J, np.ones((len(m), 1))]), b_ub=m,
                          bounds=[(-radius, radius)] * 3 + [(None, None)], method="highs")
            if res.status != 0:
                radius /= 2
                continue
            dz = res.x[:3]
        v2 = v + dz * scale
        m2 = wt.signed_margins(case, triple, v2, x, y)
        if (m2 < wt.EPS).sum() < (m < wt.EPS).sum() or m2.min() > m.min():
            v, m = v2, m2
        else:
            radius /= 2
            if radius < 1e-7:
                break
    return wt.score(case, x, y, triple, v)


def fit(case, x, y, triple, previous=None):
    best = ORIGINAL(case, x, y, triple, previous)
    CALLS["fits"] += 1
    missed = int(round((1 - best[0][0]) * len(y)))
    if 0 < missed <= NEAR:
        CALLS["nudged"] += 1
        nudged = nudge(case, x, y, triple, best)
        if nudged[0] > best[0]:
            CALLS["fixed"] += nudged[0][0] == 1.0
            best = nudged
    return best


def install():
    wt.fit = fit


def missed(case, edits, x, y):
    return k.wrong(case, edits, x, y)


def true_check():
    """The true triple fitted with and without the step on #84's 10 cases, and the 27 picks moved onto #65's labels."""
    k.cr.install("columns")
    k.br.install()
    gap = json.loads((k.OUT / "gap.json").read_text())
    rows = {r["seed"]: r for r in json.loads((k.OUT / "keep-one.json").read_text())}
    out = {"true": {}, "pick_moved": {}}
    for s, r in sorted(rows.items()):
        case, x, y = k.load(s)
        probe = k.ev.bs.make_probe(case, k.ev.bs.PROBE)
        sc = np.array([case["scale"][n] for n, _ in case["faults"]])
        if str(s) in gap:
            t = tuple(tuple(e) for e in case["faults"])
            orig = np.array([float(case["teacher"][n].flat[i]) for n, i in t])
            far = lambda e: float(np.abs((np.array([v for _, _, v in e]) - orig) / sc).max())
            start = time.perf_counter()
            before = ORIGINAL(case, x, y, t)
            mid = time.perf_counter()
            after = nudge(case, x, y, t, before)
            out["true"][str(s)] = {
                "before": {"wrong": missed(case, before[1], x, y), "spreads_from_original": far(before[1]),
                           "disagreement": k.ev.aq.disagreement(case, before[1], probe)},
                "after": {"wrong": missed(case, after[1], x, y), "spreads_from_original": far(after[1]),
                          "disagreement": k.ev.aq.disagreement(case, after[1], probe)},
                "pick_disagreement": r["pick"]["disagreement"], "pick_faults_found": r["pick"]["faults_found"],
                "fit_seconds": round(mid - start, 2), "nudge_seconds": round(time.perf_counter() - mid, 2)}
        edits = [tuple(e) for e in r["pick"]["edits"]]
        t = tuple(e[:2] for e in edits)
        moved = nudge(case, x, y, t, ((0, 0), edits))
        out["pick_moved"][str(s)] = {"wrong_before": r["pick_wrong"], "wrong_after": missed(case, moved[1], x, y),
                                     "before": r["pick"]["disagreement"],
                                     "after": k.ev.aq.disagreement(case, moved[1], probe)}
    return out


def run(out, part, parts):
    k.cr.install("columns")
    k.br.install()
    install()
    out.mkdir(parents=True, exist_ok=True)
    picks = k.picks()
    seeds = json.loads((k.OUT / "check.json").read_text())["seeds"]
    seeds = sorted(seeds, key=lambda s: (s != 641, s))
    for seed in seeds[part::parts]:
        if (out / f"case-{seed}.json").exists():
            continue
        start = time.time()
        CALLS.update(fits=0, nudged=0, fixed=0)
        k.run_seed(seed, picks[seed], out)
        path = out / f"case-{seed}.json"
        row = json.loads(path.read_text())
        row.update(nudge_calls=dict(CALLS))
        path.write_text(json.dumps(row) + "\n")
        print("done", seed, round(time.time() - start), time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


def score(folder):
    rows, out = k.score(folder)
    before = {r["seed"]: r for r in json.loads((k.OUT / "keep-one.json").read_text())}
    d = lambda r: k.result_of(r, r["chooser"]["pick"])
    out["vs_84"] = {
        "more_perfect": sorted(s for s in rows if len(rows[s]["perfect"]) > len(before[s]["perfect"])),
        "fewer_perfect": sorted(s for s in rows if len(rows[s]["perfect"]) < len(before[s]["perfect"])),
        "closer": sorted(s for s in rows if d(rows[s])[0] < d(before[s])[0]),
        "further": sorted(s for s in rows if d(rows[s])[0] > d(before[s])[0]),
        "true_perfect_84": sum(r["true_perfect"] for r in before.values()),
        "core_hours_84": sum(r["seconds"] for r in before.values()) / 3600}
    out["nudges"] = {key: sum(r["nudge_calls"][key] for r in rows.values()) for key in ("fits", "nudged", "fixed")}
    return rows, out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--true", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.true:
        result = true_check()
        (OUT / "true.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    if a.score:
        rows, result = score(a.score)
        (OUT / "perfect-fit.json").write_text(json.dumps([rows[s] for s in sorted(rows)]) + "\n")
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({key: v for key, v in result.items() if key != "per_case"}, indent=1))
        return
    run(a.output, a.part, a.parts)


if __name__ == "__main__":
    main()
