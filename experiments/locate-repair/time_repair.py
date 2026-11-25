"""Find where the repair's time goes (roadmap #73).

#72 found the whole repair holds on fresh cases (49 of 50 under 0.01%) but costs about 0.75 core-hours a case, and
nobody had measured which part costs what. This script reruns #71's stage (everything after #65's repair) on cases
of #71 and #72 (seeds 250 to 349), from the same saved labels, with each part timed:

- ``committee``: #66's rebuilt committee, #67's refits and #68's tied-label rounds (``tied_labels.run_case``);
- ``restarts``, ``boundary``, ``bigger``: the three fixes, as in #71;
- ``finals``: each candidate's final network (#64's middle walk);
- ``choose``: the disagree chooser's pool, votes and labels;
- ``probe``: scoring on the 1M-point probe, which a real repair does not run, so it is not counted as repair cost.

Inside the parts it also times the steps exclusively (a step's time leaves out the steps it calls): ranking pairs,
joint scores, coarse grid fits, continuous fits, refits, the margin refine (Nelder-Mead on the label margins), the
middle walk and the tied split. #65's own stage is taken from the seconds it recorded.

The roadmap planned all 100 cases and a cheaper order (build restarts and the bigger committee only where #68's pick
misses a label). A one-case pilot showed the three fixes cost under a second each, so the cheaper order could save
almost nothing; the run was cut to a 35-case sample (12 of the 15 slowest cases and 23 others), which is enough to
place the time. The cheaper order is still scored on the same candidates as a side note: its chooser sees only the
candidates that order builds, with its own fresh pool.

    python experiments/locate-repair/time_repair.py --output runs/time --workers 5 --only 250 256 ...
    python experiments/locate-repair/time_repair.py --score runs/time
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import early_values as ev
import pick_fixes as pf
from rebuild_committee import labels
import tied_labels
from wide_compare import EVIDENCE, cases

OUT = EVIDENCE / "time-repair"
SEEDS = tuple(range(250, 350))
CHEAP_SEED = 73  # the cheaper order's chooser draws its own pool, so it is not the full chooser's pool reused
PARTS = ("committee", "restarts", "boundary", "bigger", "finals", "choose")


class Clock:
    """Wall seconds per part, as #71 and #72 recorded them, and per step inside the parts. Probe scoring is pulled out
    of whichever part calls it. Steps are timed exclusively: a step's time leaves out the steps it calls."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.steps = defaultdict(float)
        self.stack = []

    @property
    def probe(self):
        return self.steps["probe"]

    def reset(self):
        self.seconds.clear()
        self.steps.clear()

    @contextmanager
    def part(self, name):
        start, probe = time.perf_counter(), self.probe
        yield
        self.seconds[name] += time.perf_counter() - start - (self.probe - probe)

    def tap(self, fn, step):
        def run(*args, **kwargs):
            self.stack.append(0.0)
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                spent = time.perf_counter() - start
                inner = self.stack.pop()
                self.steps[step] += spent - inner
                if self.stack:
                    self.stack[-1] += spent
        return run


CLOCK = Clock()
# wrap once per process. The probe calls go through the first two names; the rest are the repair's own steps.
tied_labels.score_fit = CLOCK.tap(tied_labels.score_fit, "probe")
ev.aq.disagreement = CLOCK.tap(ev.aq.disagreement, "probe")
ev.tc.sc.ranked_pairs = CLOCK.tap(ev.tc.sc.ranked_pairs, "rank pairs")
ev.tc.js.joint_scores = CLOCK.tap(ev.tc.js.joint_scores, "joint scores")
ev.tc.t3.fit_triple = CLOCK.tap(ev.tc.t3.fit_triple, "coarse fits")
ev.tc.ct.continuous_fits = CLOCK.tap(ev.tc.ct.continuous_fits, "continuous fits")
ev.tc.refit = CLOCK.tap(ev.tc.refit, "refits")
ev.tc.wt.refine = CLOCK.tap(ev.tc.wt.refine, "margin refine")
pf.feasible_centre = CLOCK.tap(pf.feasible_centre, "middle walk")
tied_labels.tied_split = CLOCK.tap(tied_labels.tied_split, "tied split")


def stages_of(seed):
    if seed < 300:
        return cases(pf.SOURCE / "region-labels.json")[seed]["stages_run"]
    rows = json.loads((EVIDENCE / "fresh-repair" / "repair-65.json").read_text())
    return next(r["stages_run"] for r in rows if r["seed"] == seed)


def seconds_65(seed):
    if seed < 300:
        return cases(pf.SOURCE / "region-labels.json")[seed]["seconds"]
    rows = json.loads((EVIDENCE / "fresh-repair" / "repair-65.json").read_text())
    return next(r["seconds"] for r in rows if r["seed"] == seed)


def run_seed(args):
    seed, out = args
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    clock = CLOCK
    clock.reset()
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, y = labels(case, seed) if seed < 300 else labels(case, seed, EVIDENCE / "fresh-repair")
    probe = clock.tap(ev.bs.make_probe, "probe")(case, ev.bs.PROBE)
    options = ev.TWO_STEP[stages_of(seed) - 1]
    keep = {}
    with clock.part("committee"):
        tied_labels.run_case(case, x, y, options, probe, seed, keep=keep)
        members, table, x, y = keep["members"], keep["table"], keep["x"], keep["y"]
        w = pf.pick(members, table, "coarse", case)
    built = {"pick68": (w, table[w], x, y)}
    with clock.part("restarts"):
        t, f = pf.restarts(case, x, y, members, table, w, seed)
        built["restarts"] = (t, f, x, y)
    with clock.part("boundary"):
        built["boundary"] = pf.boundary(case, x, y, w, table[w], seed)
    gate = bool(table[w][0][0] < 1.0)
    if gate:
        with clock.part("bigger"):
            t, f, _ = pf.bigger(case, x, y, options, seed)
            built["bigger"] = (t, f, x, y)
    rows, networks = {}, {}
    for n, (t, f, cx, cy) in built.items():
        before = clock.seconds["finals"]
        with clock.part("finals"):
            edits, used = pf.final_edits(case, cx, cy, t, f)
        networks[n] = edits
        rows[n] = {"winner": [list(e) for e in t], "middle_used": used,
                   "final_seconds": round(clock.seconds["finals"] - before, 1),
                   "faults_found": len(ev.t3.hits(t, case["faults"], case["width"])),
                   "disagreement": ev.aq.disagreement(case, edits, probe)}
    with clock.part("choose"):
        full = pf.choosers(case, networks, seed)["disagree"]
    cheap_names = [n for n in pf.ORDER if n in networks and (gate or n in ("pick68", "boundary"))]
    cheap = pf.choosers(case, {n: networks[n] for n in cheap_names}, seed * 1000 + CHEAP_SEED)["disagree"]
    row = {"seed": seed, "gate": gate, "candidates": rows, "full": full, "cheap": {"built": cheap_names, **cheap},
           "seconds": {k: round(clock.seconds[k], 1) for k in PARTS}, "probe_seconds": round(clock.probe, 1),
           "steps": {k: round(v, 1) for k, v in clock.steps.items() if k != "probe"},
           "seconds_65": seconds_65(seed)}
    path.write_text(json.dumps(row) + "\n")
    return seed


def score(rows, before=None):
    """Core-hours by part, and the full and cheaper orders' lines on the probe. ``before``: #71 and #72's rows."""
    d = lambda r, n: r["candidates"][n]["disagreement"]
    hours = {k: sum(r["seconds"][k] for r in rows.values()) / 3600 for k in PARTS}
    hours["repair_65"] = sum(r["seconds_65"] for r in rows.values()) / 3600
    hours["probe_not_counted"] = sum(r["probe_seconds"] for r in rows.values()) / 3600
    # the cheaper order skips restarts everywhere it does not run, and the bigger committee runs where it did already
    skipped = sum(r["seconds"]["restarts"] + r["candidates"]["restarts"]["final_seconds"]
                  for r in rows.values() if not r["gate"]) / 3600
    steps = defaultdict(float)
    for r in rows.values():
        for k, v in r.get("steps", {}).items():
            steps[k] += v / 3600
    out = {"seeds": len(rows), "core_hours": hours, "step_core_hours": dict(sorted(steps.items(), key=lambda kv: -kv[1])),
           "gate": sorted(s for s, r in rows.items() if r["gate"]),
           "core_hours_full": sum(hours[k] for k in PARTS) + hours["repair_65"],
           "core_hours_cheap": sum(hours[k] for k in PARTS) + hours["repair_65"] - skipped}
    for name in ("full", "cheap"):
        p = {s: r[name]["pick"] for s, r in rows.items()}
        under = sorted(s for s in rows if d(rows[s], p[s]) < pf.LINE)
        out[name] = {"under_0.01%": len(under), "under_0.1%": sum(d(rows[s], p[s]) < 1e-3 for s in rows),
                     "exact": sum(rows[s]["candidates"][p[s]]["faults_found"] == 3 for s in rows),
                     "labels_asked": sum(r[name]["asked"] for r in rows.values()), "over_line": [s for s in rows if s not in under]}
    full = {s: d(r, r["full"]["pick"]) for s, r in rows.items()}
    cheap = {s: d(r, r["cheap"]["pick"]) for s, r in rows.items()}
    out["cheap_vs_full"] = {"gained": sorted(s for s in rows if full[s] >= pf.LINE > cheap[s]),
                            "lost": sorted(s for s in rows if cheap[s] >= pf.LINE > full[s]),
                            "different_pick": sorted(s for s, r in rows.items() if r["cheap"]["pick"] != r["full"]["pick"])}
    if before is not None:
        out["reproduces_disagree_pick"] = sum(r["full"]["pick"] == before[s]["choosers"]["disagree"]["pick"]
                                              for s, r in rows.items())
    return out


def previous():
    rows = json.loads((pf.OUT / "pick-fixes.json").read_text()) + json.loads(
        (EVIDENCE / "fresh-repair" / "fresh-repair.json").read_text())
    return {r["seed"]: r for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.score:
        seeds = [s for s in SEEDS if (a.score / f"case-{s}.json").exists()]  # the sample, or all 100
        rows = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in seeds}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "time-repair.json").write_text(json.dumps([rows[s] for s in seeds], indent=1) + "\n")
        result = score(rows, previous())
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    slow = {251, 258, 259, 261, 263, 265, 270, 279, 281, 296, 309, 312, 317, 328, 348}
    jobs = [(s, str(a.output)) for s in sorted(a.only or SEEDS, key=lambda s: s not in slow)]
    with Pool(a.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
