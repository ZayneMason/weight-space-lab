"""Choose among the committee members that fit every label (roadmap #67).

#66 rebuilt each case's committee on its 1,096 final labels (seeds 250 to 299). The true triple is then in 46 of 50
committees, but the repair's pick (most labels, then the smallest total edit) is exact in only 30: in 12 cases the
true triple ties the winner on every label, and in 4 its fit reaches fewer labels. This script rebuilds the same
committees, keeps every member's fit, and picks a winner six ways from the same members:

- ``cost``: most labels, then the smallest total scaled edit (the repair's rule, #66's winner).
- ``coarse``: most labels, then the committee's own order, the coarse 7-value fit (#66's first member).
- ``largest-edit``: most labels, then the smallest largest single scaled edit (#57, #58).
- the same three after refitting every member within 1% of the best accuracy from 5 more starts, each start the
  member's fitted values plus a standard normal step per weight in its layer's scale, keeping the best fit.

Each winner's final value is its middle (#64's walk) when the middle fits as many labels, else the fit. Also
records how many members tie at the top before and after the refit, and, as checks on the fit itself, the true
triple's accuracy at the original network's values (information the repair does not have) and its margin fit from
the broken network's values.

    python experiments/locate-repair/tie_break_final.py --output runs/tie --workers 4
    python experiments/locate-repair/tie_break_final.py --score runs/tie
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import early_values as ev
from rebuild_committee import labels, SEEDS
from wide_compare import EVIDENCE, cases

SOURCE = EVIDENCE / "region-labels"
OUT = EVIDENCE / "tie-break-final"
STARTS, NEAR = 5, 0.01
RULES = ("cost", "coarse", "largest-edit")
wt, ct = ev.tc.ct.wt, ev.tc.ct


def pick(members, fits, rule, case):
    """The winner among ``members`` (in committee order) under ``rule``; most labels always comes first."""
    top = max(fits[t][0][0] for t in members)
    tied = [t for t in members if fits[t][0][0] == top]
    if rule == "cost":
        return max(tied, key=lambda t: fits[t][0][1])
    if rule == "coarse":
        return tied[0]
    if rule == "largest-edit":
        return min(tied, key=lambda t: ev.tc.largest_edit(case, fits[t]))
    raise ValueError(f"unknown rule {rule!r}")


def restart(case, x, y, triple, fit, rng, starts=STARTS):
    """The best of ``fit`` and margin fits from ``starts`` random steps around its values."""
    scale = np.array([case["scale"][n] for n, _ in triple])
    values = np.array([v for _, _, v in fit[1]])
    best = fit
    for _ in range(starts):
        step = values + rng.standard_normal(3) * scale
        best = ct.best_of(best, wt.refine(case, x, y, triple, wt.score(case, x, y, triple, step)))
    return best


def score_fit(case, x, y, triple, fit, probe):
    (accuracy, cost), edits = fit
    return {"winner": [list(e) for e in triple], "practice_accuracy": float(accuracy), "cost": -float(cost),
            "faults_found": len(ev.t3.hits(triple, case["faults"], case["width"])),
            "disagreement": ev.aq.disagreement(case, edits, probe),
            "middle": ev.middle_step(case, {"x": x, "y": y, "winner": triple, "edits": edits}, probe)}


def run_case(case, x, y, options, probe, seed):
    size = {"size": options["committee_size"]} if "committee_size" in options else {}
    members, fits, count = ev.tc.committee(case, x, y, pairs_kept=options.get("pairs_kept"), **size)
    top = max(fits[t][0][0] for t in members)
    near = [t for t in members if fits[t][0][0] >= top - NEAR]
    rng = np.random.default_rng([seed, 67])
    refit = dict(fits)
    for t in near:
        refit[t] = restart(case, x, y, t, fits[t], rng)
    is_true = lambda t: len(ev.t3.hits(t, case["faults"], case["width"])) == 3
    true = [t for t in members if is_true(t)]
    told = tuple(tuple(f[:2]) for f in case["faults"])
    original = wt.refine(case, x, y, told, wt.score(case, x, y, told, ct.current_values(case, told)))
    from_clean = wt.score(case, x, y, told, [float(case["teacher"][n].flat[i]) for n, i in told])[0][0]
    scored, rules = {}, {}
    for label, table in (("", fits), ("refit-", refit)):
        for rule in RULES:
            w = pick(members, table, rule, case)
            key = (w, label)
            if key not in scored:
                scored[key] = score_fit(case, x, y, w, table[w], probe)
            rules[label + rule] = scored[key]
    tied = lambda table: sum(table[t][0][0] == max(table[u][0][0] for u in members) for t in members)
    return {"seed": seed, "labels": len(y), "triples": count, "committee_size": len(members), "near": len(near),
            "tied_top": tied(fits), "tied_top_refit": tied(refit),
            "true_tied_top": bool(true) and fits[true[0]][0][0] == top,
            "true_tied_top_refit": bool(true) and refit[true[0]][0][0] == max(refit[t][0][0] for t in members),
            "committee_has_true_triple": bool(true), "true_rank": members.index(true[0]) if true else None,
            "true_accuracy": float(fits[true[0]][0][0]) if true else None,
            "true_accuracy_refit": float(refit[true[0]][0][0]) if true else None,
            "true_from_current": float(original[0][0]), "true_from_clean": from_clean,
            "rules": rules}


def run_seed(args):
    seed, out, stages = args
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.time()
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, y = labels(case, seed)
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    row = run_case(case, x, y, ev.TWO_STEP[stages - 1], probe, seed)
    row.update(stage_options=ev.TWO_STEP[stages - 1], seconds=round(time.time() - start, 1))
    path.write_text(json.dumps(row) + "\n")
    return seed


def final(fit):
    """The middle when it fits as many labels as the fit (#64's rule), else the fit."""
    m = fit["middle"]
    return m["disagreement"] if m["accuracy"] >= fit["practice_accuracy"] else fit["disagreement"]


def check(rows):
    """Lines, exact repairs and crossings for every rule, each against ``cost`` (the repair's rule)."""
    names = list(next(iter(rows.values()))["rules"])
    d = {k: {s: final(r["rules"][k]) for s, r in rows.items()} for k in names}
    f = {k: {s: r["rules"][k]["faults_found"] for s, r in rows.items()} for k in names}
    base = d["cost"]
    out = {}
    for k in names:
        v = d[k]
        out[k] = {"under_0.1%": sum(q < 1e-3 for q in v.values()), "under_0.01%": sum(q < 1e-4 for q in v.values()),
                  "median": float(np.median(list(v.values()))), "worst": max(v.values()),
                  "exact": sum(q == 3 for q in f[k].values()),
                  "closer": sorted(s for s in v if v[s] < base[s]), "further": sorted(s for s in v if v[s] > base[s]),
                  "crossed_under_0.01%": sorted(s for s in v if base[s] >= 1e-4 > v[s]),
                  "crossed_over_0.01%": sorted(s for s in v if base[s] < 1e-4 <= v[s])}
    has = [s for s, r in rows.items() if r["committee_has_true_triple"]]
    return {"seeds": len(rows), "cpu_hours": sum(r["seconds"] for r in rows.values()) / 3600,
            "true_in_committee": len(has),
            "true_tied_top": sum(rows[s]["true_tied_top"] for s in has),
            "true_tied_top_refit": sum(rows[s]["true_tied_top_refit"] for s in has),
            "tied_top_median": float(np.median([r["tied_top"] for r in rows.values()])),
            "tied_top_refit_median": float(np.median([r["tied_top_refit"] for r in rows.values()])),
            "true_short": sorted(s for s in has if not rows[s]["true_tied_top"]),
            "true_short_after_refit": sorted(s for s in has if not rows[s]["true_tied_top_refit"]),
            "true_fits_every_label_from_clean": sorted(s for s, r in rows.items() if r["true_from_clean"] == 1.0),
            "rules": out,
            "per_case": {str(s): {k: [d[k][s], f[k][s]] for k in names} for s in sorted(rows)}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    before = cases(SOURCE / "region-labels.json")
    if a.score:
        rows = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in before}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "tie-break-final.json").write_text(json.dumps([rows[s] for s in sorted(rows)], indent=1) + "\n")
        result = check(rows)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=1))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    seeds = a.only or list(SEEDS)
    jobs = [(s, str(a.output), before[s]["stages_run"]) for s in seeds]
    with Pool(a.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
