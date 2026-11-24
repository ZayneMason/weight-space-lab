"""Choose among the fixes from labels alone (roadmap #71).

#70 found that three fixes each rescue a different one of the 6 cases #68 left over 0.01%, and none made any case
worse, but choosing the best fix per case used the 1M-point probe, which a real repair cannot see. This script runs
the fixes on all 50 of #67's width-16 three-fault cases (seeds 250 to 299) and chooses among them from labels only.

For each case it reruns #68 exactly (same committees, refits, pools and labels) and builds up to four candidates:

- ``pick68``: #68's final pick.
- ``restarts``: #70's arm, the winner and the true triple when it is in the committee refitted from 20 starts.
- ``boundary``: #70's arm, 4 rounds of 16 labels where the fitted network's margin is smallest, refitting the values.
- ``bigger``: #70's arm, a committee twice as large with twice the pairs kept. #70 ran it only where the committee
  lacked the true triple, which the repair cannot know; here it runs where #68's pick misses at least one of its own
  labels, a gate the repair can see (it catches all three of #70's committee cases).

Each candidate's network is its final one: the middle (#64's walk) when the middle fits as many labels as the fit,
else the fit, exactly as #70 scored them. Two choosers then see labels no candidate was fitted on:

- ``uniform``: 64 fresh inputs uniform on [-2, 2]^2, labelled; most correct wins.
- ``disagree``: a fresh uniform pool of 200,000 inputs; up to 32 of those where any two candidates' networks give
  different labels are labelled; most correct wins. With no such input there is nothing to choose.

Ties keep #68's pick when it is among the best, else the first of boundary, restarts, bigger. The probe scores only.

    python experiments/locate-repair/pick_fixes.py --output runs/pick --workers 4
    python experiments/locate-repair/pick_fixes.py --score runs/pick
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import early_values as ev
from fix_six import ROUNDS, PER_ROUND, STARTS, is_true, margin, true_pair_rank
from rebuild_committee import labels
from refine_values import feasible_centre
from tie_break_final import NEAR, pick, restart
from tied_labels import OUT as TIED, run_case
from wide_compare import EVIDENCE, cases

SOURCE = EVIDENCE / "region-labels"
OUT = EVIDENCE / "pick-fixes"
SEEDS = tuple(range(250, 300))
ORDER = ("pick68", "boundary", "restarts", "bigger")
UNIFORM, POOL, DUEL = 64, 200_000, 32
LINE = 1e-4
tc, wt, ct = ev.tc, ev.tc.ct.wt, ev.tc.ct


def final_edits(case, x, y, triple, fit):
    """The candidate's final network edits: the middle when it fits as many labels as the fit (#64, as #70 scored)."""
    (accuracy, _), edits = fit
    values = [v for _, _, v in edits]
    centre, _ = feasible_centre(case, x, y, triple, values, np.random.default_rng([case["width"], case["seed"], 64]),
                                steps=1600, burn=400)
    (after, _), middle = wt.score(case, x, y, triple, centre)
    return (middle, True) if after >= accuracy else (edits, False)


def candidate(case, x, y, triple, fit, probe):
    edits, used = final_edits(case, x, y, triple, fit)
    return {"winner": [list(e) for e in triple], "practice_accuracy": float(fit[0][0]), "middle_used": used,
            "faults_found": len(ev.t3.hits(triple, case["faults"], case["width"])),
            "disagreement": ev.aq.disagreement(case, edits, probe)}, edits


def restarts(case, x, y, members, table, winner, seed):
    rng = np.random.default_rng([seed, 70, 1])
    true = [t for t in members if is_true(case, t)]
    pool = [winner] + [t for t in true[:1] if t != winner]
    fits = {t: restart(case, x, y, t, table[t], rng, starts=STARTS) for t in pool}
    w = pick([t for t in members if t in fits], fits, "coarse", case)
    return w, fits[w]


def boundary(case, x, y, winner, fit, seed):
    rng = np.random.default_rng([seed, 70, 2])
    for _ in range(ROUNDS):
        pool = rng.uniform(-2, 2, size=(ev.aq.POOL, 2))
        near = np.argsort(np.abs(margin(ev.tf.apply(case["arrays"], fit[1]), pool)), kind="stable")[:PER_ROUND]
        x, y = np.concatenate([x, pool[near]]), np.concatenate([y, ev.tf.predict(case["teacher"], pool[near])])
        fit = tc.refit(case, x, y, winner, fit)
    return winner, fit, x, y


def bigger(case, x, y, options, seed):
    size = 2 * options.get("committee_size", tc.COMMITTEE)
    kept = 2 * options.get("pairs_kept", ev.t3.PAIRS)
    members, fits, _ = tc.committee(case, x, y, size=size, pairs_kept=kept)
    top = max(fits[t][0][0] for t in members)
    rng = np.random.default_rng([seed, 67])
    table = dict(fits)
    for t in [t for t in members if fits[t][0][0] >= top - NEAR]:
        table[t] = restart(case, x, y, t, fits[t], rng)
    w = pick(members, table, "coarse", case)
    return w, table[w], {"has_true": any(is_true(case, t) for t in members), "true_pair_rank": true_pair_rank(case, x, y)[0]}


def choose(correct):
    """Most correct wins; ties keep #68's pick when it is among the best, else the first in ORDER."""
    best = max(correct.values())
    return next(n for n in ORDER if correct.get(n) == best)


def choosers(case, networks, seed):
    """Labels no candidate was fitted on: 64 uniform ones, and up to 32 where the candidates disagree."""
    rng = np.random.default_rng([seed, 71])
    xu = rng.uniform(-2, 2, size=(UNIFORM, 2))
    yu = ev.tf.predict(case["teacher"], xu)
    pool = rng.uniform(-2, 2, size=(POOL, 2))
    votes = np.stack([ev.tf.predict(ev.tf.apply(case["arrays"], networks[n]), pool) for n in networks])
    split = np.flatnonzero((votes != votes[0]).any(axis=0))
    duel = rng.choice(split, size=min(DUEL, len(split)), replace=False) if len(split) else split
    yd = ev.tf.predict(case["teacher"], pool[duel])
    uniform = {n: int((ev.tf.predict(ev.tf.apply(case["arrays"], e), xu) == yu).sum()) for n, e in networks.items()}
    disagree = {n: int((votes[k][duel] == yd).sum()) for k, n in enumerate(networks)}
    return {"uniform": {"correct": uniform, "pick": choose(uniform)},
            "disagree": {"split_inputs": int(len(split)), "asked": int(len(duel)), "correct": disagree,
                         "pick": choose(disagree)}}


def run_seed(args):
    seed, out, stages, *rest = args  # an optional fourth item is the folder holding labels/case-SEED.npz (#72)
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.time()
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, y = labels(case, seed, Path(rest[0])) if rest else labels(case, seed)
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    options = ev.TWO_STEP[stages - 1]
    keep = {}
    run_case(case, x, y, options, probe, seed, keep=keep)
    members, table, x, y = keep["members"], keep["table"], keep["x"], keep["y"]
    w = pick(members, table, "coarse", case)
    built = {"pick68": (w, table[w], x, y)}
    t, f = restarts(case, x, y, members, table, w, seed)
    built["restarts"] = (t, f, x, y)
    built["boundary"] = boundary(case, x, y, w, table[w], seed)
    info = {}
    if table[w][0][0] < 1.0:
        t, f, info = bigger(case, x, y, options, seed)
        built["bigger"] = (t, f, x, y)
    rows, networks = {}, {}
    for n, (t, f, cx, cy) in built.items():
        rows[n], networks[n] = candidate(case, cx, cy, t, f, probe)
    row = {"seed": seed, "labels": len(y), "bigger_ran": "bigger" in built, "bigger": info,
           "candidates": rows, "choosers": choosers(case, networks, seed),
           "committee_has_true": any(is_true(case, t) for t in members), "stage_options": options,
           "seconds": round(time.time() - start, 1)}
    path.write_text(json.dumps(row) + "\n")
    return seed


def check(rows, before=None):
    """Lines and crossings for #68's pick, each arm where it ran, each chooser and the probe's own best pick."""
    def d(r, n):
        return r["candidates"][n]["disagreement"]

    picks = {"pick68": {s: "pick68" for s in rows},
             "uniform": {s: r["choosers"]["uniform"]["pick"] for s, r in rows.items()},
             "disagree": {s: r["choosers"]["disagree"]["pick"] for s, r in rows.items()},
             "probe_best": {s: min(r["candidates"], key=lambda n: (d(r, n), ORDER.index(n))) for s, r in rows.items()}}
    for n in ORDER[1:]:
        picks[n] = {s: (n if n in r["candidates"] else "pick68") for s, r in rows.items()}
    base = {s for s, r in rows.items() if d(r, "pick68") < LINE}
    out = {}
    for name, p in picks.items():
        under = sorted(s for s in rows if d(rows[s], p[s]) < LINE)
        out[name] = {"under_0.01%": len(under), "under_0.1%": sum(d(rows[s], p[s]) < 1e-3 for s in rows),
                     "exact": sum(rows[s]["candidates"][p[s]]["faults_found"] == 3 for s in rows),
                     "gained": sorted(set(under) - base), "lost": sorted(base - set(under)),
                     "switched": sorted(s for s in rows if p[s] != "pick68"),
                     "worse": sorted(s for s in rows if d(rows[s], p[s]) > d(rows[s], "pick68"))}
    result = {"seeds": len(rows), "cpu_hours": sum(r["seconds"] for r in rows.values()) / 3600,
              "bigger_ran": sorted(s for s, r in rows.items() if r["bigger_ran"]), "picks": out,
              "no_split": sorted(s for s, r in rows.items() if r["choosers"]["disagree"]["split_inputs"] == 0)}
    if before is not None:
        result["reproduces_68"] = sum(rows[s]["candidates"]["pick68"]["winner"] == before[s]["rounds"][-1]["winner"]
                                      for s in rows)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    stages = cases(SOURCE / "region-labels.json")
    if a.score:
        before = {r["seed"]: r for r in json.loads((TIED / "tied-labels.json").read_text())}
        rows = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in SEEDS}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "pick-fixes.json").write_text(json.dumps([rows[s] for s in SEEDS], indent=1) + "\n")
        result = check(rows, before)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    # slowest first (#68's long cases and the ones that will rebuild a bigger committee), so the pool stays busy
    slow = {251, 258, 259, 261, 263, 265, 270, 279, 281, 296}
    jobs = [(s, str(a.output), stages[s]["stages_run"]) for s in sorted(a.only or SEEDS, key=lambda s: s not in slow)]
    with Pool(a.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
