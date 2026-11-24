"""Fix the six cases still over 0.01% (roadmap #70).

#68 left 6 of its 50 width-16 three-fault cases (seeds 250 to 299) over 0.01% on the usual 1M-point probe: in 258,
261 and 296 the committee lacks the true triple, in 259 the true triple fits fewer labels than the winner, and 260 and
269 edit the right three weights with values that are off. #69 showed that ties cost nothing, so this script tries
three fixes aimed at those causes, each starting from #68's final state (rerun exactly, same committees, refits,
pools and labels):

- ``bigger`` (258, 261, 296): rebuild the committee on #68's final labels with twice the committee size and twice the
  pairs kept, repeat #67's refit of the near-best members, and pick by most labels then committee order. Also records
  where the true triple's best pair ranks among all screened pairs, which decides whether it can enter at all.
- ``restarts`` (all six): refit #68's winner, and the true triple when it is in the committee, from 20 more random
  starts each (#67's steps in each layer's scale), then pick between the two by most labels, then committee order.
- ``boundary`` (all six): keep #68's winning triple, and for 4 rounds label the 16 inputs of a fresh uniform pool of
  20,000 on [-2, 2]^2 where the fitted network's logit margin is smallest, refitting only the triple's values after
  each round (64 labels in all).

As a check that the repair does not have, it also refits the true triple from 20 starts on #68's final labels, from
the broken network's values. Each final value is the middle (#64's walk) when it fits as many labels as the fit.

    python experiments/locate-repair/fix_six.py --output runs/six --workers 4
    python experiments/locate-repair/fix_six.py --score runs/six
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import early_values as ev
from rebuild_committee import labels
from tie_break_final import NEAR, final, pick, restart, score_fit
from tied_labels import OUT as TIED, run_case
from wide_compare import EVIDENCE, cases

SOURCE = EVIDENCE / "region-labels"
OUT = EVIDENCE / "fix-six"
SEEDS = (258, 259, 260, 261, 269, 296)
MISSING = (258, 261, 296)
STARTS = 20
ROUNDS, PER_ROUND = 4, 16
tc, wt, ct = ev.tc, ev.tc.ct.wt, ev.tc.ct


def is_true(case, t):
    return len(ev.t3.hits(t, case["faults"], case["width"])) == 3


def margin(arrays, x):
    """Logit 1 minus logit 0: its sign is the label, its size how far the input is from the boundary."""
    a = {n: v.astype(np.float64) for n, v in arrays.items()}
    logits = np.maximum(x @ a["hidden_weight"].T + a["hidden_bias"], 0) @ a["weight"].T + a["bias"]
    return logits[:, 1] - logits[:, 0]


def true_pair_rank(case, x, y):
    """Where the best pair of two true faults ranks in the committee's own pair order (0 is first), and how many pairs."""
    pairs = tc.sc.ranked_pairs(case["arrays"], x, y)
    order = tc.js.joint_order(tc.js.joint_scores(case, x, y, pairs, points=tc.ds.PAIR_POINTS))
    ranked = [pairs[k] for k in order]
    hits = [r for r, p in enumerate(ranked) if len(ev.t3.hits(p, case["faults"], case["width"])) == 2]
    return (hits[0] if hits else None), len(ranked)


def bigger(case, x, y, options, probe, seed):
    size = 2 * options.get("committee_size", tc.COMMITTEE)
    kept = 2 * options.get("pairs_kept", ev.t3.PAIRS)
    members, fits, count = tc.committee(case, x, y, size=size, pairs_kept=kept)
    top = max(fits[t][0][0] for t in members)
    rng = np.random.default_rng([seed, 67])
    table = dict(fits)
    for t in [t for t in members if fits[t][0][0] >= top - NEAR]:
        table[t] = restart(case, x, y, t, fits[t], rng)
    w = pick(members, table, "coarse", case)
    true = [t for t in members if is_true(case, t)]
    rank, pairs = true_pair_rank(case, x, y)
    return {"committee_size": size, "pairs_kept": kept, "triples": count, "pairs": pairs, "true_pair_rank": rank,
            "has_true": bool(true), "true_rank": members.index(true[0]) if true else None,
            "true_accuracy": float(table[true[0]][0][0]) if true else None, **score_fit(case, x, y, w, table[w], probe)}


def restarts(case, x, y, members, table, winner, probe, seed):
    rng = np.random.default_rng([seed, 70, 1])
    true = [t for t in members if is_true(case, t)]
    pool = [winner] + [t for t in true[:1] if t != winner]
    fits = {t: restart(case, x, y, t, table[t], rng, starts=STARTS) for t in pool}
    w = pick([t for t in members if t in fits], fits, "coarse", case)
    return {"winner_accuracy_before": float(table[winner][0][0]), "winner_accuracy": float(fits[winner][0][0]),
            "true_accuracy_before": float(table[true[0]][0][0]) if true else None,
            "true_accuracy": float(fits[true[0]][0][0]) if true else None, **score_fit(case, x, y, w, fits[w], probe)}


def boundary(case, x, y, winner, fit, probe, seed):
    rng = np.random.default_rng([seed, 70, 2])
    steps = []
    for k in range(1, ROUNDS + 1):
        pool = rng.uniform(-2, 2, size=(ev.aq.POOL, 2))
        near = np.argsort(np.abs(margin(ev.tf.apply(case["arrays"], fit[1]), pool)), kind="stable")[:PER_ROUND]
        new_y = ev.tf.predict(case["teacher"], pool[near])
        wrong = int((ev.tf.predict(ev.tf.apply(case["arrays"], fit[1]), pool[near]) != new_y).sum())
        x, y = np.concatenate([x, pool[near]]), np.concatenate([y, new_y])
        fit = tc.refit(case, x, y, winner, fit)
        steps.append({"round": k, "labels": len(y), "wrong_before": wrong, "accuracy": float(fit[0][0]),
                      "disagreement": ev.aq.disagreement(case, fit[1], probe)})
    return {"rounds": steps, **score_fit(case, x, y, winner, fit, probe)}


def true_check(case, x, y, probe, seed):
    """The true triple fitted from the broken values and 20 starts around them: information the repair lacks."""
    told = tuple(tuple(f[:2]) for f in case["faults"])
    fit = wt.refine(case, x, y, told, wt.score(case, x, y, told, ct.current_values(case, told)))
    fit = restart(case, x, y, told, fit, np.random.default_rng([seed, 70, 3]), starts=STARTS)
    return {"accuracy": float(fit[0][0]), "disagreement": ev.aq.disagreement(case, fit[1], probe)}


def run_seed(args):
    seed, out, stages = args
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.time()
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, y = labels(case, seed)
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    options = ev.TWO_STEP[stages - 1]
    keep = {}
    row = run_case(case, x, y, options, probe, seed, keep=keep)
    members, table, x, y = keep["members"], keep["table"], keep["x"], keep["y"]
    w = pick(members, table, "coarse", case)
    arms = {"restarts": restarts(case, x, y, members, table, w, probe, seed),
            "boundary": boundary(case, x, y, w, table[w], probe, seed)}
    if seed in MISSING:
        arms["bigger"] = bigger(case, x, y, options, probe, seed)
    row.update(final_labels=len(y), winner=[list(e) for e in w], arms=arms, true_check=true_check(case, x, y, probe, seed),
               stage_options=options, seconds=round(time.time() - start, 1))
    path.write_text(json.dumps(row) + "\n")
    return seed


def check(rows, before):
    """Each arm's final value and faults found against #68's last round, case by case."""
    out, arms = {}, sorted({a for r in rows.values() for a in r["arms"]})
    base = {s: final(before[s]["rounds"][-1]) for s in rows}
    for a in arms:
        have = {s: r["arms"][a] for s, r in rows.items() if a in r["arms"]}
        d = {s: final(v) for s, v in have.items()}
        out[a] = {"cases": sorted(have), "under_0.01%": sorted(s for s in d if d[s] < 1e-4),
                  "under_0.1%": sorted(s for s in d if d[s] < 1e-3),
                  "exact": sorted(s for s, v in have.items() if v["faults_found"] == 3),
                  "closer": sorted(s for s in d if d[s] < base[s]), "further": sorted(s for s in d if d[s] > base[s]),
                  "per_case": {str(s): [base[s], d[s], have[s]["faults_found"]] for s in sorted(have)}}
    return {"seeds": len(rows), "cpu_hours": sum(r["seconds"] for r in rows.values()) / 3600,
            "reproduces_68": sorted(s for s, r in rows.items()
                                    if r["rounds"][-1]["winner"] == before[s]["rounds"][-1]["winner"]),
            "before": {str(s): base[s] for s in sorted(rows)}, "arms": out,
            "true_fits_every_label": sorted(s for s, r in rows.items() if r["true_check"]["accuracy"] == 1.0),
            "true_check": {str(s): r["true_check"] for s, r in sorted(rows.items())}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    stages = cases(SOURCE / "region-labels.json")
    before = {r["seed"]: r for r in json.loads((TIED / "tied-labels.json").read_text())}
    if a.score:
        rows = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in SEEDS}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fix-six.json").write_text(json.dumps([rows[s] for s in SEEDS], indent=1) + "\n")
        result = check(rows, before)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    # the expensive committee rebuilds first, so the pool stays busy
    jobs = [(s, str(a.output), stages[s]["stages_run"]) for s in sorted(a.only or SEEDS, key=lambda s: s not in MISSING)]
    with Pool(a.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
