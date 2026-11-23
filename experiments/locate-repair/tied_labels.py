"""Ask labels where the tied members disagree (roadmap #68).

#67 rebuilt each committee on its 1,096 final labels (seeds 250 to 299), refit the near-best members from 5 more
starts, and picked the winner among the members that fit the most labels by the committee's own order
(``refit-coarse``). The median case had 4 members tied at the top, and labels cannot separate them. This script
reproduces #67's committee and refit, then, while two or more members tie at the top, runs up to ROUNDS rounds of
8 labels: draw a fresh uniform pool, label the 8 inputs where the tied members' networks (each at its fitted
values) split most evenly, refit every committee member on all labels and pick again by the committee's order.
After every round it scores the winner and its middle (#64's walk) on the usual 1M-point probe.

    python experiments/locate-repair/tied_labels.py --output runs/tied --workers 4
    python experiments/locate-repair/tied_labels.py --score runs/tied
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import early_values as ev
from rebuild_committee import labels, SEEDS
from tie_break_final import NEAR, pick, restart, score_fit, final
from wide_compare import EVIDENCE, cases

SOURCE = EVIDENCE / "region-labels"
BEFORE = EVIDENCE / "tie-break-final"
OUT = EVIDENCE / "tied-labels"
ROUNDS = 3


def tied_top(members, fits):
    """The members (in committee order) that fit the most labels."""
    top = max(fits[t][0][0] for t in members)
    return [t for t in members if fits[t][0][0] == top]


def tied_split(case, fits, tied, pool):
    """Per pool input, the minority share among the tied members' networks."""
    votes = np.stack([ev.tf.predict(ev.tf.apply(case["arrays"], fits[t][1]), pool) for t in tied])
    return ev.aq.committee_split(votes)


def run_case(case, x, y, options, probe, seed, rounds=ROUNDS):
    size = {"size": options["committee_size"]} if "committee_size" in options else {}
    members, fits, count = ev.tc.committee(case, x, y, pairs_kept=options.get("pairs_kept"), **size)
    top = max(fits[t][0][0] for t in members)
    near = [t for t in members if fits[t][0][0] >= top - NEAR]
    rng = np.random.default_rng([seed, 67])  # the same refit as #67
    refit = dict(fits)
    for t in near:
        refit[t] = restart(case, x, y, t, fits[t], rng)
    is_true = lambda t: len(ev.t3.hits(t, case["faults"], case["width"])) == 3
    start = pick(members, refit, "coarse", case)
    steps = [{"round": 0, "labels": len(y), "tied": len(tied_top(members, refit)),
              **score_fit(case, x, y, start, refit[start], probe)}]
    rng = np.random.default_rng([seed, 68])
    table = refit
    for k in range(1, rounds + 1):
        tied = tied_top(members, table)
        if len(tied) < 2:
            break
        pool, ties = rng.uniform(-2, 2, size=(ev.aq.POOL, 2)), rng.random(ev.aq.POOL)
        split = tied_split(case, table, tied, pool)
        picked = np.lexsort((ties, -split))[:ev.aq.PER_ROUND]
        x = np.concatenate([x, pool[picked]])
        y = np.concatenate([y, ev.tf.predict(case["teacher"], pool[picked])])
        table = {t: ev.tc.refit(case, x, y, t, table[t]) for t in members}
        w = pick(members, table, "coarse", case)
        after = tied_top(members, table)
        steps.append({"round": k, "labels": len(y), "splitting": int((split > 0).sum()), "tied_before": len(tied),
                      "tied": len(after), "true_tied_top": any(is_true(t) for t in after),
                      **score_fit(case, x, y, w, table[w], probe)})
    true = [t for t in members if is_true(t)]
    return {"seed": seed, "triples": count, "committee_size": len(members), "committee_has_true_triple": bool(true),
            "true_tied_top_refit": bool(true) and refit[true[0]][0][0] == top_of(members, refit),
            "rounds": steps}


def top_of(members, fits):
    return max(fits[t][0][0] for t in members)


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


def check(rows, before):
    """Each case's last round against #67's refit-coarse pick (``before``), with the lines after every round."""
    last = {s: r["rounds"][-1] for s, r in rows.items()}
    b = {s: final(before[s]["rules"]["refit-coarse"]) for s in rows}
    a = {s: final(last[s]) for s in rows}
    fb = {s: before[s]["rules"]["refit-coarse"]["faults_found"] for s in rows}
    fa = {s: last[s]["faults_found"] for s in rows}
    by_round = {}
    for k in range(ROUNDS + 1):
        at = {s: r["rounds"][min(k, len(r["rounds"]) - 1)] for s, r in rows.items()}
        d = {s: final(v) for s, v in at.items()}
        by_round[str(k)] = {"under_0.1%": sum(v < 1e-3 for v in d.values()),
                            "under_0.01%": sum(v < 1e-4 for v in d.values()),
                            "median": float(np.median(list(d.values()))),
                            "exact": sum(v["faults_found"] == 3 for v in at.values()),
                            "still_tied": sum(v["tied"] >= 2 for v in at.values())}
    ratio = [a[s] / b[s] for s in rows if b[s] > 0 and a[s] > 0]
    return {"seeds": len(rows), "cpu_hours": sum(r["seconds"] for r in rows.values()) / 3600,
            "reproduces_67": sorted(s for s in rows
                                    if rows[s]["rounds"][0]["winner"] == before[s]["rules"]["refit-coarse"]["winner"]),
            "asked": sorted(s for s, r in rows.items() if len(r["rounds"]) > 1),
            "rounds_run": {str(k): sum(len(r["rounds"]) - 1 == k for r in rows.values()) for k in range(ROUNDS + 1)},
            "by_round": by_round,
            "geometric_mean_change": float(np.exp(np.mean(np.log(ratio)))) if ratio else None,
            "became_exact": sorted(s for s in rows if fa[s] == 3 != fb[s]),
            "stopped_exact": sorted(s for s in rows if fb[s] == 3 != fa[s]),
            "closer": sorted(s for s in rows if a[s] < b[s]), "further": sorted(s for s in rows if a[s] > b[s]),
            "crossed_under_0.01%": sorted(s for s in rows if b[s] >= 1e-4 > a[s]),
            "crossed_over_0.01%": sorted(s for s in rows if b[s] < 1e-4 <= a[s]),
            "per_case": {str(s): {"before": b[s], "after": a[s], "faults_before": fb[s], "faults_after": fa[s],
                                  "rounds": len(rows[s]["rounds"]) - 1, "tied": [v["tied"] for v in rows[s]["rounds"]]}
                         for s in sorted(rows)}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    seeds = cases(SOURCE / "region-labels.json")
    if a.score:
        rows = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in seeds}
        before = {r["seed"]: r for r in json.loads((BEFORE / "tie-break-final.json").read_text())}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "tied-labels.json").write_text(json.dumps([rows[s] for s in sorted(rows)], indent=1) + "\n")
        result = check(rows, before)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=1))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(a.output), seeds[s]["stages_run"]) for s in (a.only or list(SEEDS))]
    with Pool(a.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
