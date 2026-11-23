"""Rebuild the committee from each case's final labels (roadmap #66).

#65 kept, for seeds 250 to 299, every case's final labels: the 1,024 practice labels plus the 64 asked during the
repair, and the 8 region-split inputs of its extra round. The committee itself was screened from the 1,024 practice
labels alone. This script reruns the pair screen and the triple committee on all 1,096 labels, with the options of
the case's last stage (TWO_STEP), picks the best-fitting member the way the repair does (accuracy, then cost),
takes its middle (#64's walk), also scores the committee's own first member when it differs, and records whether
the true triple is now in the committee, the faults found and the disagreement on the usual 1M-point probe.

    python experiments/locate-repair/rebuild_committee.py --output runs/rebuild --workers 4
    python experiments/locate-repair/rebuild_committee.py --score runs/rebuild

Each seed is written to OUTPUT/case-SEED.json, so a restart skips finished seeds. ``--score`` merges them with #65's
rows into evidence/rebuild-committee/.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

import early_values as ev
from wide_compare import EVIDENCE, cases

SOURCE = EVIDENCE / "region-labels"
OUT = EVIDENCE / "rebuild-committee"
SEEDS = range(250, 300)


def labels(case, seed, source=SOURCE):
    """The case's 1,096 final labels: #65's saved inputs and labels plus its 8 region-split inputs, labelled."""
    z = np.load(source / "labels" / f"case-{seed}.npz")
    extra = z["added_region"]
    return (np.concatenate([z["x"], extra]),
            np.concatenate([z["y"], ev.tf.predict(case["teacher"], extra)]))


def rebuild(case, x, y, options, probe):
    """Screen pairs and build the triple committee on (x, y), pick the best member and take its middle."""
    size = {"size": options["committee_size"]} if "committee_size" in options else {}
    members, fits, count = ev.tc.committee(case, x, y, pairs_kept=options.get("pairs_kept"), **size)
    winner = max(members, key=lambda t: fits[t][0])
    (accuracy, _), edits = fits[winner]
    mid = ev.middle_step(case, {"x": x, "y": y, "winner": winner, "edits": edits}, probe)
    true = [t for t in members if len(ev.t3.hits(t, case["faults"], case["width"])) == 3]
    first = members[0]  # the committee's own order breaks accuracy ties by the coarse fit, not by cost
    if first == winner:
        head = {"same_as_winner": True}
    else:
        (head_acc, _), head_edits = fits[first]
        head = {"same_as_winner": False, "winner": [list(e) for e in first], "practice_accuracy": float(head_acc),
                "faults_found": len(ev.t3.hits(first, case["faults"], case["width"])),
                "disagreement": ev.aq.disagreement(case, head_edits, probe),
                "middle": ev.middle_step(case, {"x": x, "y": y, "winner": first, "edits": head_edits}, probe)}
    return {"first_member": head, "triples": count, "committee_size": len(members),
            "committee_has_true_triple": bool(true),
            "true_rank": members.index(true[0]) if true else None,
            "true_accuracy": float(fits[true[0]][0][0]) if true else None,
            "winner": [list(e) for e in winner], "practice_accuracy": float(accuracy),
            "faults_found": len(ev.t3.hits(winner, case["faults"], case["width"])),
            "disagreement": ev.aq.disagreement(case, edits, probe), "middle": mid}


def run_seed(args):
    seed, out, stages = args
    path = Path(out) / f"case-{seed}.json"
    if path.exists():
        return seed
    start = time.time()
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, y = labels(case, seed)
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    row = rebuild(case, x, y, ev.TWO_STEP[stages - 1], probe)
    row.update(seed=seed, labels=len(y), stage_options=ev.TWO_STEP[stages - 1], seconds=round(time.time() - start, 1))
    path.write_text(json.dumps(row) + "\n")
    return seed


def final(fit):
    """The middle when it fits as many labels as the fit (#64's rule), else the fit."""
    m = fit["middle"]
    return m["disagreement"] if m["accuracy"] >= fit["practice_accuracy"] else fit["disagreement"]


def first(row):
    """The committee's own first member, as a fit with its middle: the winner's row when they are the same."""
    head = row["first_member"]
    return row if head["same_as_winner"] else head


def check(before, after):
    """Compare #65's region arm (before) with the rebuilt committee (after), seed by seed."""
    b = {s: final(r["extra"]["region"]) for s, r in before.items()}
    a = {s: final(after[s]) for s in before}
    fb = {s: r["extra"]["region"]["faults_found"] for s, r in before.items()}
    fa = {s: after[s]["faults_found"] for s in before}
    h = {s: final(first(after[s])) for s in before}
    fh = {s: first(after[s])["faults_found"] for s in before}
    had = {s: r["committee_has_true_triple"] for s, r in before.items()}
    has = {s: after[s]["committee_has_true_triple"] for s in before}
    line = lambda d: {"under_0.1%": sum(v < 1e-3 for v in d.values()), "under_0.01%": sum(v < 1e-4 for v in d.values()),
                      "median": float(np.median(list(d.values()))), "worst": max(d.values())}
    return {"seeds": len(before), "cpu_hours": sum(after[s]["seconds"] for s in before) / 3600,
            "before": {**line(b), "exact": sum(v == 3 for v in fb.values()), "true_in_committee": sum(had.values())},
            "after": {**line(a), "exact": sum(v == 3 for v in fa.values()), "true_in_committee": sum(has.values())},
            "first_member": {**line(h), "exact": sum(v == 3 for v in fh.values()),
                             "differs_from_winner": sorted(s for s in before
                                                           if not after[s]["first_member"]["same_as_winner"]),
                             "closer_than_winner": sorted(s for s in before if h[s] < a[s]),
                             "further_than_winner": sorted(s for s in before if h[s] > a[s])},
            "true_entered": sorted(s for s in before if has[s] and not had[s]),
            "true_left": sorted(s for s in before if had[s] and not has[s]),
            "became_exact": sorted(s for s in before if fa[s] == 3 != fb[s]),
            "stopped_exact": sorted(s for s in before if fb[s] == 3 != fa[s]),
            "closer": sorted(s for s in before if a[s] < b[s]), "further": sorted(s for s in before if a[s] > b[s]),
            "crossed_under_0.01%": sorted(s for s in before if b[s] >= 1e-4 > a[s]),
            "crossed_over_0.01%": sorted(s for s in before if b[s] < 1e-4 <= a[s]),
            "crossed_under_0.1%": sorted(s for s in before if b[s] >= 1e-3 > a[s]),
            "crossed_over_0.1%": sorted(s for s in before if b[s] < 1e-3 <= a[s]),
            "per_case": {str(s): {"before": b[s], "after": a[s], "faults_before": fb[s], "faults_after": fa[s],
                                  "true_before": had[s], "true_after": has[s], "true_rank": after[s]["true_rank"],
                                  "first_member": h[s], "faults_first_member": fh[s]}
                         for s in sorted(before)}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    before = cases(SOURCE / "region-labels.json")
    if a.score:
        after = {s: json.loads((a.score / f"case-{s}.json").read_text()) for s in before}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "rebuild-committee.json").write_text(json.dumps([after[s] for s in sorted(after)], indent=1) + "\n")
        result = check(before, after)
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
