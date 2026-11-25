"""Run the whole cheaper repair on fresh seeds, next to the original (roadmap #77).

#76 checked the columns refine on the whole repair and the batched refine on #71's stage, but only on 8 cases these
refines had already been tried on. This script runs the whole repair, both stages, on fresh seeds (350 to 361), two
ways:

- ``original``: the repair as #72 ran it, with ``warm_triples.refine`` as it is;
- ``cheap``: columns for every refine (``cut_refine.install("columns")``) and the committee's continuous fits batched
  (``batched_refine.install()``), in #65's stage as well as #71's, since both call ``triple_committee.committee``.

Each case runs #65's stage (``early_values.two_step`` with the middle step and the extra round, as #72), saves its
labels, then #71's stage from them (#73's timed runner, ``time_repair.run_seed``, with its labels read from this
run's folder). Both runs are timed by #73's clock, and the score compares cost, the disagree chooser's pick, the
chosen network's disagreement on a million probe inputs, and how many cases are under 0.01% and exact.

    python experiments/locate-repair/fresh_cheap.py --kind original --output runs/fc-original --workers 2
    python experiments/locate-repair/fresh_cheap.py --kind cheap --output runs/fc-cheap --workers 2
    python experiments/locate-repair/fresh_cheap.py --score runs/fc-original runs/fc-cheap
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import batched_refine as br
import cut_refine as cr
import early_values as ev
from end_to_end import chosen, labels_from
from fresh_repair import repair_65
from pick_fixes import LINE
from wide_compare import EVIDENCE

OUT = EVIDENCE / "fresh-cheap"
KINDS = ("original", "cheap")
SEEDS = tuple(range(350, 362))


def stage_65(seed, out, kind):
    """#65's stage for one case, timed, with its outcome and the path its last stage's labels went to."""
    import time_repair
    clock = time_repair.CLOCK
    path = out / f"case-{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    clock.reset()
    cr.CALLS.update(calls=0, evaluations=0)
    br.CALLS.update(calls=0, evaluations=0)
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    start = time.perf_counter()
    row = ev.two_step(case, 2, saved=lambda k: out / f"case-{seed}.stage{k}.json", middle=True, extra=True)
    wall = time.perf_counter() - start
    d, found = repair_65(row)
    result = {"seed": seed, "kind": kind, "seconds": round(wall - clock.probe, 1), "probe_seconds": round(clock.probe, 1),
              "steps": {k: round(v, 1) for k, v in clock.steps.items() if k != "probe"},
              "refine_calls": cr.CALLS["calls"] + br.CALLS["calls"], "batched_calls": br.CALLS["calls"],
              "outcome": {"disagreement": d, "faults_found": found, "stages_run": row["stages_run"]}}
    path.write_text(json.dumps(result) + "\n")
    return result


def run_seed(args):
    seed, out, kind = args
    import time_repair
    out = Path(out)
    second = out / "second"
    if (second / f"case-{seed}.json").exists():
        return seed
    first = stage_65(seed, out, kind)
    cr.CALLS.update(calls=0, evaluations=0)
    br.CALLS.update(calls=0, evaluations=0)
    time_repair.run_seed((seed, str(second)))
    path = second / f"case-{seed}.json"
    row = json.loads(path.read_text())
    row.update(kind=kind, seconds_65=first["seconds"], refine_calls=cr.CALLS["calls"] + br.CALLS["calls"],
               batched_calls=br.CALLS["calls"])
    path.write_text(json.dumps(row) + "\n")
    return seed


def stage(row):
    return sum(row["seconds"].values())


def score(runs):
    """Both runs on the same seeds: cost by stage, the line, exact repairs, and whether the picks agree."""
    out = {}
    for kind, (first, second) in runs.items():
        seeds = sorted(second)
        refine = sum(first[s]["steps"].get("margin refine", 0) + second[s]["steps"].get("margin refine", 0)
                     for s in seeds) / 3600
        total = sum(first[s]["seconds"] + stage(second[s]) for s in seeds) / 3600
        out[kind] = {"seeds": seeds, "core_hours": total,
                     "core_hours_stage_65": sum(first[s]["seconds"] for s in seeds) / 3600,
                     "core_hours_stage_71": sum(stage(second[s]) for s in seeds) / 3600,
                     "refine_share": refine / total,
                     "under_0.01%": sum(chosen(second[s])[0] < LINE for s in seeds),
                     "under_0.1%": sum(chosen(second[s])[0] < 10 * LINE for s in seeds),
                     "exact": sum(chosen(second[s])[1] == 3 for s in seeds),
                     "repair_65_under_0.01%": sum(first[s]["outcome"]["disagreement"] < LINE for s in seeds),
                     "stages_run": {str(s): first[s]["outcome"]["stages_run"] for s in seeds},
                     "chosen": {str(s): chosen(second[s]) for s in seeds},
                     "pick": {str(s): second[s]["full"]["pick"] for s in seeds},
                     "seconds": {str(s): round(first[s]["seconds"] + stage(second[s])) for s in seeds}}
    if set(out) == set(KINDS):
        o, c = out["original"], out["cheap"]
        both = [s for s in o["seeds"] if s in c["seeds"]]
        ratio = sorted(o["seconds"][str(s)] / c["seconds"][str(s)] for s in both)
        c["vs_original"] = {
            "seeds": len(both),
            "speedup": sum(o["seconds"][str(s)] for s in both) / sum(c["seconds"][str(s)] for s in both),
            "speedup_per_case": [round(ratio[0], 2), round(ratio[len(ratio) // 2], 2), round(ratio[-1], 2)],
            "same_stage_65_outcome": sum(runs["original"][0][s]["outcome"] == runs["cheap"][0][s]["outcome"] for s in both),
            "same_pick": sum(o["pick"][str(s)] == c["pick"][str(s)] for s in both),
            "same_disagreement": sum(o["chosen"][str(s)][0] == c["chosen"][str(s)][0] for s in both),
            "crossed": {"gained": [s for s in both if o["chosen"][str(s)][0] >= LINE > c["chosen"][str(s)][0]],
                        "lost": [s for s in both if c["chosen"][str(s)][0] >= LINE > o["chosen"][str(s)][0]]},
            "changed": {str(s): [o["chosen"][str(s)], c["chosen"][str(s)]] for s in both
                        if o["chosen"][str(s)] != c["chosen"][str(s)]}}
    return out


def load(folder):
    read = lambda f: json.loads(f.read_text())
    first = {read(f)["seed"]: read(f) for f in folder.glob("case-*.json") if ".stage" not in f.name}
    second = {read(f)["seed"]: read(f) for f in (folder / "second").glob("case-*.json")}
    return next(iter(second.values()))["kind"], first, second


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path, nargs="*")
    a = p.parse_args()
    if a.score:
        OUT.mkdir(parents=True, exist_ok=True)
        runs = {}
        for folder in a.score:
            kind, first, second = load(folder)
            runs[kind] = (first, second)
            rows = [{"seed": s, "stage_65": first[s], "stage_71": second[s]} for s in sorted(second)]
            (OUT / f"{kind}.json").write_text(json.dumps(rows, indent=1) + "\n")
        result = score(runs)
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    cr.install("original" if a.kind == "original" else "columns")  # before time_repair wraps the refine in its clock
    import time_repair
    if a.kind == "cheap":
        br.batch_refine = time_repair.CLOCK.tap(br.batch_refine, "margin refine")
        br.install()
    # #71's stage reads its labels, stage count and #65's seconds from this run's own first stage
    first = lambda seed: json.loads((a.output / f"case-{seed}.json").read_text())
    time_repair.labels = labels_from(a.output)
    time_repair.stages_of = lambda seed: first(seed)["outcome"]["stages_run"]
    time_repair.seconds_65 = lambda seed: first(seed)["seconds"]
    (a.output / "second").mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:  # forked after the swaps above, so every worker runs with them
        for seed in pool.imap_unordered(run_seed, [(s, str(a.output), a.kind) for s in (a.only or SEEDS)]):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
