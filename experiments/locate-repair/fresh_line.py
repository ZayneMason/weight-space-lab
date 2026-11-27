"""Run the whole repair with the gated duel and the flat-line step on fresh seeds (roadmap #83).

#81's gated duel and #82's flat-line step were chosen on #79's and #80's 150 cases. This runs the whole cheap repair
on 50 seeds nothing has been run on (600 to 649), exactly as #80 ran it (``saved_65.run_seed``: #65's stage with its
network saved, then #71's stage and every chooser on the same pool), then adds the two new steps:

- the gated duel (#81): #71's pick, or #65's saved repair when their duel has at least 32 split inputs and #65's
  repair answers more of its labels right;
- the flat-line step (#82) on that pick, from #65's final labels: where the least-constrained direction is flat, the
  middle of the segment of perfect fits along it, then 32 labels along it.

    python experiments/locate-repair/fresh_line.py --output runs/l83 --part 0 --parts 4   # parts 0 to 3, one per core
    python experiments/locate-repair/fresh_line.py --score runs/l83
    python experiments/locate-repair/fresh_line.py --truth runs/l83   # the true broken weights, refitted, on the cases over the line
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import batched_refine as br
import cut_refine as cr
import early_values as ev
import pick_fixes as pf
import saved_65
from value_labels import K, M, ROUNDS, line_arm
from wide_compare import EVIDENCE

OUT = EVIDENCE / "fresh-line"
SEEDS = tuple(range(600, 650))
LINE = pf.LINE


def gated(row):
    """#81's rule on #80's saved duel."""
    first, d = row["without"]["pick"], row["two_round"]
    won = d["correct"]["repair65"] - d["correct"][first] >= M
    return "repair65" if d["split_inputs"] >= K and won else first


def final_labels(case, folder, seed, stages_run):
    z = np.load(folder / f"case-{seed}.stage{stages_run - 1}.npz")
    return (np.concatenate([z["x"], z["added_region"]]),
            np.concatenate([z["y"], ev.tf.predict(case["teacher"], z["added_region"])]))


def run_seed(seed, out):
    path = out / f"case-{seed}.line.json"
    if path.exists():
        return
    saved_65.run_seed(seed, out)
    row = json.loads((out / f"case-{seed}.json").read_text())
    start = time.perf_counter()
    case = ev.rs.resize(ev.t3.corrupt(16, seed))
    x, y = final_labels(case, out, seed, row["stages_run"])
    probe = ev.bs.make_probe(case, ev.bs.PROBE)
    p = gated(row)
    c = row["candidates"][p]
    triple = tuple(tuple(e) for e in c["winner"])
    line = line_arm(case, x, y, triple, [v for _, _, v in c["edits"]], probe, seed)
    path.write_text(json.dumps({"seed": seed, "gated_pick": p, "line": line,
                                "seconds": round(time.perf_counter() - start, 1)}) + "\n")


def rates(seeds, d, f):
    return {"under_0.01%": sum(d[s] < LINE for s in seeds), "under_0.1%": sum(d[s] < 10 * LINE for s in seeds),
            "exact": sum(f[s] == 3 for s in seeds), "median": float(np.median([d[s] for s in seeds])),
            "over_line": {str(s): d[s] for s in seeds if d[s] >= LINE}}


def moves(seeds, before, after):
    return {"better": sorted(s for s in seeds if after[s] < before[s]),
            "worse": sorted(s for s in seeds if after[s] > before[s]),
            "gained": sorted(s for s in seeds if after[s] < LINE <= before[s]),
            "lost": sorted(s for s in seeds if before[s] < LINE <= after[s])}


def score(folder):
    rows = {s: json.loads((folder / f"case-{s}.json").read_text()) for s in SEEDS}
    lines = {s: json.loads((folder / f"case-{s}.line.json").read_text()) for s in SEEDS}
    seeds = list(SEEDS)
    d = lambda s, n: rows[s]["candidates"][n]["disagreement"]
    f = lambda s, n: rows[s]["candidates"][n]["faults_found"]
    picks = {"repair65": {s: "repair65" for s in seeds}, "without": {s: rows[s]["without"]["pick"] for s in seeds},
             "gated": {s: lines[s]["gated_pick"] for s in seeds}}
    dis = {k: {s: d(s, p[s]) for s in seeds} for k, p in picks.items()}
    faults = {k: {s: f(s, p[s]) for s in seeds} for k, p in picks.items()}
    flat = [s for s in seeds if lines[s]["line"]["flat"]]
    dis["line_middle"] = {s: lines[s]["line"]["middle"]["disagreement"] if s in flat else dis["gated"][s] for s in seeds}
    dis["line_after_32"] = {s: lines[s]["line"]["rounds"][-1]["disagreement"] if s in flat else dis["gated"][s]
                            for s in seeds}
    faults["line_middle"] = faults["line_after_32"] = faults["gated"]  # the line step keeps the pick's weights
    out = {"seeds": seeds, **{k: rates(seeds, dis[k], faults[k]) for k in dis}}
    out["steps"] = {"stage_71_over_65": moves(seeds, dis["repair65"], dis["without"]),
                    "gated_over_71": moves(seeds, dis["without"], dis["gated"]),
                    "line_middle_over_gated": moves(seeds, dis["gated"], dis["line_middle"]),
                    "line_32_over_gated": moves(seeds, dis["gated"], dis["line_after_32"])}
    out["gated_took_65"] = sorted(s for s in seeds if picks["gated"][s] == "repair65")
    out["flat"] = flat
    out["per_case_flat"] = {str(s): [dis["gated"][s], dis["line_middle"][s], dis["line_after_32"][s]] for s in flat}
    out["labels_asked"] = {"without": sum(r["without"]["asked"] for r in rows.values()),
                           "gated_and_line": sum(r["without"]["asked"] for r in rows.values())
                           + sum(r["two_round"]["asked"] for r in rows.values() if r["two_round"]["split_inputs"] >= K)
                           + ROUNDS * 8 * len(flat)}
    out["core_hours"] = sum(r["seconds_65"] + r["seconds_71"] for r in rows.values()) / 3600
    out["core_hours_line"] = sum(v["seconds"] for v in lines.values()) / 3600
    out["stages_run"] = {str(k): sum(r["stages_run"] == k for r in rows.values()) for k in (1, 2, 3)}
    out["bigger_ran"] = sorted(s for s, r in rows.items() if r["gate"])
    return rows, lines, out


def truth(folder):
    """Fit the true broken weights, and the pick's weights, on #65's final labels, in each case the pick leaves over the line."""
    cr.install("columns")
    br.install()
    rows = {s: json.loads((folder / f"case-{s}.json").read_text()) for s in SEEDS}
    out = {}
    for s, r in rows.items():
        c = r["candidates"][r["without"]["pick"]]
        if c["disagreement"] < LINE:
            continue
        case = ev.rs.resize(ev.t3.corrupt(16, s))
        x, y = final_labels(case, folder, s, r["stages_run"])
        probe = ev.bs.make_probe(case, ev.bs.PROBE)
        out[str(s)] = {"faults": case["faults"], "pick_weights": c["winner"], "labels": len(y)}
        for name, t in (("true", case["faults"]), ("pick", c["winner"])):
            t = tuple(tuple(e) for e in t)
            f = pf.wt.fit(case, x, y, t)
            edits, _ = pf.final_edits(case, x, y, t, f)
            out[str(s)][name] = {"labels_right": float(f[0][0]), "disagreement": ev.aq.disagreement(case, edits, probe)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--score", type=Path)
    p.add_argument("--truth", type=Path)
    a = p.parse_args()
    if a.truth:
        result = truth(a.truth)
        (OUT / "truth.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    if a.score:
        rows, lines, result = score(a.score)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fresh-line.json").write_text(json.dumps([{**rows[s], "line_step": lines[s]} for s in SEEDS]) + "\n")
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("per_case_flat",)}, indent=1))
        return
    cr.install("columns")
    br.install()
    ev.SAVE_EDITS = True
    a.output.mkdir(parents=True, exist_ok=True)
    for seed in list(a.only or SEEDS)[a.part::a.parts]:
        start = time.time()
        run_seed(seed, a.output)
        print("done", seed, round(time.time() - start), time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
