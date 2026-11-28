"""Run the fixed repair at width 32 (roadmap #92).

Every result since #61 is at H = 16; #60 ran an older repair at H = 32 on seeds 200 to 211 (12 of 12 under 0.1%, 8
under 0.01%, 5 exact). This runs #90's order, fixed in advance, on those seeds at H = 32:

1. #83's first part (#65's stage with its network saved, #71's stage and #81's gated duel), as ``saved_65.run_seed``;
2. #84's keep-one search where the pick misses one of #65's final labels, #89's keep-two search elsewhere, both with
   #85's fitter, then #88's refit duel, as ``fresh_end.repair``;
3. #82's flat-line step on the duel's winner. The one change from #90, taken from #91: the result is the network after
   the line step on any flat winner (#89's order), since on 331 the gate left a refit nearly identical to the pick
   without it.

Every stage builds its case as ``t3.corrupt(16, seed)``; this script swaps ``t3.corrupt`` for one that builds the
same seed at H = 32, so each stage runs unchanged.

    python experiments/locate-repair/width_32.py --output runs/w92 --only 200                 # the pilot
    python experiments/locate-repair/width_32.py --output runs/w92 --part 0 --parts 4         # first part
    python experiments/locate-repair/width_32.py --repair runs/w92 --workers 4                # search, duel, line step
    python experiments/locate-repair/width_32.py --score runs/w92
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import shutil
import time

import early_values as ev
import fresh_end as fe
import keep_one as k
import perfect_fit as pfit
from wide_compare import EVIDENCE

WIDTH = 32
OUT = EVIDENCE / "width-32-repair"
SEEDS = tuple(range(200, 212))
LINE = k.LINE
_corrupt = ev.t3.corrupt
ev.t3.corrupt = lambda width, seed, *a, **kw: _corrupt(WIDTH, seed, *a, **kw)
k.SOURCES[200] = OUT


def prepare(folder, seeds):
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        last = json.loads((folder / f"case-{seed}.json").read_text())["stages_run"] - 1
        shutil.copy(folder / f"case-{seed}.stage{last}.npz", OUT / "labels" / f"case-{seed}.npz")


def score(folder):
    seeds = [s for s in SEEDS if (folder / f"case-{s}.repair.json").exists()]
    first = {s: json.loads((folder / f"case-{s}.json").read_text()) for s in seeds}
    rows = {s: json.loads((folder / f"case-{s}.repair.json").read_text()) for s in seeds}
    arms = {
        "repair65": {s: (first[s]["candidates"]["repair65"]["disagreement"],
                         first[s]["candidates"]["repair65"]["faults_found"]) for s in seeds},
        "pick": {s: (rows[s]["pick"]["disagreement"], rows[s]["pick"]["faults_found"]) for s in seeds},
        "duel": {s: (rows[s]["chosen"]["disagreement"], rows[s]["chosen"]["faults_found"]) for s in seeds},
        "gated_line": {s: tuple(rows[s]["result"]) for s in seeds},  # #90's order
        "result": {s: tuple(rows[s]["every_winner"]) for s in seeds}}
    return rows, {
        "seeds": seeds, "width": WIDTH, **{name: fe.tally(d) for name, d in arms.items()},
        "result_vs_pick": fe.moves(seeds, arms["pick"], arms["result"]),
        "result_vs_gated_line": fe.moves(seeds, arms["gated_line"], arms["result"]),
        "keep_one": sorted(s for s in seeds if rows[s]["search"] == "keep-one"),
        "with_candidates": sorted(s for s in seeds if rows[s]["perfect"]),
        "true_perfect": sorted(s for s in seeds if rows[s]["true_perfect"]),
        "took_other": sorted(s for s in seeds if rows[s]["duel"] and rows[s]["duel"]["winner"] != "pick"),
        "flat": sorted(s for s in seeds if rows[s]["line"]["flat"]),
        "triples_fitted": {"keep-one": sum(r["searched"] for r in rows.values() if r["search"] == "keep-one"),
                           "keep-two": sum(r["searched"] for r in rows.values() if r["search"] == "keep-two")},
        "labels_asked": {"duel": sum(r["duel"]["asked"] for r in rows.values() if r["duel"]),
                         "line": sum(32 for r in rows.values() if r["line"]["flat"])},
        "core_hours": {"first_part": sum(r["seconds_65"] + r["seconds_71"] for r in first.values()) / 3600,
                       "keep_one_search": sum(r["search_seconds"] for r in rows.values()
                                              if r["search"] == "keep-one") / 3600,
                       "keep_two_search": sum(r["search_seconds"] for r in rows.values()
                                              if r["search"] == "keep-two") / 3600,
                       "duel_and_line": sum(r["seconds"] - r["search_seconds"] for r in rows.values()) / 3600},
        "per_case": {str(s): {name: arms[name][s] for name in arms} | {
            "search": rows[s]["search"], "searched": rows[s]["searched"], "candidates": len(rows[s]["perfect"]),
            "winner": rows[s]["duel"]["winner"] if rows[s]["duel"] else None} for s in seeds}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--repair", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--score", type=Path)
    a = p.parse_args()
    if a.score:
        rows, result = score(a.score)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "width-32-repair.json").write_text(json.dumps([rows[s] for s in result["seeds"]]) + "\n")
        (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({key: v for key, v in result.items() if key not in ("per_case", "seeds")}, indent=1))
        return
    k.cr.install("columns")
    k.br.install()
    if a.repair:
        seeds = a.only or [s for s in SEEDS if (a.repair / f"case-{s}.json").exists()]
        prepare(a.repair, seeds)
        pfit.install()
        jobs = [(s, fe.pick_of(json.loads((a.repair / f"case-{s}.json").read_text())), a.repair) for s in seeds]
        with Pool(a.workers) as pool:  # forked after the installs above
            for s in pool.imap_unordered(fe.repair, jobs):
                print("done", s, time.strftime("%H:%M:%S"), flush=True)
        print("finished", flush=True)
        return
    ev.SAVE_EDITS = True
    a.output.mkdir(parents=True, exist_ok=True)
    for seed in list(a.only or SEEDS)[a.part::a.parts]:
        start = time.time()
        fe.first_part(seed, a.output)
        print("done", seed, round(time.time() - start), time.strftime("%H:%M:%S"), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
