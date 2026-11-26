"""Score the cheap repair on 50 more fresh seeds against #72's held-out run (roadmap #78).

#77 ran the whole cheap repair (columns refine plus the batched committee fits, in both stages) next to the original
on 12 fresh seeds. This item runs only the cheap repair, with ``fresh_cheap.py``, on 50 more fresh seeds (400 to 449)
and compares the totals with #72's held-out run of the original repair on seeds 300 to 349. The seeds differ, so the
comparison is of rates, not case by case.

    python experiments/locate-repair/fresh_cheap.py --kind cheap --output runs/f78 --workers 4 --only $(seq 400 449)
    python experiments/locate-repair/fresh_50.py --score runs/f78
    python experiments/locate-repair/fresh_cheap.py --kind original --output runs/f78-orig --workers 2 --only 400 414
    python experiments/locate-repair/fresh_50.py --score runs/f78 --original runs/f78-orig

The last two lines rerun the original repair on the two cases the cheap run left over the 0.01% line, to see whether
the cheap refines put them there.
"""

import argparse
import json
from pathlib import Path
from statistics import median

from end_to_end import chosen
from fresh_cheap import load, score, stage
from pick_fixes import LINE
from wide_compare import EVIDENCE

OUT = EVIDENCE / "fresh-50"
SEEDS = tuple(range(400, 450))
HELD_OUT = EVIDENCE / "fresh-repair" / "check.json"


def summary(first, second):
    """The cheap run's totals in #72's terms, with #72's alongside."""
    run = score({"cheap": (first, second)})["cheap"]
    seeds = run["seeds"]
    chosen = run["chosen"]
    over = {str(s): chosen[str(s)][0] for s in seeds if chosen[str(s)][0] >= LINE}
    stages = [first[s]["outcome"]["stages_run"] for s in seeds]
    per_case = sorted(run["seconds"][str(s)] / 3600 for s in seeds)
    held = json.loads(HELD_OUT.read_text())
    held_hours = held["cpu_hours"] + held["cpu_hours_65"]
    return {
        "seeds": seeds,
        "cheap": {
            "core_hours": run["core_hours"], "core_hours_stage_65": run["core_hours_stage_65"],
            "core_hours_stage_71": run["core_hours_stage_71"], "refine_share": run["refine_share"],
            "core_hours_per_case": [round(per_case[0], 3), round(median(per_case), 3), round(per_case[-1], 3)],
            "under_0.01%": run["under_0.01%"], "under_0.1%": run["under_0.1%"], "exact": run["exact"],
            "repair_65_under_0.01%": run["repair_65_under_0.01%"],
            "repair_65_under_0.1%": sum(first[s]["outcome"]["disagreement"] < 10 * LINE for s in seeds),
            "repair_65_exact": sum(first[s]["outcome"]["faults_found"] == 3 for s in seeds),
            "gained_over_65": [s for s in seeds if chosen[str(s)][0] < LINE <= first[s]["outcome"]["disagreement"]],
            "lost_from_65": [s for s in seeds if first[s]["outcome"]["disagreement"] < LINE <= chosen[str(s)][0]],
            "over_line": over,
            "median_disagreement": median(chosen[str(s)][0] for s in seeds),
            "picks": {p: sum(second[s]["full"]["pick"] == p for s in seeds)
                      for p in sorted({second[s]["full"]["pick"] for s in seeds})},
            "bigger_ran": [s for s in seeds if second[s]["gate"]],
            "labels_asked": sum(second[s]["full"]["asked"] for s in seeds),
            "no_split": [s for s in seeds if second[s]["full"]["split_inputs"] == 0],
            "stages_run": {str(k): stages.count(k) for k in sorted(set(stages))},
            "batched_share": sum(second[s]["batched_calls"] + first[s]["batched_calls"] for s in seeds)
            / sum(second[s]["refine_calls"] + first[s]["refine_calls"] for s in seeds),
        },
        "held_out_72": {
            "seeds": held["seeds"], "core_hours": held_hours,
            "under_0.01%": held["picks"]["disagree"]["under_0.01%"], "under_0.1%": held["picks"]["disagree"]["under_0.1%"],
            "exact": held["picks"]["disagree"]["exact"], "repair_65": held["repair_65"],
            "over_line": held["over_line"], "median_disagreement": held["median"],
            "labels_asked": held["labels_asked"], "bigger_ran": held["bigger_ran"], "no_split": held["no_split"],
            "stages_run": held["stages_run"]},
        "cost_ratio": held_hours / run["core_hours"],
    }


def against_original(first, second, orig_first, orig_second):
    """The original repair on the cases it was rerun on, next to the cheap run's rows for them."""
    seeds = sorted(orig_second)
    cands = lambda row: {k: [v["disagreement"], v["faults_found"]] for k, v in row["candidates"].items()}
    cost = lambda f, s, seed: f[seed]["seconds"] + stage(s[seed])
    return {
        "seeds": seeds,
        "same_stage_65_outcome": [s for s in seeds if orig_first[s]["outcome"] == first[s]["outcome"]],
        "same_candidates": [s for s in seeds if cands(orig_second[s]) == cands(second[s])],
        "same_pick": [s for s in seeds if orig_second[s]["full"]["pick"] == second[s]["full"]["pick"]],
        "same_chosen": [s for s in seeds if chosen(orig_second[s]) == chosen(second[s])],
        "chosen": {str(s): {"original": chosen(orig_second[s]), "cheap": chosen(second[s])} for s in seeds},
        "seconds": {str(s): {"original": round(cost(orig_first, orig_second, s)), "cheap": round(cost(first, second, s))}
                    for s in seeds},
        "speedup": sum(cost(orig_first, orig_second, s) for s in seeds) / sum(cost(first, second, s) for s in seeds),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--original", type=Path)
    a = p.parse_args()
    kind, first, second = load(a.score)
    assert kind == "cheap" and sorted(second) == list(SEEDS), (kind, sorted(second))
    OUT.mkdir(parents=True, exist_ok=True)
    if a.original:
        okind, ofirst, osecond = load(a.original)
        assert okind == "original", okind
        rows = [{"seed": s, "stage_65": ofirst[s], "stage_71": osecond[s]} for s in sorted(osecond)]
        (OUT / "original.json").write_text(json.dumps(rows, indent=1) + "\n")
        result = against_original(first, second, ofirst, osecond)
        (OUT / "vs-original.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return
    rows = [{"seed": s, "stage_65": first[s], "stage_71": second[s]} for s in sorted(second)]
    (OUT / "cheap.json").write_text(json.dumps(rows, indent=1) + "\n")
    result = summary(first, second)
    (OUT / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
