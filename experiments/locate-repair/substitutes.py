"""Which weights stand in for the broken one? (roadmap #14)

Roadmap #10 found that with 1,024 practice examples the not-told editor's passing edits undo
nearly all the damage, yet most of them change an entry other than the corrupted one. This
script asks where those substitutes sit relative to the fault in the network.

For each case it lists every other entry whose smallest practice-perfect edit (1,024
practice examples) passes the benchmark rule and undoes at least 95% of the input-space
damage, and classifies it against the corrupted entry:

- mirror: the same hidden unit's weight into the other output logit (or the other output
  bias). The classifier only compares the two logits, so shifting the other one by the
  opposite amount undoes the fault exactly: an exact symmetry, not a coincidence;
- same unit: both belong to the same hidden unit (its two input weights, its bias, or its
  two outgoing weights), other than the mirror;
- same output: both feed the same output logit directly (output weight or bias);
- elsewhere.

Chance rates are the shares of all other entries in each class.

    python experiments/locate-repair/substitutes.py --widths 32 128 --seeds 50 --output runs/substitutes
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("practice_size", HERE / "practice_size.py")
ps = importlib.util.module_from_spec(_spec)
sys.modules["practice_size"] = ps
_spec.loader.exec_module(ps)
lr = ps.lr

PER_GROUP = 512
NEAR_COMPLETE = 0.95


def unit_of(name, index, width):
    if name == "hidden_weight":
        return index // 2
    if name == "hidden_bias":
        return index
    if name == "weight":
        return index % width
    return None


def output_of(name, index, width):
    if name == "weight":
        return index // width
    if name == "bias":
        return index
    return None


OUTPUT_LAYER = ("weight", "bias")
RELATIONS = ("mirror", "same unit", "same output", "elsewhere")


def relation(entry, corrupted, width):
    unit, fault_unit = unit_of(*entry, width), unit_of(*corrupted, width)
    out, fault_out = output_of(*entry, width), output_of(*corrupted, width)
    if (entry[0] == corrupted[0] and entry[0] in OUTPUT_LAYER and unit == fault_unit
            and out is not None and out != fault_out):
        return "mirror"
    if unit is not None and unit == fault_unit:
        return "same unit"
    if out is not None and out == fault_out:
        return "same output"
    return "elsewhere"


def unit_cosine(teacher, entry, corrupted, width):
    """Cosine between the input directions of the two entries' hidden units, when they sit in
    different units (None otherwise or when either entry is an output bias)."""
    unit, fault_unit = unit_of(*entry, width), unit_of(*corrupted, width)
    if unit is None or fault_unit is None or unit == fault_unit:
        return None
    a, b = teacher["hidden_weight"][unit].astype(float), teacher["hidden_weight"][fault_unit].astype(float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def chance_cosines(teacher, corrupted, width):
    """Cosines from the corrupted entry's unit to every other unit."""
    fault_unit = unit_of(*corrupted, width)
    if fault_unit is None:
        return []
    return [unit_cosine(teacher, ("hidden_bias", u), corrupted, width) for u in range(width) if u != fault_unit]


def refine(row):
    """Recompute relations and cosines for a stored row (older runs lacked mirror and cosine)."""
    width, corrupted = row["width"], tuple(row["corrupted"])
    teacher = None
    if any("cosine" not in s for s in row["substitutes"]) or "chance_cosines" not in row:
        teacher = lr.generate(width, row["seed"])["teacher"]
    for s in row["substitutes"]:
        s["relation"] = relation(tuple(s["entry"]), corrupted, width)
        if "cosine" not in s:
            s["cosine"] = unit_cosine(teacher, tuple(s["entry"]), corrupted, width)
    if "chance_cosines" not in row:
        row["chance_cosines"] = chance_cosines(teacher, corrupted, width)
    counts = dict.fromkeys(RELATIONS, 0)
    for name, size in (("hidden_weight", 2 * width), ("hidden_bias", width), ("weight", 2 * width), ("bias", 2)):
        for index in range(size):
            if (name, index) != corrupted:
                counts[relation((name, index), corrupted, width)] += 1
    row["chance_counts"] = counts
    return row


def analyse(case):
    try:
        dataset = ps.practice_set(case, PER_GROUP)
    except ValueError:
        return None
    width = case["width"]
    px, py, _ = lr.split(dataset)
    fx, fy, frepair = lr.split(case["heldout"])
    practice, final = lr.Evaluator(case["arrays"], px), lr.Evaluator(case["arrays"], fx)
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    dense = lr.Evaluator(case["arrays"], inputs)
    teacher_labels = lr.tasks._predict(case["teacher"], lr.ARCHITECTURE, inputs)
    corrupted = tuple(case["corrupted"])
    chance = dict.fromkeys(RELATIONS, 0)
    substitutes = []
    true_ok = False
    chosen = None  # the min-change editor's pick: smallest relative change, first in entry order on ties
    for name, index in lr.entries(case["arrays"]):
        entry = (name, index)
        if entry != corrupted:
            chance[relation(entry, corrupted, width)] += 1
        current = float(case["arrays"][name].flat[index])
        value = lr.smallest_perfect(practice, py, name, index, current)
        if value is None:
            continue
        change = abs(value - current) / case["scale"][name]
        if chosen is None or change < chosen["relative_change"]:
            chosen = {"entry": [name, index], "relative_change": change,
                      "relation": "corrupted" if entry == corrupted else relation(entry, corrupted, width),
                      "success": lr.success(final, fy, frepair, name, index, value)}
        if not lr.success(final, fy, frepair, name, index, value):
            continue
        if lr.damage_undone(case, dense, teacher_labels, name, index, value) < NEAR_COMPLETE:
            continue
        if entry == corrupted:
            true_ok = True
        else:
            substitutes.append({"entry": [name, index], "relation": relation(entry, corrupted, width),
                                "cosine": unit_cosine(case["teacher"], entry, corrupted, width)})
    return {"width": width, "seed": case["seed"], "corrupted": list(corrupted),
            "corrupted_layer": "hidden" if corrupted[0].startswith("hidden") else "output",
            "true_entry_repairs": true_ok, "substitutes": substitutes, "chance_counts": chance, "editor_choice": chosen,
            "chance_cosines": chance_cosines(case["teacher"], corrupted, width)}


def summarize(rows):
    out = {}
    for width in sorted({r["width"] for r in rows}):
        for layer in ("hidden", "output", "all"):
            group = [r for r in rows if r["width"] == width and (layer == "all" or r["corrupted_layer"] == layer)]
            if not group:
                continue
            subs = [s for r in group for s in r["substitutes"]]
            chance_total = {k: sum(r["chance_counts"].get(k, 0) for r in group) for k in RELATIONS}
            chosen = [r for r in group if r.get("editor_choice")]
            far = [s["cosine"] for s in subs if s["relation"] == "elsewhere" and s["cosine"] is not None]
            far_chance = [c for r in group if any(s["relation"] == "elsewhere" and s["cosine"] is not None
                                                     for s in r["substitutes"]) for c in r["chance_cosines"]]
            others = sum(chance_total.values())
            out[f"{width}/{layer}"] = {
                "cases": len(group),
                "cases_with_a_substitute": float(np.mean([bool(r["substitutes"]) for r in group])),
                "substitutes_median": float(np.median([len(r["substitutes"]) for r in group])),
                "substitute_share": {k: (sum(s["relation"] == k for s in subs) / len(subs) if subs else None)
                                     for k in chance_total},
                "chance_share": {k: v / others for k, v in chance_total.items()},
                "true_entry_repairs": float(np.mean([r["true_entry_repairs"] for r in group])),
                "editor_choice": ({k: float(np.mean([r["editor_choice"]["relation"] == k for r in chosen]))
                                   for k in ("corrupted",) + RELATIONS} if chosen else None),
                "editor_passes": float(np.mean([r["editor_choice"]["success"] for r in chosen])) if chosen else None,
                "elsewhere_units": len(far),
                "elsewhere_median_cosine": float(np.median(far)) if far else None,
                "elsewhere_cosine_above_0.9": float(np.mean([c > .9 for c in far])) if far else None,
                "chance_cosine_above_0.9": float(np.mean([c > .9 for c in far_chance])) if far_chance else None}
    return out


def figure(report, path):
    """Left: what the min-change editor picks. Right: direction cosine of other-unit substitutes
    against all other units."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, summary = report["cases"], report["summary"]
    widths = sorted({r["width"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    kinds = ("corrupted",) + RELATIONS
    labels = []
    bottoms = np.zeros(2 * len(widths))
    for kind in kinds:
        values = [summary[f"{w}/{layer}"]["editor_choice"][kind] for w in widths for layer in ("hidden", "output")]
        axes[0].bar(range(len(values)), values, bottom=bottoms, label=kind)
        bottoms += values
    labels = [f"H={w}\n{layer} fault" for w in widths for layer in ("hidden", "output")]
    axes[0].set_xticks(range(len(labels)), labels)
    axes[0].set_ylabel("share of cases")
    axes[0].set_title("Entry the min-change editor edits")
    axes[0].legend(fontsize=8, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.2))
    far = [s["cosine"] for r in rows for s in r["substitutes"] if s["relation"] == "elsewhere" and s["cosine"] is not None]
    chance = [c for r in rows if r["chance_cosines"] for c in r["chance_cosines"]]
    bins = np.linspace(-1, 1, 21)
    axes[1].hist(chance, bins=bins, density=True, alpha=.5, label="all other units")
    axes[1].hist(far, bins=bins, density=True, alpha=.7, label="units holding a substitute")
    axes[1].set_xlabel("cosine between input directions")
    axes[1].set_title("Substitutes in other units point the same way")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", type=Path, nargs="+", help="re-summarize stored substitutes.json files instead of running")
    parser.add_argument("--figure", type=Path, help="with --merge, also draw the figure here")
    args = parser.parse_args()
    if args.merge:
        reports = [json.loads(p.read_text()) for p in args.merge]
        rows = [refine(r) for rep in reports for r in rep["cases"]]
        unavailable = [u for rep in reports for u in rep["unavailable"]]
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"schema": "weight-space-substitutes-v1", "per_group": PER_GROUP, "grid_step": lr.STEP,
                  "near_complete": NEAR_COMPLETE, "summary": summarize(rows), "unavailable": unavailable, "cases": rows}
        (args.output / "substitutes.json").write_text(json.dumps(report, indent=1) + "\n")
        if args.figure:
            figure(report, args.figure)
        print(json.dumps(report["summary"], indent=1))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rows, unavailable = [], []
    for width in args.widths:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            row = analyse(lr.generate(width, seed))
            if row is None:
                unavailable.append({"width": width, "seed": seed})
                continue
            rows.append(row)
        print(width, json.dumps(summarize([r for r in rows if r["width"] == width])), flush=True)
    report = {"schema": "weight-space-substitutes-v1", "per_group": PER_GROUP, "grid_step": lr.STEP,
              "near_complete": NEAR_COMPLETE, "summary": summarize(rows), "unavailable": unavailable, "cases": rows}
    (args.output / "substitutes.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
