"""How many labelled examples does it take to find the broken weight? (roadmap #10)

Roadmap #7 found that with the benchmark's 32 practice examples, an editor not told
which entry is broken almost never edits the right one in a large network, yet often
passes anyway. This sweep keeps each case's corruption and held-out set fixed and grows
only the practice set: 16, 64, 256 and 512 examples per stratum (repair and protected).

For each size it reports how often the min-change editor edits the corrupted entry,
how often it passes the benchmark rule, how much of the input-space damage it undoes,
and how many entries admit a practice-perfect edit.

    python experiments/locate-repair/practice_size.py --widths 8 32 128 --seeds 50 --output runs/practice-size
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("locate_repair", HERE / "analyze.py")
lr = importlib.util.module_from_spec(_spec)
sys.modules["locate_repair"] = lr
_spec.loader.exec_module(lr)

SIZES = (16, 64, 256, 512)
lr.STEP = 1e-2  # coarser value grid than analyze.py so the largest practice sets stay cheap


def practice_set(case, per_group):
    rng = np.random.default_rng([case["width"], case["seed"], 7, per_group])
    return lr.tasks._dataset(rng, case["arrays"], case["teacher"], lr.ARCHITECTURE, per_group, "practice", 2)


def analyse(case, per_group):
    try:
        dataset = practice_set(case, per_group)
    except ValueError:
        return None  # the corruption's footprint is too small to supply this many repair examples
    px, py, _ = lr.split(dataset)
    fx, fy, frepair = lr.split(case["heldout"])
    practice, final = lr.Evaluator(case["arrays"], px), lr.Evaluator(case["arrays"], fx)
    inputs = np.random.default_rng([case["width"], case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    dense = lr.Evaluator(case["arrays"], inputs)
    teacher_labels = lr.tasks._predict(case["teacher"], lr.ARCHITECTURE, inputs)
    true_entry = tuple(case["corrupted"])
    rows = []
    for name, index in lr.entries(case["arrays"]):
        current = float(case["arrays"][name].flat[index])
        value = lr.smallest_perfect(practice, py, name, index, current)
        if value is None:
            continue
        rows.append({"true": (name, index) == true_entry, "value": value,
                     "relative_change": abs(value - current) / case["scale"][name],
                     "success": lr.success(final, fy, frepair, name, index, value),
                     "damage_undone": lr.damage_undone(case, dense, teacher_labels, name, index, value)})
    chosen = min(rows, key=lambda r: r["relative_change"]) if rows else None
    given = next((r for r in rows if r["true"]), None)
    return {"width": case["width"], "seed": case["seed"], "per_group": per_group, "candidates": len(rows),
            "found_true_entry": bool(chosen and chosen["true"]), "success": bool(chosen and chosen["success"]),
            "damage_undone": chosen["damage_undone"] if chosen else None,
            "given_success": bool(given and given["success"]), "given_practice_perfect": given is not None,
            "given_damage_undone": given["damage_undone"] if given else None}


def summarize(rows, widths):
    out = {}
    for width in widths:
        for size in SIZES:
            group = [r for r in rows if r["width"] == width and r["per_group"] == size]
            if not group:
                continue
            values = lambda k: [r[k] for r in group if r[k] is not None]
            out[f"{width}/{size}"] = {
                "width": width, "per_group": size, "cases": len(group),
                "found_true_entry": float(np.mean(values("found_true_entry"))),
                "success": float(np.mean(values("success"))),
                "damage_undone_median": float(np.median(values("damage_undone"))) if values("damage_undone") else None,
                "given_success": float(np.mean(values("given_success"))),
                "given_practice_perfect": float(np.mean(values("given_practice_perfect"))),
                "candidates_median": float(np.median(values("candidates")))}
    return out


def figure(summary, widths, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    panels = (("found_true_entry", "Edited the corrupted entry"), ("success", "Passed the benchmark rule"),
              ("damage_undone_median", "Damage undone (median)"))
    for ax, (key, title) in zip(axes, panels):
        for width in widths:
            points = [(size, summary[f"{width}/{size}"][key]) for size in SIZES if f"{width}/{size}" in summary]
            ax.plot(*zip(*points), marker="o", label=f"H = {width}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES, [str(2 * s) for s in SIZES])
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.set_xlabel("practice examples")
        ax.grid(alpha=.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, help="redraw this figure from an existing practice-size.json and exit")
    args = parser.parse_args()
    if args.figure:
        report = json.loads((args.output / "practice-size.json").read_text())
        figure(report["summary"], args.widths, args.figure)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rows, unavailable = [], []
    for width in args.widths:
        for seed in range(args.seeds):
            case = lr.generate(width, seed)
            results = {size: analyse(case, size) for size in SIZES}
            if any(r is None for r in results.values()):
                # Keep only cases that can supply every size, so each curve compares the same cases.
                unavailable.append({"width": width, "seed": seed, "missing": [s for s, r in results.items() if r is None]})
                continue
            rows.extend(results.values())
        print(width, json.dumps({k: v for k, v in summarize(rows, [width]).items()}), flush=True)
    report = {"schema": "weight-space-practice-size-v1", "sizes": SIZES, "grid_step": lr.STEP,
              "summary": summarize(rows, args.widths), "unavailable": unavailable, "cases": rows}
    (args.output / "practice-size.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
