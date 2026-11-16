"""Two broken weights: can single edits still find and fix them? (roadmap #20)

Roadmap #14 found that every single-entry fault has an exact twin (the same unit's weight into
the other output logit) and many near-substitutes, so a one-edit repair often passes without
touching the broken entry. This experiment corrupts two entries, in different hidden units,
so that no single edit can undo both.

Each case is a 2-input ReLU network with H hidden units (5H + 2 parameters) and 1,024 practice
examples (512 repair, 512 protected), as in #14. Strategies, all model-free:

- ``one edit``: #7's min-change rule, the smallest practice-perfect single edit;
- ``greedy``: two edits. First the single edit with the best practice accuracy (smallest change
  among ties), then the smallest practice-perfect edit on the result, or the best-accuracy
  edit if none is perfect;
- ``told``: the same greedy procedure restricted to the two corrupted entries, in the order
  that does better on practice;
- ``alternating``: the two entries the greedy repair chose (or, for ``told alternating``, the
  two corrupted entries), re-fitted in turn, each to its best practice accuracy, for up to
  six passes;
- ``joint``: a 200 x 200 grid over both entries of the pair at once (each within 2.5 tensor
  spreads of its current value; the grid does not contain the shift of 1.6 spreads exactly),
  best practice accuracy, smallest total relative change among ties;
- ``restored``: both corrupted entries put back to the teacher's values (a check on the rule).

Success is the benchmark rule on a 256-example held-out draw: repair accuracy >= 90% and no
regressions. Damage undone is measured on 20,000 uniform inputs, as in #7.

    python experiments/locate-repair/two_faults.py --widths 8 32 --seeds 50 --output runs/two-faults
    python experiments/locate-repair/two_faults.py --output runs/two-faults --figure two-faults.png
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("substitutes", HERE / "substitutes.py")
sub = importlib.util.module_from_spec(_spec)
sys.modules["substitutes"] = sub
_spec.loader.exec_module(sub)
lr = sub.lr  # analyze.py, with practice_size.py's coarser value grid (step 0.01)

PER_GROUP = 512


def corrupt(width, seed, max_tries=200):
    """The single-fault case from #7 plus a second corrupted entry in a different hidden unit
    (output biases count as their own unit). Both shifts are 1.6 tensor spreads."""
    case = lr.generate(width, seed)
    rng = np.random.default_rng([width, seed, 2])
    first = tuple(case["corrupted"])
    candidates = lr.entries(case["arrays"])
    for _ in range(max_tries):
        second = candidates[rng.integers(len(candidates))]
        if second == first or sub.relation(second, first, width) in ("mirror", "same unit"):
            continue
        source = {n: v.copy() for n, v in case["arrays"].items()}
        name, index = second
        source[name].flat[index] += np.float32(rng.choice([-1, 1]) * 1.6 * case["scale"][name])
        try:
            practice = lr.tasks._dataset(np.random.default_rng([width, seed, 7, PER_GROUP]), source, case["teacher"],
                                         lr.ARCHITECTURE, PER_GROUP, "practice", 2)
            heldout = lr.tasks._dataset(np.random.default_rng([width, seed, 10_000_000]), source, case["teacher"],
                                        lr.ARCHITECTURE, 128, "final", 2)
        except ValueError:
            continue
        return dict(case, arrays=source, faults=[list(first), [name, int(index)]], practice=practice, heldout=heldout)
    return None


def best_edit(evaluator, y, name, index, current):
    """Over the value grid: (practice accuracy, value) with the best accuracy, closest to current."""
    grid = np.round(current + np.arange(-lr.SPAN, lr.SPAN + lr.STEP / 2, lr.STEP), 6)
    accuracy = (evaluator.predict(name, index, grid) == y).mean(axis=1)
    top = np.flatnonzero(accuracy == accuracy.max())
    pick = top[np.argmin(np.abs(grid[top] - current))]
    return float(accuracy[pick]), float(grid[pick])


def choose(evaluator, y, arrays, scale, entries):
    """The edit with the best practice accuracy, smallest relative change among ties."""
    best = None
    for name, index in entries:
        current = float(arrays[name].flat[index])
        accuracy, value = best_edit(evaluator, y, name, index, current)
        key = (accuracy, -abs(value - current) / scale[name])
        if best is None or key > best[0]:
            best = key, (name, index, value)
    return best[0][0], best[1]


def apply(arrays, edits):
    out = {n: v.copy() for n, v in arrays.items()}
    for name, index, value in edits:
        out[name].flat[index] = np.float32(value)
    return out


def predict(arrays, x):
    a = {n: v.astype(np.float64) for n, v in arrays.items()}
    logits = np.maximum(x @ a["hidden_weight"].T + a["hidden_bias"], 0) @ a["weight"].T + a["bias"]
    return (logits[:, 1] > logits[:, 0]).astype(int)


def outcome(case, edits, fx, fy, frepair, inputs, teacher_labels):
    edited = apply(case["arrays"], edits)
    correct = predict(edited, fx) == fy
    before = predict(case["arrays"], inputs) != teacher_labels
    after = predict(edited, inputs) != teacher_labels
    return {"success": bool(correct[frepair].mean() >= .9 and correct[~frepair].all()),
            "damage_undone": float(1 - after.mean() / max(before.mean(), 1e-12))}


def matches(entry, faults, width):
    """Which faults an edited entry accounts for: the entry itself or its exact mirror."""
    return [k for k, fault in enumerate(faults)
            if tuple(entry) == tuple(fault) or sub.relation(tuple(entry), tuple(fault), width) == "mirror"]


def greedy(case, px, py, entries):
    """Two edits: best practice accuracy first, then the smallest perfect edit (or best again)."""
    arrays, scale = case["arrays"], case["scale"]
    _, first = choose(lr.Evaluator(arrays, px), py, arrays, scale, entries)
    after = apply(arrays, [first])
    evaluator = lr.Evaluator(after, px)
    perfect = []
    for name, index in entries:
        current = float(after[name].flat[index])
        value = lr.smallest_perfect(evaluator, py, name, index, current)
        if value is not None:
            perfect.append((abs(value - current) / scale[name], (name, index, value)))
    if perfect:
        second, accuracy = min(perfect)[1], 1.0
    else:
        accuracy, second = choose(evaluator, py, after, scale, entries)
    return [first, second], accuracy


def alternate(case, px, py, pair, passes=6):
    """Re-fit each entry of the pair in turn to its best practice accuracy, until perfect."""
    arrays, accuracy = {n: v.copy() for n, v in case["arrays"].items()}, 0.0
    for _ in range(passes):
        for name, index in pair:
            accuracy, value = best_edit(lr.Evaluator(arrays, px), py, name, index, float(arrays[name].flat[index]))
            arrays[name].flat[index] = np.float32(value)
            if accuracy == 1.0:
                break
        if accuracy == 1.0:
            break
    return [(n, i, float(arrays[n].flat[i])) for n, i in pair], accuracy


def joint(case, px, py, pair, points=200, span=2.5):
    """Best practice accuracy over a grid of both entries at once (one entry: its own best edit)."""
    arrays, scale = case["arrays"], case["scale"]
    if len(pair) == 1:
        (name, index), = pair
        accuracy, value = best_edit(lr.Evaluator(arrays, px), py, name, index, float(arrays[name].flat[index]))
        return [(name, index, value)], accuracy
    (n1, i1), (n2, i2) = pair
    c1, c2 = float(arrays[n1].flat[i1]), float(arrays[n2].flat[i2])
    offsets = np.linspace(-span, span, points)
    g1, g2 = c1 + offsets * scale[n1], c2 + offsets * scale[n2]
    best = None
    for v in g1:
        trial = {n: a.copy() for n, a in arrays.items()}
        trial[n1].flat[i1] = np.float32(v)
        accuracy = (lr.Evaluator(trial, px).predict(n2, i2, g2) == py).mean(axis=1)
        cost = abs(v - c1) / scale[n1] + np.abs(g2 - c2) / scale[n2]
        j = np.lexsort((cost, -accuracy))[0]
        if best is None or (accuracy[j], -cost[j]) > best[0]:
            best = (accuracy[j], -cost[j]), [(n1, i1, float(v)), (n2, i2, float(g2[j]))]
    return best[1], float(best[0][0])


def analyse(case):
    width, faults = case["width"], case["faults"]
    px, py, _ = lr.split(case["practice"])
    fx, fy, frepair = lr.split(case["heldout"])
    inputs = np.random.default_rng([width, case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    teacher_labels = predict(case["teacher"], inputs)
    entries = lr.entries(case["arrays"])
    practice = lr.Evaluator(case["arrays"], px)
    # one edit: smallest practice-perfect single edit
    perfect = []
    for name, index in entries:
        current = float(case["arrays"][name].flat[index])
        value = lr.smallest_perfect(practice, py, name, index, current)
        if value is not None:
            perfect.append((abs(value - current) / case["scale"][name], (name, index, value)))
    one = {"practice_perfect_candidates": len(perfect)}
    if perfect:
        edit = min(perfect)[1]
        one.update(outcome(case, [edit], fx, fy, frepair, inputs, teacher_labels), matched=matches(edit[:2], faults, width))
    else:
        one.update(success=False, damage_undone=None, matched=[])
    # greedy two edits, not told and told
    rows = {"one_edit": one}
    for label, pool in (("greedy", entries), ("told", None)):
        if pool is None:
            options = [greedy(case, px, py, [tuple(f) for f in order]) for order in (faults, faults[::-1])]
            edits, accuracy = max(options, key=lambda o: o[1])
        else:
            edits, accuracy = greedy(case, px, py, pool)
        found = sorted({k for e in edits for k in matches(e[:2], faults, width)})
        rows[label] = dict(outcome(case, edits, fx, fy, frepair, inputs, teacher_labels), practice_accuracy=accuracy,
                           faults_found=len(found), edits=[[n, int(i), v] for n, i, v in edits])
    chosen = list(dict.fromkeys((n, int(i)) for n, i, _ in rows["greedy"]["edits"]))
    told_pair = [tuple(f) for f in faults]
    for label, method, pair in (("greedy_alternating", alternate, chosen), ("told_alternating", alternate, told_pair),
                                ("greedy_joint", joint, chosen), ("told_joint", joint, told_pair)):
        edits, accuracy = method(case, px, py, pair)
        found = sorted({k for e in edits for k in matches(e[:2], faults, width)})
        rows[label] = dict(outcome(case, edits, fx, fy, frepair, inputs, teacher_labels), practice_accuracy=accuracy,
                           faults_found=len(found), edits=[[n, int(i), v] for n, i, v in edits])
    restore = [(n, i, float(case["teacher"][n].flat[i])) for n, i in faults]
    rows["restored"] = outcome(case, restore, fx, fy, frepair, inputs, teacher_labels)
    return {"width": width, "seed": case["seed"], "parameters": len(entries), "faults": faults,
            "damage_before": float((predict(case["arrays"], inputs) != teacher_labels).mean()), **rows}


def summarize(rows):
    out = {}
    for width in sorted({r["width"] for r in rows}):
        group = [r for r in rows if r["width"] == width]
        med = lambda vals: float(np.median([v for v in vals if v is not None])) if any(v is not None for v in vals) else None
        out[str(width)] = {
            "cases": len(group), "parameters": group[0]["parameters"],
            "one_edit_has_perfect_candidate": float(np.mean([r["one_edit"]["practice_perfect_candidates"] > 0 for r in group])),
            "one_edit_success": float(np.mean([r["one_edit"]["success"] for r in group])),
            "one_edit_damage_undone_median": med([r["one_edit"]["damage_undone"] for r in group]),
            "one_edit_hit_a_fault": float(np.mean([bool(r["one_edit"]["matched"]) for r in group]))}
        out[str(width)]["restored_success"] = float(np.mean([r["restored"]["success"] for r in group]))
        for label in ("greedy", "told", "greedy_alternating", "told_alternating", "greedy_joint", "told_joint"):
            out[str(width)].update({
                f"{label}_success": float(np.mean([r[label]["success"] for r in group])),
                f"{label}_damage_undone_median": med([r[label]["damage_undone"] for r in group]),
                f"{label}_practice_perfect": float(np.mean([r[label]["practice_accuracy"] == 1.0 for r in group])),
                f"{label}_found_both": float(np.mean([r[label]["faults_found"] == 2 for r in group])),
                f"{label}_found_one": float(np.mean([r[label]["faults_found"] == 1 for r in group]))})
    return out


LABELS = {"one_edit": "one edit", "greedy": "two edits", "greedy_joint": "two edits,\njoint re-fit",
          "told": "told, two edits", "told_alternating": "told, fit\nin turn", "told_joint": "told, joint"}


def figure(report, path):
    """Benchmark pass rate per repair, split by how many broken entries it touched."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    widths = sorted(report["summary"], key=int)
    fig, axes = plt.subplots(1, len(widths), figsize=(5.2 * len(widths), 4), sharey=True)
    for ax, width in zip(np.atleast_1d(axes), widths):
        group = [r for r in report["cases"] if str(r["width"]) == width]
        for k, label in enumerate(LABELS):
            found = [len(r[label]["matched"]) if label == "one_edit" else r[label]["faults_found"] for r in group]
            passed = [r[label]["success"] for r in group]
            bottom = 0
            for n, colour in ((2, "#2a7"), (1, "#8c6"), (0, "#ccc")):
                share = np.mean([p and f == n for p, f in zip(passed, found)])
                ax.bar(k, share, bottom=bottom, color=colour, label=f"passed, touched {n} broken" if k == 0 else None)
                bottom += share
        ax.set_xticks(range(len(LABELS)), list(LABELS.values()), fontsize=8)
        ax.set_title(f"H = {width}: share of {len(group)} cases passing the benchmark")
        ax.set_ylim(0, 1)
        ax.grid(alpha=.3, axis="y")
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, help="draw the figure from an existing two-faults.json in --output")
    args = parser.parse_args()
    if args.figure:
        figure(json.loads((args.output / "two-faults.json").read_text()), args.figure)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for width in args.widths:
        for seed in range(args.seeds):
            case = corrupt(width, seed)
            if case is None:
                skipped.append({"width": width, "seed": seed})
                continue
            rows.append(analyse(case))
        print(width, json.dumps(summarize([r for r in rows if r["width"] == width])[str(width)]), flush=True)
        report = {"schema": "weight-space-two-faults-v3", "per_group": PER_GROUP, "grid_step": lr.STEP,
                  "summary": summarize(rows), "skipped": skipped, "cases": rows}
        (args.output / "two-faults.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
