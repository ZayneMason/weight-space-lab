"""Locate-and-repair: can 32 labelled examples identify which weight is broken? (roadmap #7)

The editing benchmark tells the editor which entry to change. This analysis removes
that hint. Each case is a 2-input ReLU network with H hidden units and one corrupted
entry anywhere among its 5H + 2 parameters. The editor sees every weight, the 32
practice inputs and (after one no-change feedback call) their labels, and may change
any single entry. Success uses the benchmark's rule on a 256-example held-out draw:
repair accuracy >= 90% and zero regressions.

Model-free reference strategies, all with the practice labels:

* ``given``: the true entry is named; take its smallest practice-perfect edit. This
  is the benchmark as it is today, solved by the strategy the task-geometry analysis
  found best.
* ``min-change``: for every entry, find its smallest practice-perfect edit; take the
  entry whose edit is smallest relative to that tensor's spread.
* ``any``: whether *some* entry's smallest practice-perfect edit succeeds (an upper
  bound for any rule that picks among those candidates).
* ``random`` / ``random-given``: the benchmark's scripted random control (three uniform
  values in [-3, 3], chosen by practice accuracy then fewer practice regressions), with
  a random entry or the true entry.

No model, API or GPU is used. Every number is exact for the declared seeds, up to the
value grid (1e-3 steps within +-4 of the current value).

    python experiments/locate-repair/analyze.py --widths 2 8 32 128 --seeds 100 --output runs/locate-repair
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from weight_space_lab import benchmark_tasks as tasks  # noqa: E402

SPAN, STEP = 4.0, 1e-3
RANDOM_DRAWS, RANDOM_REPLICATES = 3, 400
TENSORS = ("hidden_weight", "hidden_bias", "weight", "bias")
ARCHITECTURE = {"layers": [
    {"weight": "hidden_weight", "bias": "hidden_bias", "activation": "relu"},
    {"weight": "weight", "bias": "bias", "activation": "linear"}]}


def entries(arrays):
    return [(name, i) for name in TENSORS for i in range(arrays[name].size)]


def generate(width, seed, max_tries=200):
    """A teacher network and a copy with one corrupted entry that produces visible errors."""
    rng = np.random.default_rng([width, seed])
    probe = rng.uniform(-2, 2, size=(2048, 2))
    while True:  # redraw teachers that predict (almost) one class everywhere
        teacher = {"hidden_weight": rng.standard_normal((width, 2)).astype(np.float32),
                   "hidden_bias": rng.uniform(-.5, .5, width).astype(np.float32),
                   "weight": (rng.standard_normal((2, width)) / np.sqrt(width)).astype(np.float32),
                   "bias": np.zeros(2, dtype=np.float32)}
        if .1 <= tasks._predict(teacher, ARCHITECTURE, probe).mean() <= .9:
            break
    scale = {n: max(float(teacher[n].std()), 0.1) if teacher[n].size > 2 else max(float(teacher["weight"].std()), 0.1)
             for n in TENSORS}
    candidates = entries(teacher)
    for attempt in range(1, max_tries + 1):
        name, index = candidates[rng.integers(len(candidates))]
        source = {n: v.copy() for n, v in teacher.items()}
        source[name].flat[index] += np.float32(rng.choice([-1, 1]) * 1.6 * scale[name])
        try:
            practice = tasks._dataset(rng, source, teacher, ARCHITECTURE, 16, "practice", 2)
            heldout = tasks._dataset(np.random.default_rng([width, seed, 10_000_000]), source, teacher, ARCHITECTURE, 128, "final", 2)
        except ValueError:
            continue  # the corruption is (nearly) invisible; draw another entry
        return {"width": width, "seed": seed, "attempts": attempt, "arrays": source, "teacher": teacher,
                "scale": scale, "corrupted": [name, int(index)], "practice": practice, "heldout": heldout}
    raise ValueError("no visible corruption found")


def split(dataset):
    rows = dataset["examples"]
    return (np.array([r["x"] for r in rows], dtype=np.float64), np.array([r["y"] for r in rows]),
            np.array([r["id"].split("-")[1] == "repair" for r in rows]))


class Evaluator:
    """Predictions after setting one entry to each of many values, without a full forward pass."""

    def __init__(self, arrays, inputs):
        self.a = {n: v.astype(np.float64) for n, v in arrays.items()}
        self.x = inputs
        self.z = inputs @ self.a["hidden_weight"].T + self.a["hidden_bias"]  # (N, H)
        self.h = np.maximum(self.z, 0)
        self.logits = self.h @ self.a["weight"].T + self.a["bias"]  # (N, 2)

    def predict(self, name, index, values):
        values = np.asarray(values, dtype=np.float32).astype(np.float64)[:, None]  # (G, 1)
        delta = values - self.a[name].flat[index]
        logits = np.broadcast_to(self.logits, (len(values),) + self.logits.shape).copy()  # (G, N, 2)
        if name in ("hidden_weight", "hidden_bias"):
            unit = index // 2 if name == "hidden_weight" else index
            shift = delta * self.x[:, index % 2] if name == "hidden_weight" else delta  # (G, N)
            change = np.maximum(self.z[:, unit] + shift, 0) - self.h[:, unit]
            logits += change[:, :, None] * self.a["weight"][:, unit]
        elif name == "weight":
            row, unit = divmod(index, self.a["weight"].shape[1])
            logits[:, :, row] += delta * self.h[:, unit]
        else:
            logits[:, :, index] += delta
        return (logits[:, :, 1] > logits[:, :, 0]).astype(int)  # argmax, ties to class 0


def smallest_perfect(practice, y, name, index, current):
    grid = np.round(current + np.arange(-SPAN, SPAN + STEP / 2, STEP), 6)
    perfect = (practice.predict(name, index, grid) == y).all(axis=1)
    if not perfect.any():
        return None
    best = np.flatnonzero(perfect)[np.argmin(np.abs(grid[perfect] - current))]
    return float(grid[best])


def success(final, y, repair, name, index, value):
    correct = final.predict(name, index, [value])[0] == y
    return bool(correct[repair].mean() >= .9 and correct[~repair].all())


def random_control(case, practice, y, final, fy, frepair, rng, given):
    arrays, all_entries = case["arrays"], entries(case["arrays"])
    wins = 0
    for _ in range(RANDOM_REPLICATES):
        best = None
        for _ in range(RANDOM_DRAWS):
            name, index = tuple(case["corrupted"]) if given else all_entries[rng.integers(len(all_entries))]
            value = float(rng.uniform(-3, 3))
            correct = practice.predict(name, index, [value])[0] == y
            key = (correct.mean(), -int((~correct & practice_baseline(practice, y)).sum()))
            if best is None or key > best[0]:
                best = key, (name, index, value)
        wins += success(final, fy, frepair, *best[1])
    return wins / RANDOM_REPLICATES


def practice_baseline(practice, y):
    return (practice.logits[:, 1] > practice.logits[:, 0]).astype(int) == y


def damage_undone(case, dense, teacher_labels, name, index, value):
    """Share of the corruption's input-space damage that an edit removes, measured on
    20,000 uniform inputs: 1 means the network agrees with the teacher everywhere
    again, 0 means no better than the corrupted network, negative means worse."""
    before = (dense.logits[:, 1] > dense.logits[:, 0]).astype(int) != teacher_labels
    after = dense.predict(name, index, [value])[0] != teacher_labels
    return float(1 - after.mean() / max(before.mean(), 1e-12))


def analyse(case, rng):
    px, py, _ = split(case["practice"])
    fx, fy, frepair = split(case["heldout"])
    practice, final = Evaluator(case["arrays"], px), Evaluator(case["arrays"], fx)
    inputs = np.random.default_rng([case["width"], case["seed"], 99]).uniform(-2, 2, size=(20_000, 2))
    dense = Evaluator(case["arrays"], inputs)
    teacher_labels = tasks._predict(case["teacher"], ARCHITECTURE, inputs)
    true_entry = tuple(case["corrupted"])
    rows = []
    for name, index in entries(case["arrays"]):
        current = float(case["arrays"][name].flat[index])
        value = smallest_perfect(practice, py, name, index, current)
        if value is None:
            continue
        rows.append({"entry": [name, index], "value": value, "relative_change": abs(value - current) / case["scale"][name],
                     "success": success(final, fy, frepair, name, index, value), "true": (name, index) == true_entry,
                     "damage_undone": damage_undone(case, dense, teacher_labels, name, index, value)})
    given = next((r for r in rows if r["true"]), None)
    chosen = min(rows, key=lambda r: r["relative_change"]) if rows else None
    return {"width": case["width"], "seed": case["seed"], "parameters": len(entries(case["arrays"])),
            "corrupted": case["corrupted"], "attempts": case["attempts"],
            "candidates": len(rows), "successful_candidates": int(sum(r["success"] for r in rows)),
            "given": bool(given and given["success"]), "given_practice_perfect": given is not None,
            "min_change": bool(chosen and chosen["success"]), "min_change_found_true_entry": bool(chosen and chosen["true"]),
            "any": any(r["success"] for r in rows),
            "damage_before": float(((dense.logits[:, 1] > dense.logits[:, 0]).astype(int) != teacher_labels).mean()),
            "given_damage_undone": given["damage_undone"] if given else None,
            "min_change_damage_undone": chosen["damage_undone"] if chosen else None,
            "successful_candidates_damage_undone_median": float(np.median([r["damage_undone"] for r in rows if r["success"]]))
            if any(r["success"] for r in rows) else None,
            "random": random_control(case, practice, py, final, fy, frepair, rng, given=False),
            "random_given": random_control(case, practice, py, final, fy, frepair, rng, given=True)}


def med(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def summarize(rows):
    out = {}
    for width in sorted({r["width"] for r in rows}):
        group = [r for r in rows if r["width"] == width]
        mean = lambda k: float(np.mean([r[k] for r in group]))
        out[str(width)] = {"cases": len(group), "parameters": group[0]["parameters"],
                           "given": mean("given"), "min_change": mean("min_change"), "any": mean("any"),
                           "min_change_found_true_entry": mean("min_change_found_true_entry"),
                           "random": mean("random"), "random_given": mean("random_given"),
                           "candidates_median": float(np.median([r["candidates"] for r in group])),
                           "successful_candidates_median": float(np.median([r["successful_candidates"] for r in group])),
                           "corruption_attempts_mean": mean("attempts"),
                           "damage_before_median": float(np.median([r["damage_before"] for r in group])),
                           "given_damage_undone_median": med([r["given_damage_undone"] for r in group]),
                           "min_change_damage_undone_median": med([r["min_change_damage_undone"] for r in group]),
                           "successful_candidates_damage_undone_median": med([r["successful_candidates_damage_undone_median"] for r in group])}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 8, 32, 128])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261116)
    rows = []
    for width in args.widths:
        for seed in range(args.seeds):
            rows.append(analyse(generate(width, seed), rng))
        print(width, json.dumps(summarize([r for r in rows if r["width"] == width])[str(width)]), flush=True)
    report = {"schema": "weight-space-locate-repair-v1", "grid": {"span": SPAN, "step": STEP},
              "random_control": {"draws": RANDOM_DRAWS, "replicates": RANDOM_REPLICATES},
              "summary": summarize(rows), "cases": rows}
    (args.output / "locate-repair.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
