"""How precise must a one-entry repair be? Offline geometry of the benchmark tasks.

For every task family and many declared task seeds, sweep the single editable
entry over [-3, 3] and record which values satisfy

* the benchmark success rule on the 256-example held-out draw
  (repair accuracy >= 0.9 and zero regressions), and
* a perfect score on the 32-example practice set.

From those windows the script derives the success probability of several
reference strategies, including the benchmark's scripted random control
(three uniform draws, select by practice accuracy then fewer regressions).

No model, network, API key or GPU is used. Held-out seeds here are fixed and
declared (seed + 10_000_000) so the analysis can be repeated exactly; real
campaigns draw a fresh private held-out seed per case.

    python experiments/task-geometry/analyze.py --seeds 200 --output runs/task-geometry
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from weight_space_lab import benchmark_tasks as tasks  # noqa: E402

LOW, HIGH, STEP = -3.0, 3.0, 1e-4
RANDOM_DRAWS = 3          # min(proposal budget 5, feedback budget 3) in the pilot config
RANDOM_REPLICATES = 4000


def _forward(spec, values, inputs):
    """Predictions for a batch of candidate values of the editable entry. Returns [len(values), len(inputs)]."""
    edit = spec["editable"]
    values = np.asarray(values, dtype=np.float32).astype(np.float64)
    x = np.broadcast_to(np.asarray(inputs, dtype=np.float64), (len(values),) + np.shape(inputs))
    for layer in spec["architecture"]["layers"]:
        weight = np.broadcast_to(spec["arrays"][layer["weight"]].astype(np.float64), (len(values),) + spec["arrays"][layer["weight"]].shape).copy()
        bias = np.broadcast_to(spec["arrays"][layer["bias"]].astype(np.float64), (len(values),) + spec["arrays"][layer["bias"]].shape).copy()
        for name, tensor in ((layer["weight"], weight), (layer["bias"], bias)):
            if name == edit["tensor"]:
                tensor.reshape(len(values), -1)[:, edit["flat_indices"][0]] = values
        x = np.einsum("gni,goi->gno", x, weight) + bias[:, None, :]
        if layer["activation"] == "relu":
            x = np.maximum(x, 0)
    return x.argmax(axis=2)


def _split(dataset):
    rows = dataset["examples"]
    return (np.array([r["x"] for r in rows]), np.array([r["y"] for r in rows]),
            np.array([r["id"].split("-")[1] == "repair" for r in rows]))


def _scores(spec, values, dataset, chunk=2000):
    x, y, repair = _split(dataset)
    correct = np.concatenate([_forward(spec, values[i:i + chunk], x) == y for i in range(0, len(values), chunk)])
    # Strata are defined by baseline correctness: protected rows are exactly the baseline-correct rows.
    return correct, repair


def _success(correct, repair):
    return (correct[:, repair].mean(axis=1) >= .9) & correct[:, ~repair].all(axis=1)


def _intervals(mask, grid):
    """Contiguous True runs of a boolean mask over the grid, as [low, high] pairs."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [[float(grid[a]), float(grid[b - 1])] for a, b in zip(edges[::2], edges[1::2])]


def analyse_case(family, seed, rng):
    spec = tasks.generate(family, seed, seed + 10_000_000)
    grid = np.round(np.arange(LOW, HIGH + STEP / 2, STEP), 6)
    final_correct, final_repair = _scores(spec, grid, spec["heldout"])
    practice_correct, _ = _scores(spec, grid, spec["practice"])
    success = _success(final_correct, final_repair)
    practice_perfect = practice_correct.all(axis=1)
    teacher = spec["control_value"]
    original = float(spec["arrays"][spec["editable"]["tensor"]].flat[spec["editable"]["flat_indices"][0]])
    windows = _intervals(success, grid)
    practice_windows = _intervals(practice_perfect, grid)
    both = (success & practice_perfect).sum()
    midpoint = None
    if len(practice_windows) == 1:
        mid = sum(practice_windows[0]) / 2
        c, r = _scores(spec, np.array([mid]), spec["heldout"])
        midpoint = bool(_success(c, r)[0])
    # Benchmark random control, replicated: three uniform draws, select by (practice accuracy, -regressions), first wins ties.
    draws = rng.uniform(LOW, HIGH, size=(RANDOM_REPLICATES, RANDOM_DRAWS)).astype(np.float32)
    flat = draws.reshape(-1)
    p_correct, p_repair = _scores(spec, flat, spec["practice"])
    key = p_correct.mean(axis=1) * 1000 - (~p_correct[:, ~p_repair]).sum(axis=1)
    pick = key.reshape(RANDOM_REPLICATES, RANDOM_DRAWS).argmax(axis=1)
    chosen = draws[np.arange(RANDOM_REPLICATES), pick]
    f_correct, f_repair = _scores(spec, chosen, spec["heldout"])
    random_success = float(_success(f_correct, f_repair).mean())
    width = float(success.sum() * STEP)
    return {
        "family": family, "seed": seed, "teacher_value": teacher, "corrupted_value": original,
        "success_windows": windows, "success_width": width,
        "success_width_relative": width / abs(teacher) if teacher else None,
        "teacher_in_success_window": any(lo - STEP <= teacher <= hi + STEP for lo, hi in windows),
        "practice_perfect_windows": practice_windows, "practice_perfect_width": float(practice_perfect.sum() * STEP),
        "p_success_given_practice_perfect": float(both / practice_perfect.sum()) if practice_perfect.any() else None,
        "practice_midpoint_success": midpoint,
        "random_control_success": random_success,
        "uniform_single_draw_success": float(success.mean()),
    }


def summarise(rows):
    out = {}
    for family in tasks.FAMILIES:
        group = [r for r in rows if r["family"] == family]
        if not group:
            continue
        widths = np.array([r["success_width"] for r in group])
        rel = np.array([r["success_width_relative"] for r in group])
        pp = np.array([r["p_success_given_practice_perfect"] for r in group if r["p_success_given_practice_perfect"] is not None])
        mids = [r["practice_midpoint_success"] for r in group if r["practice_midpoint_success"] is not None]
        rnd = np.array([r["random_control_success"] for r in group])
        out[family] = {
            "cases": len(group),
            "teacher_always_succeeds": all(r["teacher_in_success_window"] for r in group),
            "single_interval_cases": sum(len(r["success_windows"]) == 1 for r in group),
            "success_width_median": float(np.median(widths)),
            "success_width_p10_p90": [float(np.percentile(widths, 10)), float(np.percentile(widths, 90))],
            "success_width_relative_median": float(np.median(rel)),
            "practice_perfect_width_median": float(np.median([r["practice_perfect_width"] for r in group])),
            "p_success_given_practice_perfect_mean": float(pp.mean()) if len(pp) else None,
            "practice_midpoint_success_rate": float(np.mean(mids)) if mids else None,
            "random_control_success_mean": float(rnd.mean()),
            "random_control_success_by_case_p10_p90": [float(np.percentile(rnd, 10)), float(np.percentile(rnd, 90))],
            "uniform_single_draw_success_mean": float(np.mean([r["uniform_single_draw_success"] for r in group])),
        }
    all_random = np.array([r["random_control_success"] for r in rows])
    out["all_families"] = {"cases": len(rows), "random_control_success_mean": float(all_random.mean()),
        "expected_random_successes_in_12_cases": float(all_random.mean() * 12)}
    return out


def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "weight-space-lab"})
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    colors = {"threshold": "#2a6f97", "slope": "#c0392b", "routing": "#6a994e", "hidden": "#8e6c8a"}
    for family in tasks.FAMILIES:
        group = [r for r in rows if r["family"] == family]
        rel = np.sort([r["success_width_relative"] for r in group])
        axes[0].plot(rel, np.linspace(0, 1, len(rel)), label=family, color=colors[family], lw=2)
        pp = np.sort([r["p_success_given_practice_perfect"] for r in group if r["p_success_given_practice_perfect"] is not None])
        axes[1].plot(pp, np.linspace(0, 1, len(pp)), label=family, color=colors[family], lw=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("success-window width / |correct value|")
    axes[0].set_ylabel("fraction of task seeds")
    axes[0].set_title("How precise must the edit be?")
    axes[1].set_xlabel("P(held-out success | perfect on practice)")
    axes[1].set_title("Does practice feedback pin the value down?")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left")
    fig.savefig(path, metadata={"Date": None})
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=200, help="task seeds per family (0..N-1)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rng-seed", type=int, default=20261110)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.rng_seed)
    rows = [analyse_case(family, seed, rng) for family in tasks.FAMILIES for seed in range(args.seeds)]
    summary = {"schema": "weight-space-task-geometry-v1", "grid": {"low": LOW, "high": HIGH, "step": STEP},
               "random_control": {"draws": RANDOM_DRAWS, "replicates_per_case": RANDOM_REPLICATES, "rng_seed": args.rng_seed},
               "heldout_seed_rule": "task seed + 10_000_000 (declared; real campaigns use a private random seed)",
               "families": summarise(rows)}
    (args.output / "cases.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure = plot(rows, args.output / "geometry.svg")
    print(json.dumps({**summary, "figure": figure}, indent=2))


if __name__ == "__main__":
    main()
