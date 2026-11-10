"""Is the Breakout score a survival-time measurement? Analyse a retained evaluation.

Reads one or more ``evaluation.json`` files written by ``lab.py breakout``, or
``evidence.json`` exports of such runs (each contains every episode's score,
step count and truncation flag), and reports how much of the mean score is
explained by episode length and the step ceiling. Optionally writes a
survival-curve / score-vs-length figure.

No simulation is run here: the analysis only reads retained episode records.

    python experiments/breakout-ceiling/analyze.py evidence/breakout-full-replay/evidence.json \
        --label "released weights" --output runs/breakout-ceiling
"""
import argparse
import json
from pathlib import Path

import numpy as np


def summarise(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema") == "weight-space-evidence-v1":
        report = report["records"]["evaluation.json"]
    rows = report["results"]
    score = np.array([r["score"] for r in rows], dtype=float)
    steps = np.array([r["steps"] for r in rows], dtype=float)
    capped = np.array([r["truncated"] for r in rows], dtype=bool)
    ceiling = report["settings"]["max_steps"]
    slope, intercept = np.polyfit(steps, score, 1)
    rate = score / steps
    return {
        "source": str(path), "checkpoint_sha256": report["checkpoint_sha256"],
        "episodes": len(rows), "max_steps": ceiling,
        "mean_score": float(score.mean()), "se": float(score.std(ddof=1) / np.sqrt(len(score))),
        "capped_episodes": int(capped.sum()), "capped_share": float(capped.mean()),
        "mean_score_capped": float(score[capped].mean()) if capped.any() else None,
        "mean_score_terminated": float(score[~capped].mean()) if (~capped).any() else None,
        "mean_steps": float(steps.mean()), "median_steps": float(np.median(steps)),
        "score_steps_correlation": float(np.corrcoef(score, steps)[0, 1]),
        "score_steps_r2": float(np.corrcoef(score, steps)[0, 1] ** 2),
        "linear_fit": {"points_per_step": float(slope), "intercept": float(intercept)},
        "points_per_step_mean": float(rate.mean()), "points_per_step_p10_p90": [float(np.percentile(rate, 10)), float(np.percentile(rate, 90))],
        "score_percentiles": {str(p): float(np.percentile(score, p)) for p in (10, 25, 50, 75, 90)},
        "survival": {str(t): float((steps >= t).mean()) for t in (1000, 2500, 5000, 7500, ceiling)},
        "interpretation": "Survival beyond the ceiling is censored; scores under a higher ceiling are not estimated here.",
    }, score, steps, capped


def plot(series, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "weight-space-lab"})
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for (label, (summary, score, steps, capped)), color in zip(series.items(), ("#2a6f97", "#c0392b", "#6a994e")):
        grid = np.arange(0, summary["max_steps"] + 1, 50)
        axes[0].step(grid, [(steps >= t).mean() for t in grid], where="post", color=color, lw=2, label=label)
        axes[1].scatter(steps[~capped], score[~capped], s=6, alpha=.5, color=color, label=f"{label}: episode ended")
        axes[1].scatter(steps[capped] + np.random.default_rng(0).uniform(-60, 60, capped.sum()), score[capped], s=6, alpha=.5,
                        color=color, marker="x", label=f"{label}: hit step ceiling")
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("share of episodes still running")
    axes[0].set_title("Survival to the 10,000-step ceiling")
    axes[1].set_xlabel("episode length (steps; capped episodes jittered)")
    axes[1].set_ylabel("episode score")
    axes[1].set_title("Score tracks survival time")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(path, metadata={"Date": None})
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--label", action="append", help="one label per evaluation file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = args.label or [p.parent.name for p in args.evaluations]
    if len(labels) != len(args.evaluations):
        parser.error("Supply one --label per evaluation file")
    args.output.mkdir(parents=True, exist_ok=False)
    series = {label: summarise(path) for label, path in zip(labels, args.evaluations)}
    summary = {"schema": "weight-space-breakout-ceiling-v1", "series": {k: v[0] for k, v in series.items()}}
    summary["figure"] = plot(series, args.output / "ceiling.svg")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
