"""Does the author's bet only pay off because of sticky actions? (roadmap #16)

Roadmaps #12, #13 and #15 explain the author's ascending guesses as a hedge against MinAtar's
sticky actions: they keep the paddle early, so a repeated move rarely costs a catch. If that
is right, the accurate table's handicap should shrink as sticky actions become rarer and
grow as they become more common.

This script scores the published table, #8's accurate table and the perfect-prediction
("oracle-ascending") variant over a range of sticky-action probabilities. Scores cap at
10,000 steps, so the comparison uses the death hazard: deaths per 10,000 steps.

    python experiments/breakout-anatomy/sticky_sweep.py CHECKPOINT --output runs/breakout-sticky
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("breakout_rowswap", HERE / "rowswap.py")
rowswap = importlib.util.module_from_spec(_spec)
sys.modules["breakout_rowswap"] = rowswap
_spec.loader.exec_module(rowswap)
hedge, targets, anatomy = rowswap.hedge, rowswap.targets, rowswap.anatomy

STICKY = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3)


def _work(task):
    index, arrays, mode, sticky = task
    return targets.run_episode(index, 0, 10000, arrays, mode, sticky=sticky)


def hazard(rows):
    """Deaths per 10,000 steps, with a Poisson standard error."""
    deaths = sum(not r["truncated"] for r in rows)
    steps = sum(r["steps"] for r in rows)
    return {"deaths": int(deaths), "steps": int(steps), "per_10k_steps": 1e4 * deaths / steps,
            "per_10k_steps_se": 1e4 * np.sqrt(max(deaths, 1)) / steps}


def run(arrays, mode, sticky, episodes, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_work, [(i, arrays, mode, sticky) for i in range(episodes)], chunksize=2))
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    return {"sticky": sticky, "episodes": episodes, "mean_score": float(scores.mean()),
            "se": float(scores.std(ddof=1) / np.sqrt(len(rows))),
            "capped_share": float(np.mean([r["truncated"] for r in rows])),
            "score_per_10k_steps": 1e4 * float(scores.sum()) / sum(r["steps"] for r in rows),
            **hazard(rows), "scores": scores.astype(int).tolist()}


def rallies(index, arrays, sticky=0.0, max_steps=10000):
    """Paddle returns (catches) and bricks in one episode: how many returns hit a brick."""
    from minatar import Environment
    env = Environment("breakout", sticky_action_prob=sticky)
    env.seed(index)
    env.reset()
    rng = np.random.default_rng(index + 1_000_000)
    weights = anatomy.as_list(arrays)
    catches = bricks = 0
    for step in range(1, max_steps + 1):
        inner = env.env
        arriving = inner.ball_dir in (2, 3) and inner.ball_y == 8
        x = anatomy.breakout.observation(env.state())
        logits = anatomy.breakout.policy_logits(x, weights)
        p = np.exp(logits.astype(np.float64) - logits.max())
        p /= p.sum()
        reward, terminal = env.act(int(rng.choice(6, p=p)))
        bricks += reward
        catches += bool(arriving and not terminal)
        if terminal:
            break
    return {"episode": index, "steps": step, "catches": catches, "bricks": int(bricks)}


def figure(report, path):
    """Left: deaths per 10,000 steps. Right: bricks per 10,000 steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, rows in report["variants"].items():
        rows = [r for r in rows if r["sticky"] > 0]
        x = [r["sticky"] for r in rows]
        axes[0].errorbar(x, [r["per_10k_steps"] for r in rows], yerr=[r["per_10k_steps_se"] for r in rows],
                         marker="o", capsize=3, label=name)
    for name, rows in report["variants"].items():
        axes[1].plot([r["sticky"] for r in rows], [r["score_per_10k_steps"] for r in rows], marker="o", label=name)
    axes[0].set_yscale("log")
    axes[0].set_title("Deaths per 10,000 steps")
    axes[1].set_title("Bricks per 10,000 steps")
    for ax in axes:
        ax.set_xlabel("sticky-action probability")
        ax.grid(alpha=.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sticky", type=float, nargs="+", default=list(STICKY))
    parser.add_argument("--figure", type=Path, help="draw the figure from an existing sticky.json in --output")
    parser.add_argument("--rallies", type=int, default=0, help="without sticky actions, count returns and bricks in N episodes")
    args = parser.parse_args()
    if args.figure:
        figure(json.loads((args.output / "sticky.json").read_text()), args.figure)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = anatomy.load(args.checkpoint)
    published = hedge.published_table(arrays)
    accurate = targets.set_ascending_targets(arrays, rowswap.mix(published, rowswap.accurate_table(), set(range(10))))
    variants = {"published": (arrays, "policy"), "accurate": (accurate, "policy"), "perfect": (arrays, "oracle-ascending")}
    if args.rallies:
        out = {name: [rallies(i, weights) for i in range(args.rallies)]
               for name, (weights, mode) in variants.items() if mode == "policy"}
        (args.output / "rallies.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out))
        return
    report = {"schema": "weight-space-breakout-sticky-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
              "episodes": args.episodes, "variants": {}}
    for name, (weights, mode) in variants.items():
        report["variants"][name] = []
        for sticky in args.sticky:
            result = run(weights, mode, sticky, args.episodes, args.workers)
            report["variants"][name].append(result)
            print(name, json.dumps(result), flush=True)
            (args.output / "sticky.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
