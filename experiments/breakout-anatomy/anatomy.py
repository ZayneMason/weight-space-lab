"""Anatomy of the released hand-written Breakout network (roadmap #2).

The released Takehi et al. Experiment 5 answer is a 600-32-32-6 tanh policy whose
author described it as a program: four "landing-position error" units, two direction
detectors, and a second layer of gated left/right comparisons. This script tests that
description against the game itself and against interventions:

* ``semantics``: plays the intact network, and at every step compares what each unit
  computes with the ground truth from an exact replica of MinAtar's ball physics
  (where the ball will reach the paddle row, which way it is moving).
* ``evaluate``: runs a list of variants at the published protocol (sampled softmax
  actions, sticky actions, 10,000-step ceiling, per-episode seeds as ``lab.py
  breakout``): second-layer unit ablations, weight noise, coarser weight precision,
  greedy actions, and a sweep of the environment's sticky-action probability.
* ``figures``: receptive fields and survival curves from the saved outputs.

    python experiments/breakout-anatomy/anatomy.py all CHECKPOINT --output runs/breakout-anatomy

The episode loop is the protocol's own loop with two knobs exposed (sticky-action
probability and greedy selection). With the defaults it reproduces
``reproduce_breakout.run_episode`` exactly; ``tests/test_breakout_anatomy.py`` checks
that, and checks the ball replica against the real environment.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("reproduce_breakout", ROOT / "skills/weight-space-lab/scripts/reproduce_breakout.py")
breakout = importlib.util.module_from_spec(_spec)
sys.modules["reproduce_breakout"] = breakout
_spec.loader.exec_module(breakout)

NAMES = ["actor/dense1/kernel", "actor/dense1/bias", "actor/dense2/kernel", "actor/dense2/bias",
         "actor/logits/kernel", "actor/logits/bias"]
PADDLE, BALL, TRAIL = 0, 3, 4  # observation channels after the protocol's mapping
MOVES = {0: (-1, -1), 1: (1, -1), 2: (1, 1), 3: (-1, 1)}  # MinAtar ball_dir -> (dx, dy)
# The author's context for each first-layer target unit, as MinAtar ball directions.
CONTEXT_UNIT = {2: 0, 3: 1, 1: 2, 0: 3}  # down-right, down-left, up-right, up-left
TARGET_SCALE = 0.02  # each column of target or paddle position is 0.02 of pre-activation


def feature(row, col, channel):
    return (row * 10 + col) * 6 + channel


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {n: archive[n].astype(np.float32).copy() for n in NAMES}


def as_list(arrays):
    return [arrays[n] for n in NAMES]


def forward(x, arrays):
    w1, b1, w2, b2, w3, b3 = as_list(arrays)
    pre1 = x @ w1 + b1
    h1 = np.tanh(pre1)
    h2 = np.tanh(h1 @ w2 + b2)
    return pre1, h1, h2, h2 @ w3 + b3


# ---------------------------------------------------------------- ball physics replica

def ball_future(x, y, direction, bricks, strike, limit=1000):
    """Replay MinAtar Breakout's ball rules until the ball reaches the paddle row.

    Returns the number of steps until that move and the two catch columns: the
    paddle survives if it is under the ball's column before the move or after it
    (MinAtar checks ``ball_x == pos`` or ``new_x == pos``). Paddle actions cannot
    change the ball's path before that move, so this is exact ground truth.
    """
    bricks = np.array(bricks, dtype=bool, copy=True)
    for step in range(1, limit + 1):
        last_x, last_y = x, y
        dx, dy = MOVES[direction]
        new_x, new_y = x + dx, y + dy
        toggle = False
        if new_x < 0 or new_x > 9:
            new_x = min(max(new_x, 0), 9)
            direction = [1, 0, 3, 2][direction]
        if new_y < 0:
            new_y = 0
            direction = [3, 2, 1, 0][direction]
        elif bricks[new_y, new_x]:
            toggle = True
            if not strike:
                strike = True
                bricks[new_y, new_x] = False
                new_y = last_y
                direction = [3, 2, 1, 0][direction]
        elif new_y == 9:
            return {"steps": step, "catch": sorted({last_x, new_x})}
        if not toggle:
            strike = False
        x, y = new_x, new_y
    raise RuntimeError("ball never reached the paddle row")


def env_future(inner):
    return ball_future(inner.ball_x, inner.ball_y, inner.ball_dir, inner.brick_map == 1, inner.strike)


# ---------------------------------------------------------------- episode loop

class Recorder:
    """MinAtar's sticky-action wrapper, written out so overrides can be counted.

    It draws from the environment's RandomState exactly as ``Environment.act`` does,
    including when the probability is 0, so every sticky setting sees the same
    ball starts.
    """

    def __init__(self, env):
        self.env = env
        self.overrides = []

    def act(self, action):
        sticky = self.env.random.rand() < self.env.sticky_action_prob
        executed = self.env.last_action if sticky else action
        self.overrides.append(bool(sticky and executed != action))
        self.env.last_action = executed
        return self.env.env.act(executed)


def run_episode(index, seed, max_steps, arrays, sticky=0.1, greedy=False, trace=False):
    from minatar import Environment
    env = Environment("breakout", sticky_action_prob=sticky)
    env.seed(seed + index)
    env.reset()
    recorder = Recorder(env)
    rng = np.random.default_rng(seed + index + 1_000_000)
    weights = as_list(arrays)
    score, terminal, rows = 0, False, []
    for step in range(1, max_steps + 1):
        x = breakout.observation(env.state())
        logits = breakout.policy_logits(x, weights)
        p = np.exp(logits.astype(np.float64) - logits.max())
        p /= p.sum()
        action = int(np.argmax(p)) if greedy else int(rng.choice(6, p=p))
        if trace:
            rows.append(trace_row(env.env, x, arrays, p, action))
        r, terminal = recorder.act(action)
        score += r
        if terminal:
            break
    tail = recorder.overrides[-12:]
    return {"episode": index, "score": score, "steps": step, "truncated": not terminal,
            "overrides": int(sum(recorder.overrides)), "override_in_last_12": bool(terminal and any(tail))}, rows


def trace_row(inner, x, arrays, probabilities, action):
    pre1, h1, h2, logits = forward(x, arrays)
    future = env_future(inner)
    direction = int(inner.ball_dir)
    unit = CONTEXT_UNIT[direction]
    weight = float(arrays["actor/dense1/kernel"][feature(inner.ball_y, inner.ball_x, BALL), unit])
    # Context the network selects: layer-2 pairs are gated by the signs of units 4 and 5.
    right, down = h1[4] > 0, h1[5] > 0
    chosen = {(True, True): 0, (False, True): 1, (True, False): 2, (False, False): 3}[(bool(right), bool(down))]
    return {"ball": [int(inner.ball_x), int(inner.ball_y)], "dir": direction, "paddle": int(inner.pos),
            "catch": future["catch"], "steps_to_row9": future["steps"], "implied_target": weight / TARGET_SCALE,
            "context_true": unit, "context_chosen": chosen, "h1": h1[:6].round(4).tolist(),
            "max_probability": float(probabilities.max()), "action": action,
            "trail": [int(inner.last_x), int(inner.last_y)], "wall_frame": bool(inner.ball_x == inner.last_x)}


def _work(task):
    index, seed, max_steps, arrays, sticky, greedy = task
    return run_episode(index, seed, max_steps, arrays, sticky, greedy)[0]


def evaluate(arrays, episodes, workers, seed=0, max_steps=10000, sticky=0.1, greedy=False):
    tasks = [(i, seed, max_steps, arrays, sticky, greedy) for i in range(episodes)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_work, tasks, chunksize=5))
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    deaths = [r for r in rows if not r["truncated"]]
    return {"mean": float(scores.mean()), "se": float(scores.std(ddof=1) / math.sqrt(len(rows))),
            "median": float(np.median(scores)), "mean_length": float(np.mean([r["steps"] for r in rows])),
            "capped_share": float(np.mean([r["truncated"] for r in rows])),
            "deaths": len(deaths),
            "deaths_with_override_in_last_12_steps": int(sum(r["override_in_last_12"] for r in deaths)),
            "lengths": [r["steps"] for r in rows], "scores": scores.tolist()}


# ---------------------------------------------------------------- variants

def variants(arrays, noise_draws=3):
    out = {"intact": (arrays, {})}
    w2 = "actor/dense2/kernel"
    for unit in np.flatnonzero((arrays[w2] != 0).any(0)):
        # Silence, not zero: an idle comparison unit outputs -1 and that -30 is part of
        # its logit's baseline, so zeroing it would shift the baseline as well. Clamping
        # it to -1 means "this comparison never fires" and leaves everything else equal.
        a = {n: v.copy() for n, v in arrays.items()}
        a[w2][:, unit] = 0
        a["actor/dense2/bias"][unit] = -50
        out[f"silence-layer2-unit-{unit}"] = (a, {})
    rng = np.random.default_rng(20261116)
    for scale in (0.01, 0.03, 0.1, 0.3):
        for draw in range(noise_draws):
            a = {n: (v * (1 + scale * rng.standard_normal(v.shape))).astype(np.float32) for n, v in arrays.items()}
            out[f"noise-{scale}-draw-{draw}"] = (a, {})
    w1 = "actor/dense1/kernel"
    for step in (0.04, 0.06, 0.1):
        a = {n: v.copy() for n, v in arrays.items()}
        target = a[w1][:, :4]  # the landing-target code; units 4 and 5 use integers already
        a[w1][:, :4] = (np.round(target / step) * step).astype(np.float32)
        out[f"target-grid-{step}"] = (a, {})
    out["greedy-actions"] = (arrays, {"greedy": True})
    for sticky in (0.0, 0.025, 0.05, 0.2):
        out[f"sticky-{sticky}"] = (arrays, {"sticky": sticky})
    out["sticky-0.0-greedy"] = (arrays, {"sticky": 0.0, "greedy": True})
    return out


# ---------------------------------------------------------------- semantics

def semantics(arrays, episodes, seed=0, max_steps=10000):
    rows = []
    for index in range(episodes):
        rows.extend(run_episode(index, seed, max_steps, arrays, trace=True)[1])
    return summarize_semantics(rows), rows


def summarize_semantics(rows):
    def stats(subset):
        if not subset:
            return {"states": 0}
        err = np.array([min(abs(r["implied_target"] - c) for c in r["catch"]) for r in subset])
        inside = np.array([r["catch"][0] - 0.5 <= r["implied_target"] <= r["catch"][-1] + 0.5 for r in subset])
        return {"states": len(subset), "within_half_column_of_catch_set": float(inside.mean()),
                "mean_abs_error_columns": float(err.mean()), "median_abs_error_columns": float(np.median(err)),
                "share_error_over_1_column": float((err > 1).mean())}

    descending = [r for r in rows if r["dir"] in (2, 3)]
    ascending = [r for r in rows if r["dir"] in (0, 1)]
    by_row = {}
    for name, subset in (("descending", descending), ("ascending", ascending)):
        by_row[name] = {str(y): stats([r for r in subset if r["ball"][1] == y]) for y in range(10)
                        if any(r["ball"][1] == y for r in subset)}
    context_ok = np.array([r["context_true"] == r["context_chosen"] for r in rows])
    wall = np.array([r["wall_frame"] for r in rows])
    last_descent = [r for r in descending if r["steps_to_row9"] == 1]
    on_catch = np.array([r["paddle"] in r["catch"] for r in last_descent]) if last_descent else np.array([])
    return {"states": len(rows),
            "context": {"accuracy_all": float(context_ok.mean()),
                        "accuracy_wall_frames": float(context_ok[wall].mean()) if wall.any() else None,
                        "accuracy_other_frames": float(context_ok[~wall].mean()),
                        "wall_frames": int(wall.sum())},
            "target_vs_truth": {"descending": stats(descending), "ascending": stats(ascending)},
            "target_vs_truth_by_ball_row": by_row,
            "paddle_on_catch_column_one_step_before_row9": float(on_catch.mean()) if on_catch.size else None,
            "decisions_with_max_probability_below_0.99": float(np.mean([r["max_probability"] < 0.99 for r in rows]))}


# ---------------------------------------------------------------- figures

def receptive_fields(arrays, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w1 = arrays["actor/dense1/kernel"]
    titles = ["0: target, down-right", "1: target, down-left", "2: target, up-right", "3: target, up-left",
              "4: horizontal direction", "5: vertical direction"]
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.5), gridspec_kw={"height_ratios": [10, 10, 1.4]})
    for unit in range(6):
        for row, (channel, label) in enumerate(((BALL, "ball"), (TRAIL, "trail"))):
            grid = w1[[feature(r, c, channel) for r in range(10) for c in range(10)], unit].reshape(10, 10)
            if unit < 4 and channel == BALL:
                grid = grid / TARGET_SCALE
            limit = max(np.abs(grid).max(), 1e-9)
            ax = axes[row, unit]
            ax.imshow(grid, cmap="viridis" if unit < 4 and channel == BALL else "RdBu_r",
                      vmin=0 if unit < 4 and channel == BALL else -limit, vmax=limit)
            if unit < 4 and channel == BALL:
                for r in range(10):
                    for c in range(10):
                        ax.text(c, r, f"{grid[r, c]:.1f}".rstrip("0").rstrip("."), ha="center", va="center", fontsize=5.5, color="w")
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[unit], fontsize=9)
            if unit == 0:
                ax.set_ylabel(label + (" (target column)" if channel == BALL else ""), fontsize=8)
        paddle = w1[[feature(9, c, PADDLE) for c in range(10)], unit][None]
        axes[2, unit].imshow(paddle, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
        axes[2, unit].set_xticks([]), axes[2, unit].set_yticks([])
        if unit == 0:
            axes[2, unit].set_ylabel("paddle", fontsize=8, rotation=0, ha="right")
    fig.suptitle("First-layer receptive fields of the released Breakout network (bricks channel: all zero)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def survival_figure(results, path, max_steps=10000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    steps = np.linspace(0, max_steps, 201)
    for name in ("sticky-0.0", "sticky-0.025", "sticky-0.05", "intact", "sticky-0.2"):
        if name not in results:
            continue
        lengths = np.array(results[name]["lengths"])
        alive = [(lengths >= s).mean() for s in steps]
        label = "sticky 0.1 (published)" if name == "intact" else name.replace("-", " ")
        ax.plot(steps, alive, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("share of episodes still alive")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Survival of the released network by sticky-action probability")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["semantics", "evaluate", "figures", "all"])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--semantic-episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    arrays = load(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    source = breakout.sha(args.checkpoint)
    if args.command in ("semantics", "all"):
        summary, rows = semantics(arrays, args.semantic_episodes)
        summary.update({"episodes": args.semantic_episodes, "source_checkpoint_sha256": source})
        (args.output / "semantics.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in ("context", "target_vs_truth")}, indent=2), flush=True)
    if args.command in ("evaluate", "all"):
        results = {}
        for name, (a, options) in variants(arrays).items():
            results[name] = evaluate(a, args.episodes, args.workers, **options)
            results[name]["options"] = options
            shown = {k: v for k, v in results[name].items() if k not in ("lengths", "scores")}
            print(name, json.dumps(shown), flush=True)
        report = {"schema": "weight-space-breakout-anatomy-v1", "source_checkpoint_sha256": source,
                  "episodes": args.episodes, "max_steps": 10000, "seed": 0, "results": results}
        (args.output / "evaluations.json").write_text(json.dumps(report) + "\n")
    if args.command in ("figures", "all"):
        receptive_fields(arrays, args.output / "receptive-fields.png")
        evaluations = args.output / "evaluations.json"
        if evaluations.exists():
            survival_figure(json.loads(evaluations.read_text())["results"], args.output / "survival-by-sticky.png")


if __name__ == "__main__":
    main()
