"""Why do the author's rough guesses beat accurate ones? (roadmap #9)

Roadmap #8 found that a brick-blind lookup table with lower mean error than the
released network's ascending guesses scores far worse, and that nearly every death is
a last-moment paddle arrival. Hypothesis: the author's guesses assume the *earliest*
possible brick bounce, the case with the least time left, so they hedge the deadline
rather than the average.

This script (1) reconstructs the "earliest bounce" table from game physics alone (a
full wall of bricks in rows 1-3, as at the start of every episode) and compares it with
the published weights, and (2) fits a *deadline-aware* brick-blind table that minimises
how late the paddle could arrive rather than how far off it is on average, then
evaluates both at the published protocol.

    python experiments/breakout-anatomy/hedge.py CHECKPOINT --output runs/breakout-hedge
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("breakout_targets", HERE / "targets.py")
targets = importlib.util.module_from_spec(_spec)
sys.modules["breakout_targets"] = targets
_spec.loader.exec_module(targets)
anatomy = targets.anatomy

FIT_EPISODES = range(5000, 5050)
LATE_STEPS = 2  # a catch that needs a move in the last LATE_STEPS steps counts as late


def descent(x, y, direction, bricks, strike):
    """Steps until the ball reaches the paddle row, steps of that spent descending,
    and the catch columns."""
    total = anatomy.ball_future(x, y, direction, bricks, strike)
    # Replay the same rules until the ball first heads down.
    bricks = np.array(bricks, dtype=bool, copy=True)
    steps_up = 0
    while direction not in (2, 3):
        last_x, last_y = x, y
        dx, dy = anatomy.MOVES[direction]
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
        if not toggle:
            strike = False
        x, y = new_x, new_y
        steps_up += 1
    return {"steps": total["steps"], "descending_steps": total["steps"] - steps_up, "catch": total["catch"]}


def full_wall_catches():
    """Catch columns for every ascending cell if rows 1-3 were a full wall of bricks."""
    wall = np.zeros((10, 10), dtype=bool)
    wall[1:4, :] = True
    return {(r, c, d): anatomy.ball_future(c, r, d, wall, False)["catch"] for r, c, d in targets.cells() if r != 9}


def table_full_wall():
    return {k: float(np.mean(v)) for k, v in full_wall_catches().items()}


def on_catch_columns(table, catches, keys):
    """Share of cells whose target lies within half a column of a catch column."""
    hits = [min(abs(table[k] - c) for c in catches[k]) <= .5 for k in keys]
    return {"cells": len(hits), "on_a_catch_column": float(np.mean(hits))}


def published_table(arrays):
    w = arrays[targets.W1]
    return {(r, c, d): float(w[anatomy.feature(r, c, anatomy.BALL), targets.ASCENDING[d]]) / anatomy.TARGET_SCALE
            for r, c, d in targets.cells()}


def compare_tables(a, b, keys):
    diff = np.array([abs(a[k] - b[k]) for k in keys])
    return {"cells": int(diff.size), "within_half_column": float((diff <= .5).mean()), "mean_abs_difference": float(diff.mean())}


def ascent_observations(index, arrays, seed=0, max_steps=10000):
    """For every ascending state of the published network's play: catch set and how many
    steps the ball will spend descending."""
    from minatar import Environment
    env = Environment("breakout", sticky_action_prob=0.1)
    env.seed(seed + index)
    env.reset()
    rng = np.random.default_rng(seed + index + 1_000_000)
    weights = anatomy.as_list(arrays)
    out = {}
    for _ in range(max_steps):
        inner = env.env
        if inner.ball_dir in targets.ASCENDING:
            info = descent(inner.ball_x, inner.ball_y, inner.ball_dir, inner.brick_map == 1, inner.strike)
            out.setdefault((inner.ball_y, inner.ball_x, inner.ball_dir), []).append((info["catch"], info["descending_steps"]))
        x = anatomy.breakout.observation(env.state())
        logits = anatomy.breakout.policy_logits(x, weights)
        p = np.exp(logits.astype(np.float64) - logits.max())
        p /= p.sum()
        _, terminal = env.act(int(rng.choice(6, p=p)))
        if terminal:
            break
    return out


def fit_deadline_aware(seen):
    """Per cell, the target that minimises expected lateness: how many moves beyond
    (descending steps - LATE_STEPS) the paddle would still need when the ball turns
    down. Mean distance breaks ties."""
    grid = np.arange(0, 9.01, 0.5)
    table = {}
    for key, observations in seen.items():
        def cost(t):
            late = [max(0.0, min(abs(t - c) for c in catch) - (steps - LATE_STEPS)) for catch, steps in observations]
            dist = [min(abs(t - c) for c in catch) for catch, _ in observations]
            return np.mean(late) + 0.01 * np.mean(dist)
        table[key] = float(grid[int(np.argmin([cost(t) for t in grid]))])
    return table


def fit_most_likely(seen):
    """Per cell, the target that most often lands within half a column of a true catch
    column (an exact hit). Mean distance breaks ties."""
    grid = np.arange(0, 9.01, 0.5)
    table = {}
    for key, observations in seen.items():
        def score(t):
            dist = [min(abs(t - c) for c in catch) for catch, _ in observations]
            return (np.mean([d <= .5 for d in dist]), -np.mean(dist))
        table[key] = float(grid[max(range(len(grid)), key=lambda i: score(grid[i]))])
    return table


def observed_fit(table, seen):
    """Over every observed ascending state: mean distance to the nearest catch column,
    share of exact hits, and share where the paddle would need more moves than the
    descent leaves (margins 1-4 steps)."""
    dist, steps = [], []
    for key, observations in seen.items():
        for catch, descending in observations:
            dist.append(min(abs(table[key] - c) for c in catch))
            steps.append(descending)
    dist, steps = np.array(dist), np.array(steps)
    return {"states": int(dist.size), "mean_distance": float(dist.mean()), "exact_hit": float((dist <= .5).mean()),
            "late_by_margin": {str(m): float((dist > steps - m).mean()) for m in (1, 2, 3, 4)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    arrays = anatomy.load(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    published = published_table(arrays)
    wall = table_full_wall()
    seen = {}
    for index in FIT_EPISODES:
        for key, obs in ascent_observations(index, arrays).items():
            seen.setdefault(key, []).extend(obs)
    deadline = fit_deadline_aware(seen)
    likely = fit_most_likely(seen)
    keys_wall = sorted(wall)
    keys_seen = sorted(seen)
    catches = full_wall_catches()
    visited_rows_1_to_8 = [k for k in keys_seen if k in catches and k[0] >= 4]
    comparison = {
        "published_on_full_wall_catch_columns_all_cells": on_catch_columns(published, catches, keys_wall),
        "published_on_full_wall_catch_columns_rows_4_to_8": on_catch_columns(published, catches, [k for k in keys_wall if k[0] >= 4]),
        "published_on_full_wall_catch_columns_visited_rows_4_to_8": on_catch_columns(published, catches, visited_rows_1_to_8),
        "published_vs_full_wall_all_cells": compare_tables(published, wall, keys_wall),
        "published_vs_deadline_visited_cells": compare_tables(published, deadline, keys_seen),
        "observed_fit": {name: observed_fit(t, seen) for name, t in
                         (("published", published), ("full-wall", wall), ("deadline-aware", deadline), ("most-likely", likely))},
        "visited_cells": len(keys_seen),
        "ascending_observations": int(sum(len(v) for v in seen.values())),
    }
    print(json.dumps(comparison, indent=1), flush=True)
    variants = {"published": arrays,
                "full-wall": targets.set_ascending_targets(arrays, wall),
                "deadline-aware": targets.set_ascending_targets(arrays, deadline),
                "most-likely": targets.set_ascending_targets(arrays, likely)}
    results = {}
    for name, a in variants.items():
        results[name] = targets.evaluate(a, "network", args.episodes, args.workers)
        results[name]["ascending_target_error"] = targets.target_error(arrays, a)
        results[name]["catch_profile"] = targets.catch_profile(a, "network")
        shown = {k: v for k, v in results[name].items() if k not in ("lengths", "scores")}
        print(name, json.dumps(shown), flush=True)
    report = {"schema": "weight-space-breakout-hedge-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
              "episodes": args.episodes, "fit_episodes": [FIT_EPISODES.start, FIT_EPISODES.stop], "late_steps": LATE_STEPS,
              "comparison": comparison,
              "deadline_table": {f"{r},{c},{d}": t for (r, c, d), t in sorted(deadline.items())},
              "most_likely_table": {f"{r},{c},{d}": t for (r, c, d), t in sorted(likely.items())},
              "results": results}
    (args.output / "hedge.json").write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
