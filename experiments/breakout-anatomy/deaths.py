"""What kills the accurate table? (roadmap #13)

Roadmap #12 found that the author's ascending guesses win in ball rows 3-5, and inferred
why: a rising ball there may bounce off the wall at once, leaving the least time, and the
author's guesses bet on that bounce. This script tests the mechanism directly.

For every flight (the ball turning from rising to falling until it is caught or lost) it
records the row where the ball turned down, how many steps it then had before reaching the
paddle row, how far the paddle was from the nearest catch column at that moment, and
whether the flight ended in a death. It compares the published table with #8's accurate
table on the same 300 episodes.

    python experiments/breakout-anatomy/deaths.py CHECKPOINT --output runs/breakout-deaths
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

DESCENDING = (2, 3)


def flights(index, arrays, seed=0, max_steps=10000, sticky=0.1):
    """One record per falling flight: turn-down row, steps left, paddle distance, died."""
    from minatar import Environment
    env = Environment("breakout", sticky_action_prob=sticky)
    env.seed(seed + index)
    env.reset()
    rng = np.random.default_rng(seed + index + 1_000_000)
    weights = anatomy.as_list(arrays)
    out, open_flight, was_descending = [], None, env.env.ball_dir in DESCENDING
    for _ in range(max_steps):
        inner = env.env
        descending = inner.ball_dir in DESCENDING
        if descending and not was_descending:
            future = anatomy.env_future(inner)
            open_flight = {"row": int(inner.ball_y), "steps_left": int(future["steps"]),
                           "distance": int(min(abs(inner.pos - c) for c in future["catch"])), "died": False}
        if not descending and was_descending and open_flight is not None:
            out.append(open_flight)
            open_flight = None
        was_descending = descending
        x = anatomy.breakout.observation(env.state())
        logits = anatomy.breakout.policy_logits(x, weights)
        p = np.exp(logits.astype(np.float64) - logits.max())
        p /= p.sum()
        _, terminal = env.act(int(rng.choice(6, p=p)))
        if terminal:
            if open_flight is not None:
                open_flight["died"] = True
                out.append(open_flight)
            break
    return out


def _work(task):
    return flights(*task)


def collect(arrays, episodes, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = pool.map(_work, [(i, arrays) for i in range(episodes)], chunksize=5)
        return [f for part in parts for f in part]


def summarize(records):
    """Per turn-down row: flights, deaths, death rate per 1,000 flights, mean distance and
    mean slack (steps left minus distance) at the turn."""
    out = {}
    for row in sorted({r["row"] for r in records}):
        group = [r for r in records if r["row"] == row]
        deaths = sum(r["died"] for r in group)
        out[str(row)] = {"flights": len(group), "deaths": int(deaths),
                         "deaths_per_1000": 1000 * deaths / len(group),
                         "mean_distance": float(np.mean([r["distance"] for r in group])),
                         "mean_slack": float(np.mean([r["steps_left"] - r["distance"] for r in group])),
                         "no_slack_share": float(np.mean([r["steps_left"] - r["distance"] <= 0 for r in group]))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = anatomy.load(args.checkpoint)
    published = hedge.published_table(arrays)
    tables = {"published": published, "accurate": rowswap.mix(published, rowswap.accurate_table(), set(range(10)))}
    report = {"schema": "weight-space-breakout-deaths-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
              "episodes": args.episodes, "tables": {}}
    for name, table in tables.items():
        records = collect(targets.set_ascending_targets(arrays, table), args.episodes, args.workers)
        summary = summarize(records)
        report["tables"][name] = {"flights": len(records), "deaths": int(sum(r["died"] for r in records)), "by_turn_row": summary}
        print(name, json.dumps(report["tables"][name]), flush=True)
    (args.output / "deaths.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
