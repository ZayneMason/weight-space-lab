"""Fit the ascending guesses to the under-catch column (roadmap #18)

#8's accurate table chose, for every rising-ball cell, the column closest on average to the
*catch set*: the two columns where the paddle survives (MinAtar checks ``ball_x == pos``,
then ``new_x == pos``). That treats the corner column as just as good as the under column.
#17 found that corner catches cost the accurate table bricks, and that they almost always
come from a paddle parked one column ahead of the ball. This script refits the table with
three definitions of "right":

- **nearest**: the nearest catch column (#8's fit, refitted here as a check);
- **under**: the column the ball is in before it reaches the paddle row;
- **corner**: the column it moves into.

Each table is written into the rising-ball weights of the published network and scored with
and without sticky actions.

    python experiments/breakout-anatomy/underfit.py CHECKPOINT --output runs/breakout-underfit
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("breakout_returns", HERE / "returns.py")
returns = importlib.util.module_from_spec(_spec)
sys.modules["breakout_returns"] = returns
_spec.loader.exec_module(returns)
rowswap, hedge, targets, anatomy = returns.rowswap, returns.hedge, returns.targets, returns.anatomy

TARGETS = ("nearest", "under", "corner")
GRID = np.arange(0, 9.01, 0.5)


def target_columns(future, kind):
    if kind == "nearest":
        return future["catch"]
    return [future[kind]]


def _trace(index):
    arrays = anatomy.load(_trace.checkpoint)
    _, rows = anatomy.run_episode(index, 0, 10000, arrays, trace=True)
    return [((r["ball"][1], r["ball"][0], r["dir"]), r["catch"], r["catch_under"], r["catch_corner"])
            for r in rows if r["dir"] in targets.ASCENDING]


def observe(checkpoint, episodes, workers):
    """Every rising-ball state on the published network's own play: its cell and catch columns."""
    _trace.checkpoint = checkpoint
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [row for part in pool.map(_trace, episodes) for row in part]


def fit(observations, kind):
    """Per cell, the grid column with the least mean distance to the chosen target."""
    seen = {}
    for cell, catch, under, corner in observations:
        future = {"catch": catch, "under": under, "corner": corner}
        seen.setdefault(cell, []).append(target_columns(future, kind))
    table = {}
    for cell, columns in seen.items():
        cost = [np.mean([min(abs(t - c) for c in cs) for cs in columns]) for t in GRID]
        table[cell] = float(GRID[int(np.argmin(cost))])
    return table


def errors(observations, tables):
    """Mean distance from each table's guess to the under column and to the nearest catch column."""
    out = {}
    for name, table in tables.items():
        rows = [(table[cell], catch, under) for cell, catch, under, _ in observations if cell in table]
        out[name] = {"states": len(rows),
                     "under_column_error": float(np.mean([abs(t - u) for t, _, u in rows])),
                     "nearest_catch_error": float(np.mean([min(abs(t - c) for c in catch) for t, catch, _ in rows]))}
    return out


def score(weights, episodes, workers, sticky=0.1, start=0):
    """Score, deaths, corner share and bricks per return, from #17's catch recorder."""
    if sticky == 0:
        return returns.summarize([returns.episode(0, weights, 0.0)])
    return returns.summarize(returns.collect(weights, episodes, sticky, workers, start=start))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--errors", action="store_true",
                        help="only measure each table's error on #8's held-out episodes (6000-6009), from underfit.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = anatomy.load(args.checkpoint)
    if args.errors:
        fitted = json.loads((args.output / "underfit.json").read_text())["tables"]
        tables = {"author": hedge.published_table(arrays)}
        tables.update({kind: {tuple(int(v) for v in key.split(",")): t for key, t in fitted[kind].items()} for kind in TARGETS})
        out = errors(observe(args.checkpoint, range(6000, 6010), args.workers), tables)
        (args.output / "errors.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out))
        return
    observations = observe(args.checkpoint, targets.FIT_EPISODES, args.workers)
    published = hedge.published_table(arrays)
    tables = {kind: fit(observations, kind) for kind in TARGETS}
    accurate = rowswap.accurate_table()
    report = {"schema": "weight-space-breakout-underfit-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
              "fit_episodes": [targets.FIT_EPISODES.start, targets.FIT_EPISODES.stop], "fit_states": len(observations),
              "episodes": args.episodes,
              "nearest_matches_accurate_cells": int(sum(abs(tables["nearest"].get(k, -1) - v) < 1e-9 for k, v in accurate.items())),
              "accurate_cells": len(accurate),
              "tables": {kind: {f"{r},{c},{d}": t for (r, c, d), t in sorted(table.items())} for kind, table in tables.items()},
              "results": {}}
    variants = {"published": arrays}
    variants.update({kind: targets.set_ascending_targets(arrays, rowswap.mix(published, table, set(range(10))))
                     for kind, table in tables.items()})
    for name, weights in variants.items():
        report["results"][name] = {"sticky_0": score(weights, 1, 1, sticky=0.0),
                                   "sticky_0.1": score(weights, args.episodes, args.workers)}
        print(name, json.dumps({k: {m: v[m] for m in ("mean_score", "deaths_per_10k_steps", "bricks_per_return")}
                                for k, v in report["results"][name].items()}), flush=True)
        (args.output / "underfit.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
