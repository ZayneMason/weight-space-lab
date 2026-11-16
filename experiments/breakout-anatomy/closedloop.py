"""Is the accuracy gap a distribution-shift effect? (roadmap #11)

Roadmap #9 found that brick-blind ascending tables fitted on the published network's own
play score far below the author's earliest-bounce table, even though they are more
accurate on that play. One explanation: a fitted table is fitted to states that the
published network visits, and installing it changes which states occur.

This script refits each table on its *own* play and repeats. Three starting tables:
the author's (published), the physics-only full-wall table, and #8's best mean-distance
table. Each round plays the fit episodes with the current table, refits every visited
cell to minimise mean distance to the true catch columns (unvisited cells keep their
value), and scores the result on a quick evaluation. The last round of each chain is
scored at the full protocol.

If refitting on its own play closes the gap, the gap was distribution shift. If
refitting the author's table on its own play makes it worse, fitting for accuracy is
itself the problem.

    python experiments/breakout-anatomy/closedloop.py CHECKPOINT --output runs/breakout-closedloop
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("breakout_hedge", HERE / "hedge.py")
hedge = importlib.util.module_from_spec(_spec)
sys.modules["breakout_hedge"] = hedge
_spec.loader.exec_module(hedge)
targets = hedge.targets
anatomy = hedge.anatomy

GRID = np.arange(0, 9.01, 0.5)


def fit_mean_distance(seen, previous):
    """Per visited cell, the grid target with the smallest mean distance to the nearest
    catch column; cells not visited keep their previous value."""
    table = dict(previous)
    for key, observations in seen.items():
        cost = [np.mean([min(abs(t - c) for c in catch) for catch, _ in observations]) for t in GRID]
        table[key] = float(GRID[int(np.argmin(cost))])
    return table


def _observe(task):
    index, arrays = task
    return hedge.ascent_observations(index, arrays)


def observe(arrays, episodes, workers):
    seen = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_observe, [(i, arrays) for i in episodes]):
            for key, obs in part.items():
                seen.setdefault(key, []).extend(obs)
    return seen


def chain(start, arrays, rounds, fit_episodes, quick_episodes, workers, log):
    """Refit `rounds` times on the table's own play; returns per-round records and the last table."""
    table, records = dict(start), []
    for round_index in range(rounds + 1):
        installed = targets.set_ascending_targets(arrays, table)
        seen = observe(installed, fit_episodes, workers)
        fit = hedge.observed_fit(table, seen)
        quick = targets.evaluate(installed, "network", quick_episodes, workers)
        record = {"round": round_index, "quick_mean": quick["mean"], "quick_se": quick["se"],
                  "own_play_fit": fit, "visited_cells": len(seen)}
        if records:
            record["cells_changed"] = int(sum(abs(table[k] - records[-1]["_table"][k]) > 1e-9 for k in table))
        record["_table"] = dict(table)
        records.append(record)
        log({k: v for k, v in record.items() if k != "_table"})
        if round_index < rounds:
            table = fit_mean_distance(seen, table)
    for record in records:
        record["table"] = {f"{r},{c},{d}": t for (r, c, d), t in sorted(record.pop("_table").items())}
    return records, table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--quick-episodes", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = anatomy.load(args.checkpoint)
    fit_episodes = hedge.FIT_EPISODES
    published = hedge.published_table(arrays)
    starts = {"published": published, "full-wall": hedge.table_full_wall()}
    # #8's best mean-distance table is the first refit of the published table on published play.
    starts["best-mean-distance"] = fit_mean_distance(observe(arrays, fit_episodes, args.workers), published)
    report = {"schema": "weight-space-breakout-closedloop-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
              "fit_episodes": [fit_episodes.start, fit_episodes.stop], "quick_episodes": args.quick_episodes,
              "episodes": args.episodes, "rounds": args.rounds, "chains": {}}
    for name, start in starts.items():
        records, final = chain(start, arrays, args.rounds, fit_episodes, args.quick_episodes, args.workers,
                               lambda r, name=name: print(name, json.dumps(r), flush=True))
        full = targets.evaluate(targets.set_ascending_targets(arrays, final), "network", args.episodes, args.workers)
        full.pop("lengths")
        print(name, "final", json.dumps({k: v for k, v in full.items() if k != "scores"}), flush=True)
        report["chains"][name] = {"rounds": records, "final_full_protocol": full}
        (args.output / "closedloop.json").write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
