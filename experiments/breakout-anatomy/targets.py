"""Where does the hand-written Breakout program lose? (roadmap #8)

The anatomy run (``anatomy.py``) found that the released network's *descending*
landing predictions are exact, while its *ascending* predictions miss the true landing
column by more than one column about half the time, because the network cannot see
bricks. Silencing the ascending comparisons still destroys play, so pre-positioning
during the climb matters even when it is wrong.

This script swaps only the ascending target code (first-layer units 2 and 3, ball
channel) for alternatives and evaluates each at the published protocol:

* ``published``: the released weights.
* ``follow-ball``: target the ball's current column.
* ``center``: always target column 4.5.
* ``no-brick-physics``: where the ball would land if there were no bricks.
* ``best-blind``: for each ball cell and direction, the column that minimizes the mean
  distance to the true catch columns in held-out play (episodes 5000-5049, disjoint from
  the evaluation seeds). The best any brick-blind lookup table of this shape can do on
  that data.
* ``oracle-ascending``: the network, with units 2 and 3 fed the true landing column
  computed from the full game state (bricks included). Not a network; an upper bound
  for perfect ascending prediction.
* ``oracle``: no network. Move toward the true catch columns every step.

All variants keep every other weight, the deadband and the gating unchanged.

    python experiments/breakout-anatomy/targets.py CHECKPOINT --output runs/breakout-targets
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("breakout_anatomy", HERE / "anatomy.py")
anatomy = importlib.util.module_from_spec(_spec)
sys.modules["breakout_anatomy"] = anatomy
_spec.loader.exec_module(anatomy)

W1 = "actor/dense1/kernel"
ASCENDING = {1: 2, 0: 3}  # MinAtar ball_dir -> first-layer unit (up-right, up-left)
FIT_EPISODES = range(5000, 5050)


def set_ascending_targets(arrays, table):
    """table[(row, col, dir)] -> target column; written into the ball weights of units 2/3."""
    out = {n: v.copy() for n, v in arrays.items()}
    for (row, col, direction), target in table.items():
        out[W1][anatomy.feature(row, col, anatomy.BALL), ASCENDING[direction]] = np.float32(target * anatomy.TARGET_SCALE)
    return out


def cells():
    return [(r, c, d) for r in range(10) for c in range(10) for d in ASCENDING]


def table_follow_ball():
    return {(r, c, d): float(c) for r, c, d in cells()}


def table_center():
    return {(r, c, d): 4.5 for r, c, d in cells()}


def table_no_brick():
    empty = np.zeros((10, 10), dtype=bool)
    out = {}
    for r, c, d in cells():
        if r == 9:
            continue
        catch = anatomy.ball_future(c, r, d, empty, False)["catch"]
        out[(r, c, d)] = float(np.mean(catch))
    return out


def fit_best_blind(arrays, episodes=FIT_EPISODES, max_steps=10000):
    """Choose, per ascending cell, the column closest on average to the true catch set."""
    seen = {}
    for index in episodes:
        _, rows = anatomy.run_episode(index, 0, max_steps, arrays, trace=True)
        for row in rows:
            if row["dir"] in ASCENDING:
                key = (row["ball"][1], row["ball"][0], row["dir"])
                seen.setdefault(key, []).append(row["catch"])
    grid = np.arange(0, 9.01, 0.5)
    table, counts = {}, {}
    for key, catches in seen.items():
        cost = [np.mean([min(abs(t - c) for c in catch) for catch in catches]) for t in grid]
        table[key] = float(grid[int(np.argmin(cost))])
        counts[key] = len(catches)
    return table, counts


def oracle_action(inner):
    catch = anatomy.env_future(inner)["catch"]
    if inner.pos < catch[0]:
        return 3
    if inner.pos > catch[-1]:
        return 1
    return 0


def run_episode(index, seed, max_steps, arrays, mode, sticky=0.1, events=None):
    from minatar import Environment
    env = Environment("breakout", sticky_action_prob=sticky)
    env.seed(seed + index)
    env.reset()
    recorder = anatomy.Recorder(env)
    rng = np.random.default_rng(seed + index + 1_000_000)
    w1, b1, w2, b2, w3, b3 = anatomy.as_list(arrays)
    score, terminal = 0, False
    since_move = 0  # steps since the policy last chose a move
    for step in range(1, max_steps + 1):
        inner = env.env
        if events is not None:
            at_catch = inner.ball_dir in (2, 3) and inner.ball_y == 8
            distance_before = min(abs(inner.pos - c) for c in anatomy.env_future(inner)["catch"]) if at_catch else None
        if mode == "oracle":
            action = oracle_action(inner)
        else:
            x = anatomy.breakout.observation(env.state())
            pre1 = x @ w1 + b1
            if mode == "oracle-ascending" and inner.ball_dir in ASCENDING:
                target = float(np.mean(anatomy.env_future(inner)["catch"]))
                pre1[ASCENDING[inner.ball_dir]] = anatomy.TARGET_SCALE * (target - inner.pos)
            logits = np.tanh(np.tanh(pre1) @ w2 + b2) @ w3 + b3
            p = np.exp(logits.astype(np.float64) - logits.max())
            p /= p.sum()
            action = int(rng.choice(6, p=p))
        since_move = 0 if action in (1, 3) else since_move + 1
        r, terminal = recorder.act(action)
        if events is not None and at_catch:
            # settle: steps the paddle had already been still when the ball reached the
            # paddle row (0 = the policy chose a move on this very step).
            events.append({"settle": since_move, "distance_before": distance_before,
                           "override": recorder.overrides[-1], "died": bool(terminal)})
        score += r
        if terminal:
            break
    return {"episode": index, "score": score, "steps": step, "truncated": not terminal}


def catch_profile(arrays, mode, episodes=range(7000, 7020)):
    """Per catch event: how long the paddle had been settled, and whether it died there."""
    events = []
    for index in episodes:
        run_episode(index, 0, 10000, arrays, mode, events=events)
    out = {"catch_events": len(events), "deaths": int(sum(e["died"] for e in events))}
    for label, test in (("0", lambda k: k == 0), ("1", lambda k: k == 1), ("2", lambda k: k == 2), ("3+", lambda k: k >= 3)):
        group = [e for e in events if test(e["settle"])]
        out[f"settle_{label}"] = {"share": len(group) / max(len(events), 1), "events": len(group),
                                  "deaths": int(sum(e["died"] for e in group))}
    return out


def _work(task):
    return run_episode(*task)


def evaluate(arrays, mode, episodes, workers, seed=0, max_steps=10000):
    tasks = [(i, seed, max_steps, arrays, mode) for i in range(episodes)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_work, tasks, chunksize=5))
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    return {"mean": float(scores.mean()), "se": float(scores.std(ddof=1) / math.sqrt(len(rows))),
            "mean_length": float(np.mean([r["steps"] for r in rows])),
            "capped_share": float(np.mean([r["truncated"] for r in rows])),
            "lengths": [r["steps"] for r in rows], "scores": scores.tolist()}


def target_error(arrays, table_arrays, episodes=range(6000, 6010)):
    """Mean distance from the ascending target to the true catch set, on fresh seeds,
    measured along the published network's own trajectories."""
    w = table_arrays[W1]
    errors = []
    for index in episodes:
        _, rows = anatomy.run_episode(index, 0, 10000, arrays, trace=True)
        for row in rows:
            if row["dir"] in ASCENDING:
                (c, r), d = row["ball"], row["dir"]
                t = w[anatomy.feature(r, c, anatomy.BALL), ASCENDING[d]] / anatomy.TARGET_SCALE
                errors.append(min(abs(t - k) for k in row["catch"]))
    errors = np.array(errors)
    return {"states": int(errors.size), "mean_abs_error_columns": float(errors.mean()),
            "share_error_over_1_column": float((errors > 1).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    arrays = anatomy.load(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    best, counts = fit_best_blind(arrays)
    tables = {"follow-ball": table_follow_ball(), "center": table_center(),
              "no-brick-physics": table_no_brick(), "best-blind": best}
    variants = {"published": (arrays, "network")}
    variants.update({name: (set_ascending_targets(arrays, t), "network") for name, t in tables.items()})
    variants["oracle-ascending"] = (arrays, "oracle-ascending")
    variants["oracle"] = (arrays, "oracle")
    results = {}
    for name, (a, mode) in variants.items():
        results[name] = evaluate(a, mode, args.episodes, args.workers)
        if mode == "network":
            results[name]["ascending_target_error"] = target_error(arrays, a)
        results[name]["catch_profile"] = catch_profile(a, mode)
        shown = {k: v for k, v in results[name].items() if k not in ("lengths", "scores")}
        print(name, json.dumps(shown), flush=True)
    report = {"schema": "weight-space-breakout-targets-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
              "episodes": args.episodes, "seed": 0, "max_steps": 10000, "fit_episodes": [FIT_EPISODES.start, FIT_EPISODES.stop],
              "best_blind_table": {f"{r},{c},{d}": [t, counts[(r, c, d)]] for (r, c, d), t in sorted(best.items())},
              "results": results}
    (args.output / "targets.json").write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
