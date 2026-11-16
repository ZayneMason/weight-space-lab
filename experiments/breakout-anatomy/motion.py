"""Does paddle motion explain the row-3 deaths? (roadmap #15)

Roadmap #13 found that on flights turning down at row 3 the accurate table starts the
paddle closer to the catch than the author's table (0.76 against 0.94 columns) yet dies
3.4 times as often, with five or more steps of slack. Distance at the turn cannot explain
that. This script follows each falling flight step by step and records:

- the paddle's motion at the turn: toward the catch, away from it, or not moving;
Only flights that reach the paddle row count (a falling ball can hit a brick and turn back up).

- whether the paddle was already on a catch column one step before the ball arrived;
- whether the policy chose a move on the final step ("unsettled"): a sticky override of
  the previous move on that step can carry the paddle off the catch;
- whether a sticky override changed the executed move in the last two steps.

Published table and #8's accurate table on the same 300 episodes as #13.

    python experiments/breakout-anatomy/motion.py CHECKPOINT --output runs/breakout-motion
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("breakout_deaths", HERE / "deaths.py")
deaths = importlib.util.module_from_spec(_spec)
sys.modules["breakout_deaths"] = deaths
_spec.loader.exec_module(deaths)
rowswap, hedge, targets, anatomy = deaths.rowswap, deaths.hedge, deaths.targets, deaths.anatomy

DESCENDING = deaths.DESCENDING
MOVE = {"l": -1, "r": 1}


def distance(pos, catch):
    return int(min(abs(pos - c) for c in catch))


def turn_motion(previous_pos, pos, catch):
    """Motion over the step that ended at the turn, relative to this flight's catch."""
    before, after = distance(previous_pos, catch), distance(pos, catch)
    if after < before:
        return "toward"
    return "away" if after > before else "still"  # "still" includes moving between two catch columns


def flights(index, arrays, seed=0, max_steps=10000, sticky=0.1):
    from minatar import Environment
    env = Environment("breakout", sticky_action_prob=sticky)
    env.seed(seed + index)
    env.reset()
    rng = np.random.default_rng(seed + index + 1_000_000)
    weights = anatomy.as_list(arrays)
    out, flight, previous_pos = [], None, env.env.pos
    was_descending = env.env.ball_dir in DESCENDING
    for _ in range(max_steps):
        inner = env.env
        descending = inner.ball_dir in DESCENDING
        if descending and not was_descending:
            future = anatomy.env_future(inner)
            flight = {"row": int(inner.ball_y), "steps_left": int(future["steps"]), "catch": future["catch"],
                      "distance": distance(inner.pos, future["catch"]),
                      "turn_motion": turn_motion(previous_pos, inner.pos, future["catch"]),
                      "steps": [], "died": False}
        if not descending and was_descending and flight is not None:
            out.append(finish(flight))
            flight = None
        was_descending = descending
        x = anatomy.breakout.observation(env.state())
        logits = anatomy.breakout.policy_logits(x, weights)
        p = np.exp(logits.astype(np.float64) - logits.max())
        p /= p.sum()
        chosen = int(rng.choice(6, p=p))
        # MinAtar's sticky wrapper, written out so overrides are visible (same RNG draws).
        override = env.random.rand() < env.sticky_action_prob
        executed = env.last_action if override else chosen
        env.last_action = executed
        previous_pos = inner.pos
        if flight is not None:
            flight["steps"].append({"pos": int(inner.pos),
                                    "chosen_moves": inner.action_map[chosen] in MOVE,
                                    "override_changed": bool(override and executed != chosen)})
        _, terminal = inner.act(executed)
        if terminal:
            if flight is not None:
                flight["died"] = True
                out.append(finish(flight))
            break
    return out


def finish(flight):
    """Reduce a flight's step log to the per-flight features."""
    steps, catch = flight.pop("steps"), flight.pop("catch")
    reached = len(steps) == flight["steps_left"]  # not turned back up by a brick on the way down
    if not steps or not reached:
        return dict(flight, reached=False, settled=None, unsettled_final=None, late_override=None)
    final = steps[-1]
    return dict(flight, reached=True,
                settled=distance(final["pos"], catch) == 0,  # on a catch column before the final move
                unsettled_final=final["chosen_moves"],
                late_override=any(s["override_changed"] for s in steps[-2:]))


def _work(task):
    return flights(*task)


def collect(arrays, episodes, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = pool.map(_work, [(i, arrays) for i in range(episodes)], chunksize=5)
        return [f for part in parts for f in part]


def rate(group):
    return {"flights": len(group), "deaths": int(sum(r["died"] for r in group)),
            "deaths_per_1000": 1000 * sum(r["died"] for r in group) / len(group) if group else None}


def summarize(records):
    out = {}
    for row in sorted({r["row"] for r in records}):
        group = [r for r in records if r["row"] == row and r["reached"]]
        dead = [r for r in group if r["died"]]
        out[str(row)] = {
            **rate(group),
            "turn_motion_share": {m: float(np.mean([r["turn_motion"] == m for r in group])) for m in ("toward", "still", "away")},
            "by_turn_motion": {m: rate([r for r in group if r["turn_motion"] == m]) for m in ("toward", "still", "away")},
            "settled_share": float(np.mean([r["settled"] for r in group])),
            "unsettled_final_share": float(np.mean([r["unsettled_final"] for r in group])),
            "settled_but_unsettled_share": float(np.mean([r["settled"] and r["unsettled_final"] for r in group])),
            "mean_distance_by_turn_motion": {m: (float(np.mean([r["distance"] for r in group if r["turn_motion"] == m]))
                                                 if any(r["turn_motion"] == m for r in group) else None)
                                             for m in ("toward", "still", "away")},
            "deaths_settled_share": float(np.mean([r["settled"] for r in dead])) if dead else None,
            "deaths_out_of_reach_share": float(np.mean([r["steps_left"] < r["distance"] for r in dead])) if dead else None,
            "deaths_late_override_share": float(np.mean([r["late_override"] for r in dead])) if dead else None}
    return out


def figure(report, path):
    """Left: paddle motion at the turn by row. Right: deaths per 1,000 by motion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    motions, rows, names = ("toward", "still", "away"), ("2", "3", "4"), ("published", "accurate")
    colors = {"toward": "#4C78A8", "still": "#9C9C9C", "away": "#E45756"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels, x = [], 0
    for row in rows:
        for name in names:
            stats = report["tables"][name]["by_turn_row"][row]
            bottom = 0
            for m in motions:
                share = stats["turn_motion_share"][m]
                axes[0].bar(x, share, bottom=bottom, color=colors[m], label=m if x == 0 else None)
                bottom += share
            labels.append(f"row {row}\n{name}")
            x += 1
    axes[0].set_xticks(range(len(labels)), labels, fontsize=8)
    axes[0].set_ylabel("share of flights")
    axes[0].set_title("Paddle motion when the ball turns down")
    axes[0].legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    width = 0.13
    for i, name in enumerate(names):
        for j, m in enumerate(motions):
            values = [report["tables"][name]["by_turn_row"][row]["by_turn_motion"][m]["deaths_per_1000"] or 0 for row in rows]
            offset = (i * 3 + j - 2.5) * width
            axes[1].bar(np.arange(len(rows)) + offset, values, width, color=colors[m],
                        hatch="//" if name == "accurate" else None, edgecolor="white")
    axes[1].set_xticks(range(len(rows)), [f"turned down at row {r}" for r in rows])
    axes[1].set_ylabel("deaths per 1,000 flights")
    axes[1].set_title("Death rate by motion (plain: published, hatched: accurate)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--figure", type=Path, help="draw the figure from an existing motion.json in --output")
    args = parser.parse_args()
    if args.figure:
        figure(json.loads((args.output / "motion.json").read_text()), args.figure)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = anatomy.load(args.checkpoint)
    published = hedge.published_table(arrays)
    tables = {"published": published, "accurate": rowswap.mix(published, rowswap.accurate_table(), set(range(10)))}
    report = {"schema": "weight-space-breakout-motion-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
              "episodes": args.episodes, "tables": {}}
    for name, table in tables.items():
        records = collect(targets.set_ascending_targets(arrays, table), args.episodes, args.workers)
        report["tables"][name] = {"flights": len(records), "deaths": int(sum(r["died"] for r in records)),
                                  "by_turn_row": summarize(records)}
        print(name, json.dumps(report["tables"][name]), flush=True)
    (args.output / "motion.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
