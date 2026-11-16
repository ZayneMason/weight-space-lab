"""Does the catch column set the return and the brick rate? (roadmap #17)

Roadmap #16 found that without sticky actions nothing dies, yet the accurate ascending table
scores 569 against the author's 834 in 10,000 steps: fewer of its returns break a brick.
MinAtar Breakout has two ways to catch a falling ball:

- **under**: the paddle is under the ball's current column; the ball goes back up with the
  same horizontal direction;
- **corner**: the paddle is under the column the ball is about to enter; the ball goes back
  up with its horizontal direction reversed.

A table that parks the paddle a column differently changes which return the ball gets. This
script records, for every catch, the catch type and how many bricks the ball breaks before
the next catch, for the published and accurate tables, without sticky actions and at the
published setting.

    python experiments/breakout-anatomy/returns.py CHECKPOINT --output runs/breakout-returns
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
UNDER_FROM_CORNER = {0: 1, 1: 0}  # rising left <-> rising right


def catch_type(ball_x, pos):
    """MinAtar checks ``ball_x == pos`` first, then the column the ball moves into."""
    return "under" if ball_x == pos else "corner"


def episode(index, arrays, sticky=0.1, max_steps=10000, under_only=False):
    """One episode: a record per catch (its type and the bricks broken before the next catch).

    ``under_only`` is the counterfactual: every corner catch sends the ball back up with its
    horizontal direction kept, as an under catch would. The paddle and the policy are unchanged.
    """
    from minatar import Environment
    env = Environment("breakout", sticky_action_prob=sticky)
    env.seed(index)
    env.reset()
    rng = np.random.default_rng(index + 1_000_000)
    weights = anatomy.as_list(arrays)
    out, current, score, terminal = [], None, 0, False
    for step in range(1, max_steps + 1):
        inner = env.env
        arriving = inner.ball_dir in DESCENDING and inner.ball_y == 8
        ball_x = inner.ball_x
        x = anatomy.breakout.observation(env.state())
        logits = anatomy.breakout.policy_logits(x, weights)
        p = np.exp(logits.astype(np.float64) - logits.max())
        p /= p.sum()
        reward, terminal = env.act(int(rng.choice(6, p=p)))
        score += int(reward)
        if current is not None:
            current["bricks"] += int(reward)
            current["top"] = min(current["top"], int(env.env.ball_y))
        if terminal:
            if current is not None:
                current["died"] = True
                out.append(current)
            break
        if arriving and env.env.ball_dir not in DESCENDING:
            if current is not None:
                out.append(current)
            kind = catch_type(ball_x, env.env.pos)
            if kind == "corner" and under_only:
                env.env.ball_dir = UNDER_FROM_CORNER[env.env.ball_dir]
            current = {"type": kind, "column": int(ball_x), "step": step, "bricks": 0, "top": 9, "died": False}
    return {"episode": index, "score": score, "steps": step, "died": bool(terminal), "records": out}


def returns(index, arrays, sticky=0.1, max_steps=10000, under_only=False):
    return episode(index, arrays, sticky, max_steps, under_only)["records"]


def _work(task):
    return episode(*task)


def collect(arrays, episodes, sticky, workers, under_only=False):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_work, [(i, arrays, sticky, 10000, under_only) for i in range(episodes)], chunksize=2))


def since_corner(records):
    """Returns since the last corner catch, per record (None before the first corner)."""
    out, count = [], None
    for r in records:
        count = 0 if r["type"] == "corner" else (None if count is None else count + 1)
        out.append(count)
    return out


SINCE_BINS = ((0, 0), (1, 5), (6, 20), (21, 10**9))


def summarize(episodes):
    records = [r for e in episodes for r in e["records"]]
    steps = sum(e["steps"] for e in episodes)
    deaths = sum(e["died"] for e in episodes)
    out = {"episodes": len(episodes), "mean_score": float(np.mean([e["score"] for e in episodes])),
           "steps": steps, "deaths": deaths, "deaths_per_10k_steps": 1e4 * deaths / steps,
           "steps_per_return": steps / max(len(records), 1),
           "returns": len(records), "bricks_per_return": float(np.mean([r["bricks"] for r in records])),
           "no_brick_share": float(np.mean([r["bricks"] == 0 for r in records]))}
    for kind in ("under", "corner"):
        group = [r for r in records if r["type"] == kind]
        out[kind] = {"share": len(group) / len(records),
                     "bricks_per_return": float(np.mean([r["bricks"] for r in group])) if group else None,
                     "no_brick_share": float(np.mean([r["bricks"] == 0 for r in group])) if group else None}
    # the row where each return turned back down: 0 is the top wall, 2-4 under a brick
    out["by_turn_row"] = {}
    for row in sorted({r["top"] for r in records}):
        group = [r for r in records if r["top"] == row]
        out["by_turn_row"][str(row)] = {"share": len(group) / len(records),
                                        "deaths_per_1000": 1000 * float(np.mean([r["died"] for r in group]))}
    pairs = [(r, s) for e in episodes for r, s in zip(e["records"], since_corner(e["records"]))]
    out["by_returns_since_corner"] = {}
    for label, test in [("before first", lambda s: s is None)] + [
            (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+", lambda s, lo=lo, hi=hi: s is not None and lo <= s <= hi)
            for lo, hi in SINCE_BINS]:
        group = [r["bricks"] for r, s in pairs if test(s)]
        out["by_returns_since_corner"][label] = {"returns": len(group),
                                                 "bricks_per_return": float(np.mean(group)) if group else None}
    return out


def cycle(records, skip=200, window=200, longest=200):
    """Catch columns, no-brick returns by column, and the period of the catch-column sequence."""
    columns = [r["column"] for r in records]
    period = next((p for p in range(1, longest)
                   if all(columns[i] == columns[i + p] for i in range(skip, skip + window))), None)
    count = lambda rs: {str(c): sum(r["column"] == c for r in rs) for c in range(10)}
    return {"returns": len(records), "period_in_returns": period, "catches_by_column": count(records),
            "no_brick_by_column": count([r for r in records if r["bricks"] == 0]),
            "first_corner_step": next((r["step"] for r in records if r["type"] == "corner"), None)}


def figure(report, path):
    """Left: bricks per return by returns since the last corner catch. Right: where returns turn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    results = report["results"]
    width = 0.8 / len(results)
    for k, (name, result) in enumerate(results.items()):
        bins = result["sticky_0.1"]["by_returns_since_corner"]
        labels = list(bins)
        axes[0].bar([i + k * width for i in range(len(labels))],
                    [bins[b]["bricks_per_return"] or 0 for b in labels], width, label=name)
        rows = result["sticky_0.1"]["by_turn_row"]
        axes[1].bar([int(r) + k * width for r in rows], [rows[r]["share"] for r in rows], width, label=name)
    axes[0].set_xticks([i + 0.4 - width / 2 for i in range(len(labels))], labels)
    axes[0].set_xlabel("returns since the last corner catch")
    axes[0].set_title("Bricks per return (sticky 0.1)")
    axes[1].set_xlabel("row where the return turned down (0 = top wall)")
    axes[1].set_title("Share of returns (sticky 0.1)")
    axes[1].set_xticks([r + 0.4 - width / 2 for r in (0, 2, 3, 4)], ["0", "2", "3", "4"])
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=9, frameon=False)
    for ax in axes:
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--figure", type=Path, help="draw the figure from an existing returns.json in --output")
    parser.add_argument("--cycle", action="store_true", help="without sticky actions, describe each table's catch cycle")
    args = parser.parse_args()
    if args.figure:
        figure(json.loads((args.output / "returns.json").read_text()), args.figure)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = anatomy.load(args.checkpoint)
    published = hedge.published_table(arrays)
    tables = {"published": arrays,
              "accurate": targets.set_ascending_targets(arrays, rowswap.mix(published, rowswap.accurate_table(), set(range(10))))}
    report = {"schema": "weight-space-breakout-returns-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
              "episodes": args.episodes, "results": {}}
    if args.cycle:
        out = {name: cycle(returns(0, weights, sticky=0.0)) for name, weights in tables.items()}
        (args.output / "cycle.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out))
        return
    runs = {"published": (tables["published"], False), "accurate": (tables["accurate"], False),
            "accurate, corners returned as under": (tables["accurate"], True)}
    for name, (weights, under_only) in runs.items():
        report["results"][name] = {
            # without sticky actions every episode follows one cycle, so one episode is enough
            "sticky_0": summarize([episode(0, weights, 0.0, 10000, under_only)]),
            "sticky_0.1": summarize(collect(weights, args.episodes, 0.1, args.workers, under_only))}
        print(name, json.dumps(report["results"][name]), flush=True)
    (args.output / "returns.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
