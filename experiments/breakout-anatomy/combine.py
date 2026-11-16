"""Combine the author's guesses with the under-column table, row by row (roadmap #19)

#18 found that refitting the rising-ball guesses to the under column fixes the brick rate
(834 without sticky actions) but still trails the author's table at the published setting
(357 against 482), because the author's guesses keep the paddle safe on short flights (#12,
#13, #15). This script swaps rows between the two tables to find out whether a mix beats both.

- the author's table with one row replaced by the under-column table, for every row;
- the under-column table with one row replaced by the author's, for every row;
- a few groups (the author's rows 3-5, 2-5, 3-4 inside the under-column table).

    python experiments/breakout-anatomy/combine.py CHECKPOINT --output runs/breakout-combined
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("breakout_underfit", HERE / "underfit.py")
underfit = importlib.util.module_from_spec(_spec)
sys.modules["breakout_underfit"] = underfit
_spec.loader.exec_module(underfit)
returns, rowswap, hedge, targets, anatomy = (underfit.returns, underfit.rowswap, underfit.hedge,
                                             underfit.targets, underfit.anatomy)

UNDER = ROOT / "evidence/breakout-underfit/underfit.json"
GROUPS = ((3, 4, 5), (2, 3, 4, 5), (3, 4))
AUTHOR_GROUPS = ((3, 8),)  # the two single swaps that scored above the author's table
ALL_ROWS = set(range(10))


def under_table(path=UNDER):
    raw = json.loads(Path(path).read_text())["tables"]["under"]
    return {tuple(int(v) for v in key.split(",")): float(value) for key, value in raw.items()}


def variants(published, under):
    """name -> table: both pure tables, every single-row swap, and the groups."""
    base_under = rowswap.mix(published, under, ALL_ROWS)
    rows = sorted({key[0] for key in under})
    out = {"author": published, "under": base_under}
    for row in rows:
        out[f"author, row {row} from under"] = rowswap.mix(published, under, {row})
        out[f"under, row {row} from author"] = rowswap.mix(base_under, published, {row})
    for group in GROUPS:
        out[f"under, rows {group[0]}-{group[-1]} from author"] = rowswap.mix(base_under, published, set(group))
    for group in AUTHOR_GROUPS:
        out[f"author, rows {','.join(map(str, group))} from under"] = rowswap.mix(published, under, set(group))
    return out


def paired(report, base="author"):
    """Each variant's mean score difference from ``base`` on the same seeds, with its standard error."""
    ref = report["results"][base]["sticky_0.1"].get("scores")
    out = {}
    for name, result in report["results"].items():
        scores = result["sticky_0.1"].get("scores")
        if name == base or not ref or not scores:
            continue
        d = np.array(scores, dtype=np.float64) - np.array(ref, dtype=np.float64)
        out[name] = {"difference": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", help="score only these variant names")
    parser.add_argument("--seed-start", type=int, default=0, help="first episode index (held-out checks use 200 onward)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = anatomy.load(args.checkpoint)
    tables = variants(hedge.published_table(arrays), under_table())
    path = args.output / "combined.json"
    report = json.loads(path.read_text()) if path.exists() else {
        "schema": "weight-space-breakout-combined-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
        "episodes": args.episodes, "seed_start": args.seed_start, "results": {}}
    for name, table in tables.items():
        if (args.only and name not in args.only) or name in report["results"]:
            continue
        weights = targets.set_ascending_targets(arrays, table)
        result = {"sticky_0": underfit.score(weights, 1, 1, sticky=0.0), "sticky_0.1": underfit.score(weights, args.episodes, args.workers, start=args.seed_start)}
        report["results"][name] = result
        print(name, json.dumps({k: {m: v[m] for m in ("mean_score", "score_se", "deaths_per_10k_steps", "bricks_per_return")}
                                for k, v in result.items()}), flush=True)
        path.write_text(json.dumps(report, indent=1) + "\n")
    report["paired_vs_author"] = paired(report)
    if "under" in report["results"]:
        report["paired_vs_under"] = paired(report, base="under")
    path.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
