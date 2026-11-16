"""Where does the author's ascending table win? (roadmap #12)

Roadmaps #8, #9 and #11 found that replacing the released network's ascending guesses
with a more accurate brick-blind table costs about 200 points, and that refitting does not
recover it. This script localizes the loss by ball row: for each row, swap only that row's
ascending cells (both directions) from the published table to #8's best-mean-distance
table ("accurate into published"), and the reverse ("published into accurate"). Each
variant is scored on the first 300 episodes of the published protocol.

    python experiments/breakout-anatomy/rowswap.py CHECKPOINT --output runs/breakout-rowswap
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("breakout_hedge", HERE / "hedge.py")
hedge = importlib.util.module_from_spec(_spec)
sys.modules["breakout_hedge"] = hedge
_spec.loader.exec_module(hedge)
targets = hedge.targets
anatomy = hedge.anatomy

ROOT = HERE.parents[1]
ACCURATE = ROOT / "evidence/breakout-targets/targets.json"


def accurate_table(path=ACCURATE):
    """#8's best-mean-distance table, keyed (row, col, dir)."""
    raw = json.loads(Path(path).read_text())["best_blind_table"]
    return {tuple(int(v) for v in key.split(",")): float(value[0]) for key, value in raw.items()}


def mix(base, donor, rows):
    """base with every cell of the given rows taken from donor (where donor has it)."""
    return {key: (donor[key] if key[0] in rows and key in donor else value) for key, value in base.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--groups", nargs="+", default=[],
                        help="extra row groups to swap together, e.g. 3,4 3,4,5 (added to an existing report)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = anatomy.load(args.checkpoint)
    published = hedge.published_table(arrays)
    accurate = accurate_table()
    rows = sorted({key[0] for key in accurate})
    variants = {"published": published, "accurate": mix(published, accurate, set(rows))}
    for row in rows:
        variants[f"accurate-into-published/row-{row}"] = mix(published, accurate, {row})
        variants[f"published-into-accurate/row-{row}"] = mix(variants["accurate"], published, {row})
    for group in args.groups:
        chosen = {int(r) for r in group.split(",")}
        label = "-".join(str(r) for r in sorted(chosen))
        variants[f"accurate-into-published/rows-{label}"] = mix(published, accurate, chosen)
        variants[f"published-into-accurate/rows-{label}"] = mix(variants["accurate"], published, chosen)
    existing = args.output / "rowswap.json"
    previous = json.loads(existing.read_text())["results"] if args.groups and existing.exists() else {}
    report = {"schema": "weight-space-breakout-rowswap-v1", "source_checkpoint_sha256": anatomy.breakout.sha(args.checkpoint),
              "episodes": args.episodes, "rows": rows,
              "cells_changed": {name: sum(abs(t[k] - published[k]) > 1e-6 for k in t) for name, t in variants.items()},
              "results": dict(previous)}
    for name, table in variants.items():
        if name in previous:
            continue
        result = targets.evaluate(targets.set_ascending_targets(arrays, table), "network", args.episodes, args.workers)
        result.pop("lengths")
        report["results"][name] = result
        print(name, json.dumps({k: v for k, v in result.items() if k != "scores"}), flush=True)
        (args.output / "rowswap.json").write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
