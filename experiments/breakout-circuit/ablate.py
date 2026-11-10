"""Anatomy of the released Breakout network: sparsity and single-unit ablations.

The released Experiment 5 answer is a 600-32-32-6 tanh network, but only a few
units carry nonzero weights. This script reports that structure and evaluates
zero-ablations of each used first-layer unit (incoming weights and bias set to
zero, so tanh outputs 0) under the published evaluation settings.

Episode seeds match ``lab.py breakout``: with ``--seed 0`` and N episodes, the
unablated run equals the first N episodes of the full replay.

    python experiments/breakout-circuit/ablate.py runs/breakout-full/reference/published.npz \
        --episodes 200 --workers 3 --output runs/breakout-circuit
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("reproduce_breakout", ROOT / "skills/weight-space-lab/scripts/reproduce_breakout.py")
breakout = importlib.util.module_from_spec(spec)
sys.modules["reproduce_breakout"] = breakout  # worker processes pickle its functions by module name
spec.loader.exec_module(breakout)

NAMES = ["actor/dense1/kernel", "actor/dense1/bias", "actor/dense2/kernel", "actor/dense2/bias", "actor/logits/kernel", "actor/logits/bias"]


def structure(arrays):
    w1, b1, w2, b2, w3, b3 = (arrays[n] for n in NAMES)
    used1 = np.flatnonzero((w1 != 0).any(0) | (b1 != 0))
    used2 = np.flatnonzero((w2 != 0).any(0) | (b2 != 0))
    actions = np.flatnonzero((w3 != 0).any(0))
    values = np.concatenate([a[a != 0] for a in arrays.values()])
    return {"parameters": int(sum(a.size for a in arrays.values())),
            "nonzero": int(sum((a != 0).sum() for a in arrays.values())),
            "nonzero_by_tensor": {n: int((arrays[n] != 0).sum()) for n in NAMES},
            "layer1_units_used": used1.tolist(), "layer1_inputs_per_unit": {int(u): int((w1[:, u] != 0).sum()) for u in used1},
            "layer2_units_used": used2.tolist(), "input_features_used": int((w1 != 0).any(1).sum()),
            "actions_driven_by_layer2": actions.tolist(), "logit_bias": b3.tolist(),
            "distinct_nonzero_values": int(len(np.unique(values)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        arrays = {n: archive[n].copy() for n in NAMES}
    anatomy = structure(arrays)
    variants = {"intact": arrays}
    for unit in anatomy["layer1_units_used"]:
        ablated = {n: a.copy() for n, a in arrays.items()}
        ablated["actor/dense1/kernel"][:, unit] = 0
        ablated["actor/dense1/bias"][unit] = 0
        variants[f"ablate-layer1-unit-{unit}"] = ablated
    results = {}
    for name, tensors in variants.items():
        folder = args.output / name
        folder.mkdir()
        np.savez(folder / "model.npz", **tensors)
        summary = breakout.evaluate(folder / "model.npz", folder / "evaluation.json", episodes=args.episodes,
                                    seed=0, workers=args.workers, max_steps=args.max_steps)
        rows = json.loads((folder / "evaluation.json").read_text())["results"]
        results[name] = {"mean": summary["mean"], "se": summary["se"], "mean_length": summary["mean_length"],
                         "capped_share": float(np.mean([r["truncated"] for r in rows])),
                         "checkpoint_sha256": summary["checkpoint_sha256"]}
        print(name, json.dumps(results[name]), flush=True)
    report = {"schema": "weight-space-breakout-circuit-v1", "source_checkpoint_sha256": breakout.sha(args.checkpoint),
              "episodes": args.episodes, "max_steps": args.max_steps, "seed": 0, "anatomy": anatomy, "ablations": results,
              "limits": "Pilot: zero-ablation of single first-layer units only; episodes 0..N-1 of the published protocol."}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["anatomy"], indent=2))


if __name__ == "__main__":
    main()
