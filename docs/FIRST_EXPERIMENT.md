# First experiment: inspect, edit, evaluate, compare, review

The offline example is a deterministic sign classifier with a prewritten repair.
It is a demonstration of the tools, not an external-agent experiment. Its six
predefined examples are illustrative; the legacy name `heldout.json` does not
turn them into an independent unseen test set.

## Prerequisites and cost

Python 3.11+ plus NumPy 2.5.4 and SafeTensors 0.8.0 are sufficient. With uv:

```bash
uv run --script lab.py demo runs/sign-demo
```

uv may fetch software packages on first invocation. The experiment itself makes
no network requests and needs no credentials, model downloads or GPU. For a fully
offline invocation, install `requirements.txt` into the chosen Python environment
before disconnecting, then run `python lab.py demo runs/offline-demo`.

## Review the result

Open the generated `report.html` in a local browser. No server or CDN is used.
The page includes the before/after number-line plot, each example's predictions,
the exact JSON patch and reason, hashes, and links to these files:

| File | Purpose |
|---|---|
| `original.npz`, `inspection.json` | Preserved source and tensor metadata |
| `architecture.json`, `heldout.json` | Explicit MLP and six example inputs/labels |
| `edit.json` | Proposed patch, source hash and rationale |
| `candidate/model.npz` | Separate edited candidate |
| `candidate/patch.json`, `candidate/receipt.json` | Applied patch and actual changed counts |
| `baseline.json`, `candidate.json` | Matched per-example evaluations |
| `comparison.json` | Recomputed accuracy, fixes and regressions |
| `behavior.svg`, `report.html` | Local visualization and report |
| `experiment.json` | Relative file hashes, adapter/scope, versions and code hashes |

The fixed source logits are inverted. The prewritten patch scales both weight
entries by -1, leaves both bias entries unchanged, and repairs all six examples:
0/6 correct becomes 6/6. Two of four stored entries change. Rollback means choosing
the unchanged original checkpoint.

```bash
python lab.py inspect runs/sign-demo/original.npz
python lab.py tensor runs/sign-demo/original.npz weight --start 0 --count 2
python lab.py verify runs/sign-demo
python lab.py export-evidence runs/sign-demo runs/sign-evidence
```

Every workflow requires a new output directory. A collision fails clearly and
preserves the existing result. An interrupted/failed workflow retains partial
artifacts and `failure.json`; it is not treated as a completed experiment.

The integrity ledger is unsigned. Demo verification checks actual tensor changes,
reapplies the recorded patch, reruns both MLP evaluations, and recomputes the
comparison. An export omits checkpoint files, benchmark prompts and hosted-model
transcripts; inspect its text fields before redistribution. It is metadata, not
an independently signed scientific attestation.

## Supported adapters

`inspect`, `tensor`, `edit`/`patch`, `evaluate` and `compare` reuse the existing
checkpoint CLI. Edits support individual NPZ/SafeTensors files in F16/F32/F64;
evaluation supports explicitly described dense MLP classifiers only. Qwen BF16
reconstruction uses the [separate GPU adapter](../qwen/README.md).

For an independently selected candidate under a declared feedback budget, use the
[research controller protocol](../experiments/threshold-repair/PROTOCOL.md) instead
of presenting this hand-authored repair as discovery.
