# Qwen published-edit reproduction

Reconstruct the published p50c BF16 edit to Qwen3.5-0.8B, verify both tensor hashes,
and evaluate the frozen baseline and edited checkpoints on MATH-500.

The [report](../docs/REPORT.md) and [aggregate evidence](../evidence/README.md) document
the completed two-question pilot. A full benchmark result is not yet available.
Fresh language-model edit discovery is not implemented.

## Setup

The measured environment used Ubuntu 24.04 under WSL, an RTX 3090 with 24 GB VRAM,
Python 3.12, and the packages in `requirements.lock.txt`. The runner requires Linux,
a compatible NVIDIA driver, and a working C/C++ compiler and Python development
headers. On Ubuntu, install `build-essential` and `python3.12-dev` through the
system package manager. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
before the commands below.

Run from the repository root:

```bash
export WEIGHT_SPACE_LAB_HOME="$HOME/.local/share/weight-space-lab"
uv venv "$WEIGHT_SPACE_LAB_HOME/.venv" --python 3.12
uv pip sync --python "$WEIGHT_SPACE_LAB_HOME/.venv/bin/python" qwen/requirements.lock.txt
"$WEIGHT_SPACE_LAB_HOME/.venv/bin/python" qwen/preflight.py --workdir "$WEIGHT_SPACE_LAB_HOME/experiment"
"$WEIGHT_SPACE_LAB_HOME/.venv/bin/python" qwen/reproduce_qwen.py prepare "$WEIGHT_SPACE_LAB_HOME/experiment"
"$WEIGHT_SPACE_LAB_HOME/.venv/bin/python" qwen/audit_export.py "$WEIGHT_SPACE_LAB_HOME/experiment" "$WEIGHT_SPACE_LAB_HOME/export-audit.json"
```

Preparation downloads pinned public model weights, reference assets, and the dataset.
The output directory must be new. The original checkpoint remains separate from
the reconstructed candidate. Assets remain under their upstream licenses.

## Evaluate

```bash
bash qwen/run.sh smoke   # Two problems, four samples each, 256 new tokens.
bash qwen/run.sh pilot   # Same two problems, 32,768 new tokens.
bash qwen/run.sh full    # 500 problems x four samples x two checkpoints.
```

The wrapper uses GPU 0 by default and a 65% memory budget. Set `CUDA_VISIBLE_DEVICES`,
`WEIGHT_SPACE_GPU_MEMORY`, or `WEIGHT_SPACE_LAB_PYTHON` explicitly to use another
device, budget, or environment. Check available VRAM before running. Both checkpoints
run sequentially with identical settings. The full comparison has up to 131,072,000
output tokens and can take many hours. Pilot outputs remain separate from full results.

Outputs live under `$WEIGHT_SPACE_LAB_HOME/{smoke,pilot,full}-paired/`. Each completed
problem retains its exact prompt, all four outputs and token IDs, seeds, finish
reasons, extraction, and grades. Repeating the same command resumes completed
problems; incompatible source, package, input, or job settings are rejected.
Keep runtime directories outside the repository.

The tested CUDA 13.0 pins avoid a compiler/header mismatch. The wrapper discovers
the pip CUDA toolkit, adds a workspace-local runtime link when required, and adds
WSL's driver library paths when present. FlashInfer sampling and full decode CUDA
graphs are enabled; `--eager` on the Python CLI is available for diagnosis.

## Scientific scope

- Research and model revisions, dataset revision, sampling, and grader versions are
  pinned in `protocol.resolved.json`.
- Thinking is enabled; sampling uses temperature 0.6, top-p 0.95, top-k 20, presence
  penalty 1.5, four samples per problem, and a 32,768-new-token ceiling.
- Accuracy averages all samples. It does not select the best answer of four.
- Local calls group four problems with per-problem seed equal to dataset ordinal.
  The released evaluator calls a whole shard with seed zero. This is a documented
  statistical reproduction attempt rather than an identical replay.
- Bootstrap intervals resample problems, keeping each problem's four samples together.
- The published candidate is frozen. Benchmark scores must not guide fresh edit selection.

## Tests

```bash
"$WEIGHT_SPACE_LAB_HOME/.venv/bin/python" -m unittest discover -s qwen -p 'test_*.py' -v
```

Tests use small synthetic tensors and the attributed answer-extraction fixture;
they do not download weights or require a GPU. The independently verified GPU
experiment is separate from these unit tests.
