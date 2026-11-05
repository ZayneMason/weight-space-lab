# GPU configuration

GPU placement is configurable for a single NVIDIA/CUDA host: one card, independent
workers on multiple cards, or groups for tensor/pipeline parallelism. The first
adapter runs paired evaluation of the frozen, attributed Qwen reproduction. The
synthetic editor benchmark still uses CPU targets; Claude/Codex CLI and Fireworks
editor models run externally.

This infrastructure does not establish a new weight-editing result or certify that
a model fits. AMD, Metal, MIG selectors, multi-host scheduling, and automatic
checkpoint sharding are not supported.

## Inspect and plan

Python 3.11+ is sufficient for inventory and planning; no Torch/vLLM import or GPU
allocation is required. These commands work on Windows and Linux/WSL:

```bash
python lab.py gpu inspect
python lab.py gpu plan --config experiments/gpu-layouts/auto.json
```

Planning returns JSON with `ready`, worker UUID masks, and reasons for excluded or
unavailable devices. A successful planning command can report `ready: false`.
Planning creates no output files or model processes. Missing NVIDIA inventory
yields no ready workers.

| Configuration | Placement |
| --- | --- |
| [auto.json](../experiments/gpu-layouts/auto.json) | One independent worker per eligible visible card |
| [single.json](../experiments/gpu-layouts/single.json) | Both arms sequentially on physical GPU 0 |
| [two-workers.json](../experiments/gpu-layouts/two-workers.json) | One arm on GPU 0 and one on GPU 1, concurrently |
| [tensor-parallel.json](../experiments/gpu-layouts/tensor-parallel.json) | Both arms sequentially, each spanning GPUs 0 and 1 with TP=2 |
| [pipeline-parallel.json](../experiments/gpu-layouts/pipeline-parallel.json) | Both arms sequentially, each spanning GPUs 0 and 1 with PP=2 |

Edit selectors to match the inventory. Indices refer to **physical `nvidia-smi`
indices**, not renumbered CUDA ordinals. Complete UUIDs are more stable than indices.
For four-card TP=2, use `device_groups: [["0","1"],["2","3"]]`. Each group must
contain exactly `tensor_parallel_size * pipeline_parallel_size` cards. Groups must
be disjoint. One worker reuses its group sequentially; tasks are assigned
round-robin across multiple workers, with at most one task per group. The paired
adapter has two tasks, so at most two groups perform work.

By default, parent `CUDA_VISIBLE_DEVICES` limits selectable cards. Numeric mask
entries follow the same physical-index convention. UUID prefixes and MIG IDs are
rejected; an empty mask or `-1` selects none. Set `respect_visible_devices: false`
only when intentionally selecting from host inventory instead. Children receive
resolved UUID masks and `CUDA_DEVICE_ORDER=PCI_BUS_ID`; the parent environment is
not modified.

## Settings

Unknown fields are rejected; omitted fields use these defaults.

| Field | Default | Meaning |
| --- | --- | --- |
| `device_groups` | `"auto"` | Eligible visible cards, or explicit lists of indices/UUIDs |
| `tensor_parallel_size` | `1` | TP per worker, integer 1–64 |
| `pipeline_parallel_size` | `1` | PP per worker, integer 1–64 |
| `memory_fraction` | `0.65` | vLLM GPU memory utilization per card, finite value in (0,1] |
| `headroom_mib` | `1024` | Additional free memory required before launch |
| `max_workers` | `null` | Optional limit on groups, integer 1–64 |
| `execution` | `"parallel"` | `parallel` or `sequential` across groups |
| `max_num_seqs` | `16` | vLLM maximum sequences, integer 1–256 |
| `batch_problems` | `4` | Problems per generation batch, integer 1–64 |
| `respect_visible_devices` | `true` | Honor parent visibility mask |

Auto selection skips cards with insufficient free memory and incomplete groups.
Explicit groups retain readiness errors without silently falling back. The
free-memory threshold on **each card** is
`ceil(total_mib * memory_fraction) + headroom_mib`. Lowering it allows a smaller
allocation attempt; it does not make a large checkpoint fit. Two 24 GiB cards do
not become a transparent 48 GiB allocation. TP/PP must be supported by the engine
and model; topology, model dimensions, and KV cache still matter. See
[vLLM 0.33 parallelism](https://docs.vllm.ai/en/v0.33.0/serving/parallelism_scaling/)
and [independent GPU instances](https://docs.vllm.ai/en/v0.33.0/serving/data_parallel_deployment/).

## Qwen execution

Prepare the isolated environment and frozen checkpoints using
[Qwen setup](../qwen/README.md). Execute under Linux/WSL with Linux paths and the
Qwen environment's Python. Windows execution is rejected before creating a run.

```bash
python lab.py gpu pair /path/to/lab/experiment /path/to/new-gpu-run \
  --config experiments/gpu-layouts/auto.json --mode smoke \
  --python /path/to/lab/.venv/bin/python
```

This prints commands, source/prepared-manifest hashes, placement, and token ceilings.
Add `--execute` to launch. This wrapper also sets the CUDA library paths described
in the Qwen guide and **executes** the pair:

```bash
export WEIGHT_SPACE_LAB_HOME=/path/to/lab
export WEIGHT_SPACE_GPU_CONFIG="$PWD/experiments/gpu-layouts/auto.json"
export WEIGHT_SPACE_GPU_OUTPUT=/path/to/new-gpu-run
bash qwen/run.sh smoke
python lab.py gpu audit /path/to/new-gpu-run
```

Without `WEIGHT_SPACE_GPU_CONFIG`, `qwen/run.sh` retains its single-card behavior.
`WEIGHT_SPACE_LAB_PYTHON` selects the interpreter; `WEIGHT_SPACE_GPU_TIMEOUT` changes
the wrapper's per-worker timeout.

| Mode | Problems per arm | Samples per problem | New tokens per sample | Maximum output tokens, both arms |
| --- | --- | --- | --- | --- |
| `smoke` (default) | 2 | 4 | 256 | 4,096 |
| `pilot` | 2 | 4 | 32,768 | 524,288 |
| `full` | 500 | 4 | 32,768 | 131,072,000 |

Smoke is a plumbing check with a short cap. Pilot/full can take substantial time.
`--timeout-seconds` defaults to 3,600 per worker, accepting 1–604,800. Logs are
monitored against 64 MiB per worker; overflow fails the run and retains a truncated
prefix. `--eager` disables CUDA graph capture. This route initiates no hosted
editor calls or new model downloads; workers use prepared weights and
`HF_HUB_OFFLINE=1`.

## Comparison and evidence

Records include the immutable plan, source snapshots with upstream notices,
commands, GPU identities, per-worker logs, journal/state, individual Qwen results,
and comparison. The adapter checks frozen checkpoint hashes before model loading.
Completed runs receive `integrity.json`; `gpu audit` checks file inventory/hashes,
completion state, source snapshots, and declared placement. This is an unsigned
local integrity check; it does not rerun inference or regrade generated text.

Arms must match protocol, problem IDs, sampling, versions, batch size, engine, and
environment. Different UUID masks require their shared placement manifest;
comparison without it rejects them. Jobs must match the plan's sampling bounds and
engine placement settings. GPU model/memory/driver groups must match by default.
The direct CLI's `--allow-hardware-difference` explicitly permits and labels an
exception. Matched GPU classes do not prove bitwise or latency equivalence.

Existing GPU processes are preserved. Free memory is rechecked before each worker
starts. Exclusive lock files prevent overlapping runs using the same local
temporary lock directory. These are snapshots and cooperative locks, not OS
reservations: other programs, users, containers, or WSL instances can still
allocate memory. Stale locks require investigation and are not automatically
deleted. Failed/interrupted runs stop only owned process groups and preserve
partial records plus failure details. Use a **new output directory** for another
attempt; the GPU controller does not automatically resume interrupted pairs.

Inventory, planning, CPU fixture processes, cleanup, placement rejection, and
artifact validation have automated tests. TP/PP model execution requires validation
on the intended hardware/checkpoint. Upstream reproductions remain separate from
new experiments.
