# Weight Space Lab

An experimental toolkit for inspecting, editing, and evaluating neural-network
weights directly, with source preservation and recorded experiment evidence.

It includes a shared skill for **Codex and DeepSeek Harness**, a CPU Breakout
reproduction, and a GPU runner for the published Qwen3.5-0.8B p50c edit.

[Implementation and results report](docs/REPORT.md) · [Breakout results](RESULTS.md) ·
[Qwen reproduction](qwen/README.md) · [Evidence](evidence/README.md)

## Status

This is a research prototype. An agent supplies hypotheses; the tools apply and
measure explicit edits. Automatic discovery of fresh language-model edits is not
implemented, and no capability gain is guaranteed.

| Capability | Status |
|---|---|
| Inspect and edit individual NPZ/SafeTensors checkpoints | Working; F16/F32/F64 edits |
| Reproduce MinAtar Breakout and evaluate a fresh agent-written network | Implemented; recorded 1,000-episode results |
| Reconstruct the published Qwen p50c BF16 edit | Both tensor hashes verified; 318 other tensors unchanged |
| Paired Qwen inference and grading | Two-question pilot completed; full MATH-500 result pending |
| Discover fresh Qwen edits with bounded practice feedback | Future work |

The Qwen pilot produced 1/8 correct samples for the original model and 6/8 for the
published edit, on only two questions. It validates the pipeline and is too small
to establish benchmark improvement. Details and limits are in the report.

## Quick start: checkpoint toolkit

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). The first
invocation installs the script's pinned dependencies.

```bash
uv run --script skills/weight-space-lab/scripts/weight_lab.py demo ./demo-run
uv run --script skills/weight-space-lab/scripts/weight_lab.py inspect ./demo-run/original.npz
```

The demo applies a prewritten repair to a small sign classifier. It checks the
plumbing; it is not an autonomous discovery experiment. See the
[CLI reference](skills/weight-space-lab/references/cli.md) for tensor reads,
set/add/scale operations, direction removal, edit budgets, and paired MLP evaluation.

The generic editor handles one `.npz` or `.safetensors` file at a time. It does not
handle quantized formats or arbitrary checkpoint directories. BF16 Qwen
reconstruction uses the separate model-specific runner below.

## Breakout experiment

No GPU is required. Use a new working directory and an absolute path to
`skills/weight-space-lab/scripts/reproduce_breakout.py` as `SCRIPT`:

```bash
uv run --script SCRIPT fetch ./reference
uv run --script SCRIPT verify-reference ./reference
uv run --script SCRIPT evaluate ./reference/published.npz ./replayed.json
uv run --script SCRIPT fresh ./reference ./fresh-run
uv run --script SCRIPT evaluate ./fresh-run/fresh.npz ./fresh-eval.json
uv run --script SCRIPT compare ./reference ./replayed.json --candidate ./fresh-eval.json
```

The fresh step uses the existing Codex sign-in for one tool-free editor response.
It consumes the signed-in account's model allowance. Model and effort defaults,
tool-event auditing, parser restrictions, licensing, and protocol differences are
documented in the [reproduction guide](skills/weight-space-lab/references/reproduction.md).
Keep the final evaluation out of candidate selection.

Recorded results over 1,000 episodes: released weights scored **502.9 ± 7.7 SE**;
a fresh agent-written network scored **511.5 ± 5.5 SE**. This does not establish an
improvement between the two networks. Raw Breakout run records are not bundled.

## Qwen experiment

The [Qwen guide](qwen/README.md) covers an isolated Linux/WSL environment,
checkpoint reconstruction, independent export auditing, smoke checks, and full
paired evaluation. It pins the research, model, and dataset revisions and preserves
every generation. GPU memory and runtime requirements are described there.

This route reconstructs an existing published edit. It does not implement the
original editor sandbox or forward-pass server. Its small pilot is complete;
the full 4,000-generation comparison is pending.

## Plugin installation

### Codex

The repository includes `.codex-plugin/plugin.json` and shared `skills/` resources.
Add the package to a local plugin marketplace following the
[official packaging instructions](https://developers.openai.com/plugins/build/plugins),
then install it through the Plugins UI. An unpacked copy of the skill directory
can also be used with skill discovery.

### DeepSeek Harness

Use the supported `plugin_manager` tool's `install_bundle` action with this
repository directory, or the DSH CLI:

```bash
dsh plugin --profile desktop add /absolute/path/to/weight-space-lab
```

Read installation warnings and restart if requested. The plugin registers a skill
through the `skills` service and does not override a preset. Use a host preset
with shell execution to run the Python helpers.

## Development

With Python 3.12 and NumPy, SafeTensors, MinAtar, PyTorch, and the pinned grader
dependencies installed:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m unittest discover -s qwen -p 'test_*.py' -v
node --test tests/plugin.test.mjs
python scripts/package.py
```

The CPU tests do not require a GPU. One optional Breakout integration test skips
when its downloaded reference assets are absent. The package builder writes a
Codex ZIP, DSH tarball, and checksum manifest under `dist/`.

## Attribution

Independent implementation inspired by Takehi et al.'s
[article](https://rikiyat.github.io/blog/weight-space-programming/) and
[released experiments](https://github.com/RikiyaT/lm_understand-logs).
Project code is [MIT licensed](LICENSE); third-party material retains its original
terms. See [NOTICE.md](NOTICE.md). No model weights, benchmark questions, credentials,
or generated-answer transcripts are committed to this repository.
