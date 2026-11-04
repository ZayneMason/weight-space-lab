# Reproduce the research, then test a fresh attempt

This route runs Experiment 5's MinAtar **Breakout** task on a CPU. It is a real network evaluation, not the toy sign-classifier demo. It supports both Codex and DSH as the orchestration host; the automated experimental editor is the Codex CLI, using the published model and effort by default.

## What this reproduces

- Research source pinned to `RikiyaT/lm_understand-logs` commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
- Exact published 600–32–32–6 tanh network, with 20,486 float32 parameters.
- Exact input-channel mapping and six-action output layout from the original prompt.
- MinAtar 1.0.15, softmax-sampled actions, sticky action probability 0.1, 1,000 episodes, maximum 10,000 steps per episode.
- Hash verification of every downloaded asset and element-for-element reconstruction of the published weights from the published answer.

The original evaluator's seeds and backend were not released. This implementation records its own seeds, uses float32 dense inference and float64 softmax, and creates a fresh seeded environment per episode. Report statistical agreement, not bit-exact score reproduction. PPO checkpoints were not released; the PPO numbers are published references, not locally reproduced baselines.

## Run the reproduction

Resolve `scripts/reproduce_breakout.py` relative to this skill directory. Use `uv run --script <absolute-script-path> ...`; it pins NumPy 2.5.4 and MinAtar 1.0.15. Alternatively, use a Python environment with those packages installed. Set writable `MPLCONFIGDIR` and `XDG_CACHE_HOME` paths if the sandbox blocks plotting-library caches. The evaluation does not display a GUI.

Create a working directory first. Each command's output must be new:

```sh
python reproduce_breakout.py fetch ./reference
python reproduce_breakout.py verify-reference ./reference
python reproduce_breakout.py evaluate ./reference/published.npz ./replayed.json --episodes 1000 --seed 0 --workers 4
python reproduce_breakout.py compare ./reference ./replayed.json
```

`fetch` downloads the original prompt, answer, network, score table, and licenses. Data/networks use CC BY 4.0; the prompt includes GPL-3.0 MinAtar source. Keep the downloaded attribution and licenses with those assets when sharing. The plugin package contains the runner, not these third-party assets.

For a quick installation check, `--episodes 20` is useful, but label it a pilot. The full protocol uses 1,000. The script writes per-episode scores and lengths, mean, standard error, a normal-approximation 95% interval, runtime, package versions, and checkpoint/evaluator hashes. Changing settings changes the experiment.

## One fresh network-writing attempt

After validating the published network, run:

```sh
python reproduce_breakout.py fresh ./reference ./fresh-run --model gpt-6-astra --effort xhigh
python reproduce_breakout.py evaluate ./fresh-run/fresh.npz ./fresh-eval.json --episodes 1000 --seed 0 --workers 4
python reproduce_breakout.py compare ./reference ./replayed.json --candidate ./fresh-eval.json
```

The fresh command starts one ephemeral Codex conversation in a new directory. It uses the original prompt, neutral instructions, no project instructions or user configuration, and disables tools, plugins, browsing, and skill discovery. It uses the existing Codex sign-in and consumes that account's model allowance. It records the requested model, effort, exact command, source/prompt hashes, stdout event stream, stderr, final answer, and parse/audit results. It exposes no published answer, network, or scores to the editor. Training-data contamination cannot be ruled out; this is a fresh conversation, not proof the model has never encountered the research.

The parser interprets only numeric assignments in the final `edits` block. It does not execute generated Python. Unlike the original permissive parser, it rejects the entire reply if any statement is unsupported instead of silently skipping that statement. This stricter behavior is reported as a protocol difference. A parse failure is an experimental failure to report, not permission to repair the answer or make another call silently.

The audit rejects tool events, failed turns, and unknown event types. It allows only explicitly recognized CLI startup notices before the model turn; those notices remain in the audit record. `finalize ./fresh-run` can re-audit and parse an already completed run after a parser/auditor fix without making another model call; it leaves the original `run.json` intact and writes `finalized.json`.

For a different editor or a DSH-native run, use the original `reference/prompt.txt` in a fresh tool-free conversation, then save the response and run:

```sh
python reproduce_breakout.py import-answer ./answer.md ./fresh.npz
```

That route cannot claim the Codex tool-free event audit. Preserve the other host's transcript, model, settings, and tool audit independently and label the editor variation. Do not reuse the orchestration chat, which has seen the published solution.

Freeze the checkpoint before evaluation. Do not feed final evaluation scores back into a nominally one-shot attempt. If the user later wants iterative edits, start a separate protocol with practice seeds, a stated feedback budget, and untouched final seeds. Do not present a best-of-many selection as a one-shot result.

## Interpreting the outcome

Compare the replay to the published mean and uncertainty. Compare the fresh attempt to the replay using identical settings and seeds; report both absolute scores and paired differences. A fresh attempt may perform poorly or fail to parse. Report all attempts and explain uncertainty without promising capability gains. The default model may resolve to an updated hosted model over time; retain the CLI transcript and model metadata available from the host.

## Qwen is a separate model-specific experiment

The article's Qwen run is substantially different. Its public release includes exact BF16 weight diffs against `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`. The p50c edit can be rebuilt with tensor-hash checks. MATH-500 evaluation uses thinking enabled, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, a 32,768-token ceiling, four samples per problem, and math-verify grading. A full paired MATH-500 run has 4,000 generations and a ceiling of 131,072,000 output tokens.

The package now includes this reconstruction and evaluation route in `qwen/`; see the [Qwen guide](../../../qwen/README.md). Both edited tensor hashes and a two-problem pilot at the published token limit have been verified. The full benchmark remains pending. The original agent/forward-pass harness is absent from the release, and fresh edit discovery remains unimplemented. Do not substitute short generations, quantized weights, or altered sampling and call it the published result.

Sources: [Experiment 5](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch), [Experiment 8](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math), [licenses](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE).
