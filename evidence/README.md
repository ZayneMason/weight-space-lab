# Included evidence

Evidence is grouped by origin. `controller-controls.json` describes prewritten
manual controls for the independent controller. `sign-demo/` is the prewritten
offline demonstration; `breakout-smoke/` is an attributed bounded reproduction.
Neither is an external-agent discovery result or a full new benchmark.

`qwen/` contains a fixed November 4, 2026 snapshot:

- `reconstruction.json`: source revisions, downloaded-asset hashes, exported
  checkpoint hashes, and both edited-tensor hash checks.
- `export-audit.json`: independent comparison of all 320 exported tensors.
- `pilot-baseline.json` and `pilot-p50c.json`: aggregate results for two problems
  with four samples per problem and a 32,768-new-token ceiling.
- `pilot-comparison.json`: paired pilot metrics and a problem-clustered bootstrap.

Only two questions were used in the pilot. Its confidence intervals are not a
reliable estimate of full-benchmark performance. Final MATH-500 results are pending.

Private workstation paths, model files, benchmark questions, generated answers,
runtime process state, and raw transcripts are excluded. The source hashes in the
reconstruction record refer to the measured original exports, not a promise that
future environment versions reproduce identical random trajectories.

Earlier Breakout results are documented in [RESULTS.md](../RESULTS.md). The raw
Breakout transcript and episode records are not included in this repository.

`qwen/pilot-samples.json` is a sanitized derivation of the existing raw local pilot:
per-sample correctness, token counts, finish reasons, and content digests. Its
aggregate values are checked against both stored summaries. Questions, answers,
generated text/token IDs, private paths, and environment variables are omitted.
The export does not independently regrade the omitted generated text.

New workflow runs keep relative artifact ledgers, snapshots of executed toolkit
source, and local reports. `lab.py export-evidence` creates a metadata-only export
without checkpoints or prompts; review free-text fields before sharing it.
The prior raw Breakout records were not available in the local bundle. A new bounded
smoke evaluation is a separate measurement and must not stand in for those records.

Independent controller experiments live under generated `runs/` directories,
separately from upstream reproductions. Their private seeds, held-out data, editor
transcripts and runtime state should not be automatically published. Manual controls
are explicitly labeled and do not count as external-agent research trials.
