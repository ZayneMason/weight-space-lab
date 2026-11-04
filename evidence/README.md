# Included evidence

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
