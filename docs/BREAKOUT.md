# MinAtar Breakout: attributed reproduction

This adapter evaluates the published network from Takehi et al.'s
[Experiment 5](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch).
Weight Space Lab supplies the local runner and evidence workflow; the published
network and research result belong to the upstream authors.

## Prerequisites and cost

Python 3.11+, NumPy 2.5.4 and MinAtar 1.0.15. Install `requirements.txt` and
`requirements-breakout.txt` into a CPU environment, or use the complete uv command:

```bash
uv run --with minatar==1.0.15 --script lab.py breakout runs/breakout-smoke --mode smoke
```

This fetches six pinned public research assets, verifies their hashes and numeric
answer reconstruction, evaluates 8 episodes (seed 0, at most 1,000 steps each),
replays the first 200 steps of episode zero, and writes `report.html`. The report
shows **MinAtar Breakout**, not an Atari emulator. No credentials, GPU, or hosted
model allowance are used. Dependencies and reference downloads need network access.

With a previously downloaded reference, avoid downloading it again:

```bash
python lab.py breakout runs/another-smoke --mode smoke --reference runs/breakout-smoke/reference
```

The supplied reference must match the pinned assets and canonical source metadata.
The new run copies the reference and licenses without changing the original.

## Full versus smoke

```bash
uv run --with minatar==1.0.15 --script lab.py breakout runs/breakout-full --mode full
```

Full mode enforces 1,000 episodes and a 10,000-step ceiling. It is longer than a
smoke check. Smoke mode permits 2..50 episodes and a bounded step ceiling; those
scores are not full benchmark results. Published full-evaluation scores are context
only when episode/step counts differ; full-score differences are not calculated
for the smoke workflow.

The original environment seeds/backend were not released. Local seeds are recorded
as `env=seed+episode`, `action=seed+episode+1000000`, with a fresh environment each
episode. A replay uses the same implementation, policy and seeds as evaluation;
episode zero is selected before scores are seen. Only a prefix is displayed if the
frame cap is reached. Verification reproduces that episode and checks its frames.

`evaluation.json` retains every episode's score, length and truncation flag.
`verification.json`, `comparison.json`, `replay.json`, `reference/sources.json`,
the licenses, and `experiment.json` connect the result to its inputs and settings.
The adapter rejects incomplete records, changed metrics, or incompatible paired
evaluations. Failed workflows retain `failure.json` and partial evidence.

## Fresh generation: explicit opt-in

The following step starts one fresh tool-free editor response using the signed-in
Codex account and consumes its model allowance. It is not part of the smoke or
full reproduction command. Invoke it only when that cost is authorized:

```bash
python lab.py breakout-tools fresh runs/breakout-smoke/reference runs/fresh-breakout
```

The output directory must be new. Keep the published answer, weights and scores
out of the editor context. Preserve every attempt, failed parse and regression;
never silently retry. The deeper [reproduction reference](../skills/weight-space-lab/references/reproduction.md)
documents the model defaults, audit and statistical differences.

Earlier recorded 1,000-episode measurements are in [RESULTS.md](../RESULTS.md).
Their raw transcripts and episode logs are not available in this repository.
New bounded runs are separate evidence and do not replace or recreate those logs.
