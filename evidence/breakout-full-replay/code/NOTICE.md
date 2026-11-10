# Attribution and external assets

Weight Space Lab is an independent implementation inspired by Takehi et al.'s
[weight-space programming experiments](https://github.com/RikiyaT/lm_understand-logs).
Project code is MIT licensed. Third-party material retains its original license.

- The Qwen answer extractor follows the released `exp8_math/code/mathqa.py`.
  A small verbatim reference fixture is included for parity tests. Its copyright
  and MIT terms are retained in [licenses/upstream-research.txt](licenses/upstream-research.txt).
- Experiment manifests, logs, and networks downloaded from that research repository
  are CC BY 4.0, except material explicitly covered by another license. Fetch commands
  retain source URLs, hashes, and upstream license records.
- MinAtar is a separate dependency. Its GPL-3.0 source appears in the upstream
  Breakout prompt; that prompt and the downloaded networks are not bundled here.
- Qwen checkpoints and MATH-500 content are downloaded from their pinned sources.
  Their licenses and dataset terms remain applicable. Model weights, benchmark
  questions, and generated answer transcripts are not committed to this repository.

Aggregate results and checkpoint verification metadata under `evidence/` describe
local measurements; they do not replace the upstream papers or full raw run records.
