# Breakout reproduction — November 4, 2026

The local replay and a fresh one-shot network both reached the scale of the published Breakout result.

| Network | Mean score | Standard error | Episodes |
|---|---:|---:|---:|
| Published result | 517.411 | 7.417 | 1,000 |
| Released weights, evaluated locally | 502.880 | 7.747 | 1,000 |
| Fresh GPT-6 Astra response, evaluated locally | 511.530 | 5.547 | 1,000 |

The replay differs from the published mean by 1.35 combined standard errors. The fresh network scores 8.65 points above the local replay, with a paired-difference standard error of 9.00; this run does not establish an improvement over the released network. It demonstrates a successful fresh weight-writing attempt on the same task.

The paper's best 100M-step and 500M-step PPO scores are 15.779 and 25.344. Those are published references only: their checkpoints were not released, and PPO was not retrained here.

## Protocol

- Original experiment: Takehi et al., Experiment 5, MinAtar Breakout.
- Source commit: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
- Network: 600 → 32 → 32 → 6, tanh, 20,486 float32 parameters; original channel/action mappings.
- MinAtar 1.0.15; sampled softmax actions; sticky action probability 0.1; maximum 10,000 steps; 1,000 episodes per evaluated network.
- Local environment seeds: episode index, starting at zero. Separate action RNG seeds: episode index + 1,000,000. Each episode creates a fresh environment.
- Local evaluator: float32 dense inference, float64 softmax; NumPy 2.5.4, Python 3.14.6.
- Original evaluator seeds/backend were not released. This is a statistical reproduction of the reported performance, not an identical replay of its random trajectories.

## Fresh editor

One fresh Codex CLI conversation requested `gpt-6-astra`, reasoning effort `xhigh`, using the original prompt and neutral instructions. Tools, browsing, plugins, project instructions, and skill discovery were disabled. No published answer, checkpoint, score, or local evaluation result was included in the editor prompt. No retries or feedback were used.

The response took 623.4 seconds and produced 76 valid edit statements, yielding 1,076 nonzero weights. Its final network was frozen before evaluation. CLI-reported usage: 8,661 input tokens (8,448 cached), 17,055 output tokens, of which 13,441 were reasoning tokens. A fresh conversation does not establish absence of prior exposure in model training.

The original audit function rejected four CLI startup notices because the CLI labels them as error items. After inspection, the auditor was updated to recognize only those known pre-turn notices while continuing to reject tool calls and execution errors. The existing response was re-audited and parsed with `finalize`; **no second model call was made**. The raw `run.json` retains the initial audit rejection, and `finalized.json` records the successful audit and parser hash.

## Evidence

Local run directory: `runs/breakout-reproduction/` (excluded from distributable packages).

- `reference/sources.json`: source URLs and asset hashes; licenses retained alongside downloads.
- `reference/published.npz`: released weights. Parsing the published answer reconstructs every tensor exactly: 92 statements, 20,486 parameters.
- `replayed-final.json`: all 1,000 local replay scores and settings. An earlier full replay produced the same score before the CLI-auditor update.
- `fresh-astra/request.json`, `events.jsonl`, `answer.md`, `run.json`, `finalized.json`: fresh request, transcript, response, initial record, and final audit.
- `fresh-astra/fresh.npz`: generated network.
- `fresh-1000.json`: all 1,000 fresh-network scores and settings.
- `comparison.json`: paired comparison and published references.

Hashes:

```text
Original prompt: 766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876
Released checkpoint: 6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a
Fresh response: d501497f92ca85095a5ca57b08f1a6e5ff1fe38c1818d2b990e058dad82ca5eb
Fresh checkpoint: 82f3d4a81c02cdd7151666231552b9ee5c95fbf16ceb28ff4ade5b4ab72907c5
Final evaluator/parser script: f5a5a4a3bb41ef7531f15adfadc26adcf95ad71f79b3eaa164fedf48746e20da
```

The experiment is repeatable through `skills/weight-space-lab/references/reproduction.md`. Future hosted-model responses may differ; report every attempt. The Qwen math result has not been reproduced by this package.

Sources: [published Experiment 5](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch), [original score table](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch/eval/minatar_eval.json).
