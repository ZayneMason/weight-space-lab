# Breakout: anatomy of the released hand-written network

**Date:** November 10, 2026 · **Status:** pilot · **Cost:** CPU, about 2 minutes

The released Takehi et al. Experiment 5 answer is a 600-32-32-6 tanh policy, but most
of its parameters are zero. `ablate.py` reports which units carry weights and
evaluates zero-ablations of each used first-layer unit under the published settings
(sampled softmax actions, sticky actions 0.1, 10,000-step ceiling, episode seeds as in
`lab.py breakout`).

```bash
python experiments/breakout-circuit/ablate.py runs/breakout-full/reference/published.npz \
    --episodes 200 --workers 3 --output runs/breakout-circuit
```

## Structure

| | Count |
|---|---:|
| Parameters in the architecture | 20,486 |
| Nonzero parameters | **815** (4.0%) |
| First-layer units with any weight | 6 of 32 (91–190 nonzero inputs each; 208 of 600 input features used) |
| Second-layer units with any weight | 8 of 32 (24 nonzero weights, constants ±1000 and ±200) |
| Output logits driven by layer 2 | 2 of 6: left and right (weights of 30) |
| Logit biases | no-op 0, left/right 90, other three actions −100 |
| Distinct nonzero values | 61 |

It reads like a program written in weights: pairs of ±1000 second-layer weights act as
sign detectors on the first-layer features, and the output layer turns them into a
left/right choice.

## Pilot: single-unit ablations (episodes 0–199)

| Variant | Mean score (SE) | Mean episode length | Episodes at the ceiling |
|---|---:|---:|---:|
| Intact | 481.65 (17.31) | 6,881.8 | 45.5% |
| Unit 0 zeroed | 1.84 (0.15) | 25.6 | 0% |
| Unit 1 zeroed | 4.88 (0.14) | 58.0 | 0% |
| Unit 2 zeroed | 8.36 (0.25) | 96.5 | 0% |
| Unit 3 zeroed | 8.26 (0.22) | 94.9 | 0% |
| Unit 4 zeroed | 0.53 (0.04) | 11.3 | 0% |
| Unit 5 zeroed | 0.53 (0.04) | 11.3 | 0% |

The intact run equals the first 200 episodes of the
[full replay](../../evidence/breakout-full-replay/) exactly. **Every one of the six
first-layer units is essential**: removing any single one cuts the mean score by more
than 98%. Units 4 and 5 are the widest (190 and 180 inputs) and the most damaging to
remove.

Evidence: [`evidence/breakout-circuit/summary.json`](../../evidence/breakout-circuit/summary.json).
Limits: a 200-episode pilot of single-unit zero-ablations of the released network only.
The fresh agent-written network (1,076 nonzero weights) is not in this repository.
Second-layer ablations, receptive-field images and the released-versus-fresh
comparison are [roadmap #2](../../docs/ROADMAP.md).
