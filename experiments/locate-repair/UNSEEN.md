# The two-step rule end to end on 50 unseen cases (roadmap #54)

**Date:** November 19, 2026 · **Status:** done · **Cost:** CPU, about 19.8 core-hours (four runs side by side)

[#53](TWO_STEP.md) put all 100 three-fault cases under 0.1% with a two-step rule, but those are the seeds the
rule was designed on. This experiment builds the rule into `early_values.py` as one option and runs it on 50
three-fault cases no earlier experiment has touched (seeds 100 to 149):

1. run with 200 pairs and a committee of 200 triples;
2. if the finished repair misses any of its 64 labels, rerun keeping 1,000 triples;
3. if that repair still misses a label, rerun with 1,000 pairs.

```bash
python experiments/locate-repair/early_values.py --early 2 --two-step --seed-start 100 --seeds 50 --output runs/two-step-unseen
python experiments/locate-repair/unseen_check.py --plain-from runs/two-step-unseen
```

The runs were split four ways (seeds 100 to 112, 113 to 125, 126 to 138, 139 to 149) and merged with
`early_values.py --early 2 --two-step --merge`. Every case saves each step it runs, so the plain run (step 1)
is scored from the same runs. `two-step-unseen.json` holds the finished cases, `plain.json` the step-1 rows
and `check.json` the scores.

## Results (measured)

**50 unseen three-fault cases, after all 64 labels:**

| | 200 pairs, 200 triples | **Two-step rule** |
|---|---:|---:|
| Under 0.1% disagreement | 46 | **50** |
| Under 0.01% | 31 | **35** |
| Exact broken entries | 20 | **27** |
| Every visible fault | 29 | **35** |
| True triple in the committee | 29 | **40** |
| Median disagreement | 0.0055% | **0.0045%** |
| Worst case | 0.48% | **0.069%** |
| CPU hours | 10.9 | 19.8 |

1. **The rule holds on unseen cases: all 50 under 0.1%,** against 46 for the plain run, at 1.8 times its cost.
   All four plain failures (seeds 102, 104, 123 and 129, between 0.13% and 0.48%) end between 0.0049% and
   0.015%.
2. **Each step fires about as often as on the design seeds.** The first flag fires on 15 of 50 cases (27 of
   100 in #53) and the second on 6 of those (10 of 100). The 1,000-pair screen takes 7.4 of the 19.8 CPU hours.
3. **No case got worse past 0.1%, and only one got worse past 0.01%.** Seed 133 goes from 0.0074% to 0.016%
   after its larger-committee rerun. Five cases cross 0.01% the other way.
4. **Three cases still miss a label after both reruns** (124, 126 and 149). Their true triple never reaches the
   committee at any step, and each ends close to where it started: 0.012%, 0.069% and 0.045%. Seed 126 is the
   worst case of the 50.
5. **The flag is still a 0.1% alarm, not a 0.01% one.** Seven of the 15 cases left above 0.01% were never
   flagged (worst: seed 114 at 0.037%).

## Interpretation

- *Inference:* the rule's 100 of 100 in #53 was not tuned to those seeds. On 50 new cases it fixes every
  failure of the plain run and fires its expensive step on a similar share of cases.
- The three cases the rule cannot fix are a different kind from #51's: there the true triple was cut on the
  tie-break, but here it does not reach the committee even with 1,000 pairs and 1,000 triples. They stay under
  0.1% because the repairs they do find are close substitutes. Roadmap #55 looks for where they lose it.

## Files

- `unseen_check.py` scores the rule against its own step-1 rows; two tests (one synthetic, one on the evidence).
- `evidence/two-step-unseen/two-step-unseen.json`, `plain.json` and `check.json`.
