# The two-step repair on a network twice as wide (roadmap #60)

**Date:** November 20, 2026 · **Status:** done · **Cost:** CPU, about 25 core-hours (four seeds at a time)

Every three-fault result since #28 is at hidden width 16. This experiment runs the full two-step rule (#54:
`--early 2 --two-step`, coarse order) unchanged on 12 fresh three-fault cases at width 32, seeds 200 to 211.
A width-32 network has about twice as many entries, and the screen ranks about 2.3 times as many candidate
triples (median 29,468 against 13,001). Every size, budget and threshold is the same as at width 16.

```bash
python experiments/locate-repair/early_values.py --width 32 --early 2 --two-step \
    --only 200 --output runs/width32/p0                          # one process per seed, four at a time
python experiments/locate-repair/early_values.py --width 32 --early 2 --two-step \
    --merge runs/width32/p*/early-values.json --output runs/width32/merged
python experiments/locate-repair/width32_check.py
```

The comparison arm is #54's run on seeds 100 to 149 at width 16, `evidence/two-step-unseen/two-step-unseen.json`.
The seeds differ, so this compares shares, not seed by seed. Results are in `evidence/width-32/`.

## Results (measured)

| Three-fault cases | Width 16 (#54, 50 cases) | Width 32 (12 cases) |
|---|---:|---:|
| Under 0.1% disagreement | 50 (100%) | 12 (100%) |
| Under 0.01% | 35 (70%) | 8 (67%) |
| Exact three weights | 27 (54%) | 5 (42%) |
| True triple in the final committee | 40 (80%) | 6 (50%) |
| Median disagreement | 0.0045% | 0.0053% |
| Worst case | 0.069% | 0.052% (204) |
| Cases ending at stage 1 / 2 / 3 | 35 / 9 / 6 | 7 / 1 / 4 |
| CPU hours per case | 0.40 | 2.06 (5.2×) |
| Slowest case | 1.3 h | 5.3 h (210) |

1. **The repair still works at width 32.** All 12 end under 0.1% of a million inputs and 8 under 0.01%, about the
   same shares as width 16. Fewer end on the exact weights (5 of 12) and the true triple is in the final committee
   half as often, but the networks they leave behave as well.
2. **The third stage stops helping.** At width 16 the 1,000-pair rerun reaches 6 cases; it ends three of them
   (102, 109, 129) on the exact weights with a perfect practice fit and makes the other three slightly worse. At
   width 32 it reaches 4 cases (202, 205, 210, 211) and makes all four worse than the second stage left them. None
   ends exact and none gets the true triple into its committee. Seed 211 goes from 0.0081% to 0.030%, and seed 210
   fits every practice label with a triple that has none of the true weights.
3. **The third stage is most of the cost.** It takes 45% of the width-32 CPU (11.1 of 24.7 hours) against 22% at
   width 16. Stopping after the second stage at width 32 would leave 12 under 0.1%, 10 under 0.01% (202 and 211
   cross back) and 5 exact, for about 1.1 hours per case. Without it, width 16 would drop to 49, 33 and 24.
4. **Cost grows faster than the network.** Even the seven cases that stop after the first stage take 0.6 to 0.9
   hours each, three to four times the width-16 median of 0.23 hours, in line with ranking 2.3 times as many triples
   (inferred: the run does not time the screen separately).

## Interpretation

- The two-step rule carries over to a wider network without retuning: the share of repairs that behave like the
  original holds. What drops is how often it lands on the very weights that broke, which matches #39's finding
  that the repair often fixes the network with other weights.
- At width 32 the third stage's flag (no committee member fits every practice label) fires on cases that are
  already close (210 was at 0.0026% after the first stage). A wider pair screen then finds a better practice fit
  that behaves worse. This is inferred from four cases.
- A rule that keeps the second stage's repair unless the third stage fits every practice label would keep all
  three width-16 wins and undo the three width-16 losses. At width 32 it would undo three of the four losses but
  not 210, whose third-stage triple fits every practice label. That rule can be scored from the saved per-stage
  results without new runs (#61).
- Twelve cases is a small sample: one more exact case would move the share by 8 points.
