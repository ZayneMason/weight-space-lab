# The third-stage rule on fresh cases (roadmap #62)

**Date:** November 20, 2026 · **Status:** done · **Cost:** 29.6 core-hours

#61 found, on runs it had already seen, that keeping the 1,000-pair third stage only when it fits every practice
label never lost a line. This experiment builds that rule into the repair tool and runs it on 50 fresh width-16
three-fault cases, seeds 150 to 199.

```bash
python experiments/locate-repair/early_values.py --early 2 --two-step --final-choice last-if-perfect \
    --only 150 --output runs/third-rule/s150                     # one process per seed 150 to 199, four at a time
python experiments/locate-repair/early_values.py --early 2 --two-step --final-choice last-if-perfect \
    --merge runs/third-rule/s*/early-values.json --output runs/third-rule/merged
python experiments/locate-repair/third_rule_check.py
```

`--final-choice last-if-perfect` keeps the last stage unless it ran more than one stage and its last stage misses
a practice label, then it keeps the stage before. The default, `last`, leaves every earlier run unchanged. Every
stage's result stays in `two_step`, so both final choices are scored from the same run. The stage the tool kept
matches #61's rescoring on all 50 cases. Results are in `evidence/third-stage-rule/`.

## Results (measured)

Under 0.1% / under 0.01% / exact, and the worst case:

| Rule | Fresh, seeds 150 to 199 (50) | Pooled with #54, seeds 100 to 199 (100) |
|---|---|---|
| Last (current default) | **49** / 32 / 31, 0.136% | **99** / 67 / 58, 0.136% |
| Third only if perfect | 48 / 32 / 31, 0.136% | 98 / 67 / 58, 0.136% |
| No third stage | 48 / 30 / 29, 0.136% | 97 / 63 / 53, 0.44% |
| Oracle (sees the probe) | 49 / 32 / 31, 0.136% | 99 / 68 / 58, 0.136% |

1. **The rule did not hold on fresh cases.** Seven of 50 cases reached the third stage. In the five where it
   missed a practice label, the rule went back to the second stage: one ended closer (157: 0.078% to 0.062%) and
   four further (160: 0.0009% to 0.0010%, 170: 0.023% to 0.026%, 186: 0.022% to 0.034%, 197: 0.039% to 0.102%).
   Seed 197 crosses the 0.1% line, so the rule loses one case there.
2. **A perfect third-stage fit is still worth keeping.** The other two cases (159, 198) reached a perfect practice
   fit in the third stage, found all three broken weights, and crossed under 0.01% (0.057% to 0.0093%, 0.085% to
   0.0086%). Both rules keep those. Across seeds 100 to 199, all five exact third-stage wins (102, 109, 129, 159,
   198) come with a perfect fit.
3. **Short of a perfect fit, the third stage is a coin flip.** It ended closer than the second stage in 4 of 8
   such cases across both seed ranges: 0 of 3 in seeds 100 to 149, which is what #61 saw, and 4 of 5 here.
4. **Nothing that reads only practice accuracy beats the current rule on these cases.** The oracle ties it on all
   three lines in the fresh run and gains one case under 0.01% in the pool.
5. **Skipping the third stage still costs too much at width 16.** It saves 26% of the CPU (29.6 to 21.8 hours) but
   loses 159 and 198 and two lines under 0.01%.
6. **The fresh seeds came out a little differently from #54** with the current rule: 49 / 32 / 31 against
   50 / 35 / 27. The one case over 0.1% (171, 0.136%) fits every practice label in the first stage with only one of
   the three broken weights, so no stage rule can see it.

## Interpretation

- Keep `last` as the default at width 16. #61's gain came from three cases where the imperfect third stage happened
  to end further, and five fresh cases did not repeat it. The `last-if-perfect` flag stays for width 32, where all
  four imperfect third stages ended further in #60, but that is still in-sample there (inferred, 4 cases).
- Most of what is left under the 0.01% line is not about which weights to edit. Of the 33 pooled cases at or over
  0.01%, 17 edit exactly the three broken weights, so their gap is in the values. That points the next step at
  refining the values once the weights are right (#63), not at another search stage.
