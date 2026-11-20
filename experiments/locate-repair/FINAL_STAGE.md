# Which stage's repair to keep (roadmap #61)

**Date:** November 20, 2026 · **Status:** done · **Cost:** no new runs, under a minute of CPU

The two-step rule (#54) runs up to three stages and always keeps the last one. #60 found that the 1,000-pair third
stage helps at width 16 only when it reaches a perfect practice fit, and makes all four width-32 cases it reaches
worse. This experiment rescores the saved per-stage results (`two_step` in each case) under other final-choice
rules. Each rule looks only at practice accuracy, which the repair sees. None looks at the million-input probe.
Every stage of a case is scored on the same practice labels.

```bash
python experiments/locate-repair/final_stage_choice.py
```

Rules:

- **current:** keep the last stage run.
- **third only if perfect:** keep the third stage only when it fits every practice label, otherwise the second.
- **best practice:** keep the stage with the highest practice accuracy (ties to the later stage, or the earlier).
- **no third stage:** never run it.
- **oracle:** the stage closest to the original on the probe, an upper bound no real rule can reach.

Arms: #54's coarse-order run on seeds 100 to 149 (width 16), #60's run on seeds 200 to 211 (width 32), and #58's
largest-edit run on seeds 100 to 149, a separate run that reaches the third stage in 9 cases. Results are in
`evidence/final-stage-choice/check.json`.

## Results (measured)

Under 0.1% / under 0.01% / exact, and the worst case:

| Rule | Width 16 (50) | Width 32 (12) | Width 16, largest-edit run (50) |
|---|---|---|---|
| Current | 50 / 35 / 27, 0.069% | 12 / 8 / 5, 0.052% | 49 / 34 / 28, 0.32% |
| **Third only if perfect** | 50 / 35 / 27, **0.047%** | 12 / **10** / 5, 0.052% | 49 / 34 / 28, 0.32% |
| Best practice | 50 / 35 / 27, 0.064% | 12 / 9 / 5, 0.052% | 49 / 34 / 28, 0.32% |
| No third stage | 49 / 33 / 24, 0.44% | 12 / 10 / 5, 0.052% | 48 / 31 / 25, 0.44% |
| Oracle | 50 / 36 / 27, 0.047% | 12 / 10 / 5, 0.052% | 49 / 34 / 28, 0.11% |

1. **"Third only if perfect" never loses a line and matches the oracle on lines at width 32.** At width 16 it
   changes the three cases where the third stage fell short (124, 126, 149), all three to a closer repair, and the
   worst case drops from 0.069% to 0.047%. At width 32 it changes 202, 205 and 211, all closer, and 202 and 211
   cross under 0.01% (0.014% to 0.008%, 0.030% to 0.008%).
2. **It misses seed 210.** That case's third stage fits every practice label with a triple that has none of the
   true weights, so the rule keeps it (0.0068%). Its first stage was at 0.0026%. It stays under 0.01%. Seed 104 in
   the largest-edit run is the same kind of case: a perfect practice fit on no true weights, at 0.32% where its
   second stage was at 0.11%.
3. **On the largest-edit run it changes five cases: four closer and one a hair further** (126: 0.0461% to
   0.0469%). No line moves.
4. **"Best practice" is safer but gains less.** It never ends further than the current rule in any arm, but at width
   32 it misses 202, whose second and third stages tie on practice accuracy.
5. **Skipping the third stage saves CPU only at width 32.** It saves 45% there (24.7 to 13.5 hours) with the same
   lines as the oracle, but at width 16 it loses 102, 109 and 129, the three cases the third stage ends exact.
   No other rule saves CPU, because each has to run the third stage to see its practice fit.

## Interpretation

- A perfect practice fit is the main signal that the third stage found something real: all six of its exact wins
  come with one. It is not proof, since two of the eight perfect third-stage fits (210, and 104 in the
  largest-edit run) hold none of the true weights. Short of a perfect fit, the wider pair screen tends to find a better practice fit that
  behaves worse (inferred from 19 third-stage cases across three runs, two of them on the same seeds).
- The rule was chosen after seeing the width-16 and width-32 runs, so these numbers are in-sample. The largest-edit
  run is only a partial check: same seeds, a different tie-break. A clean test needs fresh seeds (#62).
- "Third only if perfect" should become the default final choice once it holds on fresh cases. It costs nothing.
