# Take the middle only when the labels leave room, on fresh cases (roadmap #64)

**Date:** November 21, 2026 · **Status:** done · **Cost:** 32.4 core-hours

#63 found, on cases it had already seen, that moving a repair's values to the middle of the region that fits its
labels helps most where that region is wide. This experiment fixes the rule beforehand and runs it inside the
two-step repair on 50 fresh width-16 three-fault cases, seeds 200 to 249, for every case, not only exact ones.

```bash
python experiments/locate-repair/early_values.py --early 2 --two-step --middle \
    --only 200 --output runs/middle/s200                         # one process per seed 200 to 249, four at a time
python experiments/locate-repair/early_values.py --early 2 --two-step --middle \
    --merge runs/middle/s*/early-values.json --output runs/middle/merged
python experiments/locate-repair/middle_check.py
```

`--middle` runs #63's 1,600-step walk on the winner's values after each stage's last round and records the result
in the row's `middle` field. The repair itself (`switch`) is unchanged. The rule, fixed before the run: use the
middle when the walk's mean chord is at least 0.005 spreads and the middle fits every label the repair fits. The
check also scores #63's unconditional arm, which uses the middle whenever it fits as many labels as the repair.
The final labels were not written to disk in this run, only used in it. Results are in `evidence/middle-step/`.

## Results (measured)

| | Under 0.1% | Under 0.01% | Closer / further | Median | Worst | Crossed under / over 0.01% |
|---|---|---|---|---|---|---|
| Repair (current) | 50 | 36 | | 0.0053% | 0.096% | |
| Rule (chord at least 0.005) | 50 | 37 | 6 / 2 | 0.0046% | 0.088% | 224 / none |
| Middle whenever it fits | 50 | **39** | 32 / 15 | 0.0045% | 0.088% | 208, 224, 235 / none |

31 of 50 repairs edit exactly the three broken weights, in every arm, since the step moves only values. 29 cases
stop after one stage, 10 after two and 11 reach the third.

1. **The rule fixed beforehand helps a little.** It uses the middle in 8 cases: 6 end closer and 2 further (200:
   0.0007% to 0.0026%; 237: 0.0064% to 0.0067%). Seed 224 crosses under 0.01% (0.0125% to 0.0088%). None crosses
   over.
2. **The unconditional middle does better on these fresh cases.** It is closer in 32 cases and further in 15, and
   brings 3 more under 0.01% (208: 0.018% to 0.0094%; 224; 235: 0.014% to 0.0093%) with none moving over. Seed 211's
   middle fits fewer labels than its repair, so that arm keeps the repair there.
3. **The width bands from #63 did not hold.** Grouped by mean chord, the geometric-mean change is 0.82 for the 20
   narrowest regions (under 0.002 spreads, 14 closer), 0.99 for the 22 in between, and 0.81 for the 8 widest (6
   closer). In #63 it was 0.96, 0.92 and 0.73.
4. **The two fresh results agree with #63's own unconditional arm.** There the 1,600-step middle was closer in 37 of
   58 and moved four cases under 0.01% and two over. Here it is closer in 32 of 50 with three under and none over.

## Interpretation

- Taking the middle of the region that fits is a small, mostly positive change everywhere, not a fix that only
  works where the region is wide (inferred from 108 cases across #63 and #64). It costs seconds per case and needs
  no new labels.
- It moves only a few cases across the 0.01% line. Of the 11 cases still over 0.01% after it, only 3 edit exactly
  the three broken weights, so most of what is left is a wrong weight among the three, not the values. One more
  round of labels placed where the region is unsure is #65.
