# Ask the next labels where the region of values is still unsure (roadmap #65)

**Date:** November 22, 2026 · **Status:** done · **Cost:** 33.3 core-hours

#64 found that moving a repair's values to the middle of the region that fits its labels is a small gain almost
everywhere. This experiment adds one more round of 8 labels after the repair and asks where to put them: where the
committee splits most evenly (the usual rule, one more round of it), or where networks sampled from the middle
step's walk split most evenly. It runs on 50 fresh width-16 three-fault cases, seeds 250 to 299.

```bash
python experiments/locate-repair/early_values.py --early 2 --two-step --middle --extra \
    --only 250 --output runs/region/s250                         # one process per seed 250 to 299, four at a time
python experiments/locate-repair/early_values.py --early 2 --two-step --middle --extra \
    --merge runs/region/s*/early-values.json --output runs/region/merged
python experiments/locate-repair/region_check.py
```

`--extra` runs after `--middle`'s walk on each stage's finished repair. It draws one fresh uniform pool of 5,000
inputs and picks 8 of them two ways:

- **Committee splits:** the inputs where the committee's members disagree most evenly, as in every earlier round.
- **Region splits:** the inputs where 200 evenly spaced points of the 1,600-step walk, each a full set of values
  for the winner's three weights that fits every label, disagree most evenly.

Each arm labels its 8 inputs, refits every committee member on all 1,096 labels, picks the winner again and takes
that winner's middle. The final value is the middle whenever it fits as many labels as the arm's own fit (#64's
unconditional rule). The repair itself (`switch`) is unchanged. This time the labels are on disk: each case's
1,088 final labels and both arms' 8 extra inputs are in `evidence/region-labels/labels/case-SEED.npz`. Results are
in `evidence/region-labels/`.

## Results (measured)

| | Under 0.1% | Under 0.01% | Closer / further than the repair | Geometric-mean change | Median | Crossed under / over 0.01% |
|---|---|---|---|---|---|---|
| Repair (current) | 47 | 36 | | | 0.0050% | |
| Middle, no new labels (#64) | 48 | 37 | 32 / 17 | 0.83 | 0.0037% | 255, 272, 281 / 276, 290 |
| 8 committee-split labels | 48 | 37 | 17 / 9 | 0.79 | 0.0040% | 260 / none |
| 8 committee-split labels, then middle | **49** | **41** | 36 / 13 | 0.65 | 0.0035% | 255, 260, 272, 274, 281 / none |
| 8 region-split labels | 48 | **41** | 35 / 7 | 0.54 | 0.0033% | 255, 262, 269, 272, 274 / none |
| 8 region-split labels, then middle | 48 | **41** | **42 / 8** | 0.56 | **0.0026%** | 255, 262, 272, 274, 281 / none |

30 of 50 repairs edit exactly the three broken weights, and every arm keeps that count at 30. 33 cases stop after
one stage, 9 after two and 8 reach the third.

1. **Region-split labels move the values further than committee-split labels.** Both arms ask 8 labels from the
   same pool. Against each other, the region arm with its middle ends closer in 32 cases and further in 9. Against
   the repair, it is closer in 42 of 50 with none moving over either line, and the typical error drops by about
   44% (geometric mean 0.56), against 35% for the committee arm and 17% for the middle alone.
2. **The committee has little left to disagree about.** In the median case only 1 of the 5,000 pool inputs splits
   the committee at all, so most of its 8 picks are chosen by the tie-break. The walk's sampled networks split on a
   median of 3 inputs. Without the middle, committee-split labels change only 26 cases.
3. **The middle step adds nothing on top of region-split labels.** Region labels alone already give a geometric
   mean of 0.54. Taking the middle afterwards trades 269 (back over 0.01%) for 281 (under). After committee-split
   labels the middle still helps (0.65 with it), which fits the committee's labels leaving the region wide.
4. **Neither kind of label fixes a wrong weight.** The winner changes in 13 cases with committee splits and 20 with
   region splits, but always to another triple with the same number of right weights, except seed 279's committee
   arm (one right weight to none). Seed 260, the one exact case where committee splits help a lot (0.125% to
   0.0007%), barely moves with region splits (0.120%).
5. **What is left over 0.01% is mostly the committee.** 9 cases stay over 0.01% after the region arm: 2 exact (260,
   269) and 7 that edit at least one wrong weight. None of those 7 committees contains the true triple, so no
   number of labels on that committee could find it. Across all 20 inexact cases, 15 committees lack the true triple
   and 13 cases end under 0.01% anyway. Seed 296 stays near 0.8% in every arm.

## Interpretation

- Labels chosen where the region of fitting values is still unsure are the better use of 8 more labels at the end
  of a repair: they beat one more committee round and the free middle step, and moved no case over 0.01% here
  (inferred from 50 cases; the margin over the committee arm in the 0.01% count is zero, the gain is in how close).
- The remaining failures are mostly a committee that never held the true triple, not too few labels for the
  values. The final labels are now on disk, so the next step can rebuild the committee from them without rerunning
  the repair. That is #66.
