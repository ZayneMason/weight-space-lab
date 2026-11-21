# Refine the values once the repair has the right weights (roadmap #63)

**Date:** November 21, 2026 · **Status:** done · **Cost:** about 25 core-hours to replay, minutes to score

#62 found that 17 of the 33 width-16 cases over 0.01% in seeds 100 to 199 edit exactly the three broken weights.
Their gap is in the values. The repair's margin fit stops at the first point that fits every label and keeps the
smallest change among equals, so it may sit at the edge of the region of values that fit. This experiment takes
every case whose repair edits exactly the three broken weights (58 cases) and compares the repair's values with
the middle of that region, on the same labels. No new labels and no new search.

```bash
python experiments/locate-repair/refine_values.py --list                                   # the 58 exact cases
python experiments/locate-repair/refine_values.py --replay 164 --output runs/refine-values  # one process per case
python experiments/locate-repair/refine_values.py --score runs/refine-values
```

- **Replay.** The final labelled set (1,024 practice labels plus 64 targeted ones) was not saved by earlier runs.
  `switch_triples.run` now takes a `keep` dict that receives it, and `--replay` reruns each case's kept stage to
  get it back. All 58 replays end at the saved disagreement exactly.
- **Middle of the region.** A hit-and-run walk inside the region of values that miss no more labels than the
  repair does, started at the repair's values: each step takes a random direction in units of each tensor's
  spread, finds how far the region reaches along it by bisection (at most half a spread), and moves to a uniform
  point on that chord. The middle is the walk's mean. Three walks per case: 400 steps after 100 of burn-in, the
  same with other random draws, and 1,600 after 400. For 57 cases the region is the set that fits every label;
  seed 160's repair misses one label, so its region allows one miss.

Results are in `evidence/refine-values/check.json`.

## Results (measured)

| | Under 0.01% (of 58) | Closer / further | Geometric mean of the change | Crossed under / over 0.01% |
|---|---|---|---|---|
| Repair (current) | 41 | | | |
| Middle, 400 steps | 42 | 35 / 21 | 0.88 | 116, 121 / 100 |
| Middle, 400 steps, other draws | 41 | 30 / 26 | 0.96 | 116 / 130 |
| Middle, 1,600 steps | 43 | 37 / 20 | 0.89 | 116, 121, 169, 182 / 100, 130 |

All 58 stay under 0.1% in every arm, and the worst case drops from 0.071% to 0.057%.

1. **The middle of the region is a little closer on average, but the 0.01% line barely moves.** Depending on the
   walk, 0 to 2 more of the 58 cases end under 0.01%, and one or two cases already under it move over (100: 0.0044%
   to between 0.0061% and 0.052%; 130: 0.0078% to between 0.0089% and 0.011%).
2. **The two short walks agree on whether the middle is closer in 43 of 58 cases**, so in the rest the walk's own
   noise is as large as the effect.
3. **The gain is all in cases where the labels leave room.** Grouped by how far the region reaches along the long
   walk's chords (in spreads):

   | Mean chord | Cases | Geometric mean of the change | Closer in every walk |
   |---|---|---|---|
   | Under 0.002 | 29 | 0.96 | 7 |
   | 0.002 to 0.005 | 15 | 0.92 | 6 |
   | 0.005 or more | 14 | 0.73 | 11 |

   The widest regions give the biggest gains: seed 194 goes from 0.071% to between 0.010% and 0.017%, and seed
   164 from 0.016% to 0.013%. The bands were chosen after seeing the results.
4. **Most of the 17 exact misses are not at an edge.** Seven are closer in every walk (116, 139, 146, 148, 164,
   169, 194). In the rest the region is narrow or the middle is no better (114 and 168 are pinned within 0.003
   spreads and do not move). Seed 114's repair overshoots one of the three true changes by 40% while fitting all
   1,088 labels, so those labels cannot tell its values from the true ones.

## Interpretation

- The repair's fit does sit at an edge where the labels leave room, and taking the middle there helps (inferred
  from 14 cases with wide regions, chosen after the fact). Where the region is narrow, the labels already pin the
  values, and what is left over 0.01% needs labels in other places, not a better point among these.
- As a rule for the repair, "take the middle when the region is wide" is cheap (seconds per case on the final
  labels) and worth testing on fresh cases with the threshold fixed beforehand (#64).
