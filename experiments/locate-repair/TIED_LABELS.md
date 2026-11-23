# Ask labels where the tied members disagree (roadmap #68)

**Date:** November 23, 2026 · **Status:** done · **Cost:** 22.8 core-hours

#67 found that the median case has 4 committee members that fit every one of its 1,096 final labels, and that the
committee's own order is the better way to pick among them. This experiment asks whether more labels, placed where
those tied members disagree, can tell them apart. It runs on #65's 50 width-16 three-fault cases, seeds 250 to 299.

```bash
python experiments/locate-repair/tied_labels.py --output runs/tied --workers 4
python experiments/locate-repair/tied_labels.py --score runs/tied
```

For each case the script rebuilds #66's committee on the 1,096 final labels and repeats #67's refit (same random
starts), so round 0 is #67's `refit-coarse` pick; it reproduces that pick in all 50 cases. Then, while two or more
members still fit the most labels, it runs up to 3 rounds:

- Draw a fresh uniform pool of 20,000 inputs on [-2, 2]².
- Label the 8 inputs where the tied members' networks, each at its fitted values, split most evenly.
- Refit every committee member on all labels (1,104, 1,112, then 1,120), pick by most labels then committee order,
  and score the winner and its middle (#64's walk) on the usual 1M-point probe.

The final value is the middle whenever it fits as many labels as the fit (#64's rule). Three changes from the roadmap
text: the pool is the usual 20,000 rather than 5,000, the split is measured at the fitted values rather than the
middles, and every member is refit rather than only the tied ones. Results are in `evidence/tied-labels/`.

## Results (measured)

| After round | Under 0.1% | Under 0.01% | Exact | Median | Cases still tied |
|---|---|---|---|---|---|
| 0 (#67's refit-coarse) | 48 | 42 | 35 | 0.0029% | 44 |
| 1 | 48 | **44** | 36 | 0.0026% | 42 |
| 2 | 48 | 44 | **37** | 0.0029% | 41 |
| 3 | 48 | 44 | 37 | 0.0027% | 41 |

Against #67's refit-coarse pick, the last round ends closer in 16 cases and further in 8, with a geometric-mean
change of 0.76. Seeds 250 and 255 become exact and none stops being exact; 255 and 287 cross under 0.01% and none
crosses over.

1. **Most ties cannot be broken by labels.** 44 of 50 cases start with a tie, and 41 are still tied after 3 rounds
   (24 more labels). In the median round, none of the 20,000 pool inputs splits the tied members at all: 87 of 127
   rounds found no input where they disagree, and in 19 cases no round ever did. Those tied members are the same
   network to within 1 in 20,000 random inputs.
2. **Where the tied members do disagree, one round helps.** Six cases had 10 or more splitting inputs in round 1
   (255, 258, 260, 261, 281, 287), and all six end closer. Seed 255 goes from one right weight at 0.062% to the exact
   triple at 0.0022%, and 287 from 0.068% to 0.0030%. 258 goes from one right weight to two (0.20% to 0.14%), and
   260 (0.099% to 0.061%) and 261 (0.055% to 0.031%) improve but stay over 0.01%.
3. **The moves the other way are small.** The 8 cases that end further all stay under 0.01% except 296, which is
   0.81% before and after. The largest relative move is 270, 0.0022% to 0.0040%.
4. **Most remaining ties are the size the mirror symmetry predicts.** Only the difference between the two output
   logits matters, so an output-layer edit can be swapped for its mirror into the other logit ([ACTIVE_QUERIES.md](ACTIVE_QUERIES.md),
   #22), and a winner with k output-layer edits has 2^k mirror variants. Of the 41 cases still tied at the end, 28
   tie exactly 2^k members. The other 13 tie more: 253 (175), 254 (200), 265, 267, 268, 270, 273, 277, 281, 282,
   283, 285 (77) and 287. The true triple is among the tied members in 37 of the 41.
5. **What stays over 0.01% is unchanged in kind.** The 6 cases over 0.01% after round 3 are 258, 259, 260, 261, 269
   and 296. 258, 261 and 296 have no true triple in the committee; 259's true triple fits fewer labels than the
   winner; 260 and 269 are exact and over the line on their values alone.

## Interpretation

- Targeted labels are the right tool for the few ties between genuinely different networks, and where the tied
  members agree they only add 8 labels that change nothing, so no case got much worse. But they barely move the
  totals (2 more cases under 0.01%, 2 more exact) because almost every tie is between
  members that compute the same function on random inputs (measured on the pool; that they are identical
  everywhere is inferred).
- In 28 of the 41 remaining ties the count matches the winner's mirror variants, which no label can separate and
  which "exact" already counts as found (inferred from the counts; the tied members themselves were not saved).
  Those ties need no breaking. The 13 cases that tie more members than mirrors explain are the open question:
  which other weights the extra members edit, and whether they are the same network everywhere or only on the
  pool. That is #69.
