# Choose among the members that fit every label (roadmap #67)

**Date:** November 23, 2026 · **Status:** done · **Cost:** 21.1 core-hours

#66 rebuilt each committee on its 1,096 final labels (seeds 250 to 299). The true triple was then in 46 of 50
committees, but the repair's pick edited exactly the three broken weights in only 30. This experiment rebuilds the
same committees, keeps every member's fit, and picks the winner six ways from the same members, to see whether a
better tie-break or a better fit closes that gap.

```bash
python experiments/locate-repair/tie_break_final.py --output runs/tie --workers 4
python experiments/locate-repair/tie_break_final.py --score runs/tie
```

Every rule first keeps the members that fit the most labels, then breaks the tie:

- **cost:** the smallest total scaled edit. This is the repair's rule and #66's winner.
- **coarse:** the committee's own order, which ranks by the coarse 7-value fit. This is #66's first member.
- **largest-edit:** the smallest largest single scaled edit (#57, #58).
- **refit-:** the same three rules after refitting every member within 1% of the best accuracy from 5 more starts
  (its fitted values plus a standard normal step per weight in its layer's scale), keeping each member's best fit.

The final value is the winner's middle (#64's walk) when the middle fits as many labels, else the fit. The run also
records how many members tie at the top, and, as a check on the fit, whether the true triple at the original
network's values fits every label (information the repair does not have). Results are in
`evidence/tie-break-final/`.

## Results (measured)

| Rule | Exact | Under 0.1% | Under 0.01% | Median | Closer / further than cost | Geometric-mean change |
|---|---|---|---|---|---|---|
| cost (the repair's rule) | 30 | 46 | 42 | 0.0029% | | |
| coarse | 34 | 47 | 42 | 0.0029% | 12 / 12 | 0.95 |
| largest-edit | 34 | 47 | **43** | 0.0029% | 14 / 11 | 0.94 |
| refit-cost | 31 | 47 | 42 | 0.0028% | 12 / 7 | 0.97 |
| refit-coarse | **35** | **48** | 42 | 0.0029% | 15 / 11 | **0.92** |
| refit-largest-edit | **35** | **48** | 42 | 0.0029% | 17 / 12 | 0.97 |

The worst case is 0.81% under every rule (seed 296, whose committee lacks the true triple).

1. **The tie-break moves the exact count, not the behavior.** Ranking ties by the coarse fit or by the largest
   single edit instead of the total edit finds the right three weights in 34 cases instead of 30, and the refit
   takes that to 35. But the 0.01% line moves by at most one case, the median does not move, and the typical
   disagreement drops by 3% to 8%. 37 cases are exact under at least one of the six rules.
2. **The labels never rule the true values out.** In all 50 cases the true triple at the original network's
   values fits all 1,096 labels. So every time its fit falls short, the fit is at fault, not the labels.
3. **More starts fix half of those short fits.** Without the refit, the true triple ties the top in 42 of the 46
   committees that hold it. Five more starts per near-best member lift it to the top in 255 and 269, leaving 259
   and 277 short.
4. **Most remaining ties are between near-identical networks.** The median case has 4 members tied at the top, and
   2 cases have all 200 tied. After the refit, the true triple ties the top in 9 cases where refit-coarse still
   picks another member. 8 of those 9 end under 0.01% anyway, most with 10 to 200 members tied. The one that
   costs something is 255: 3 members tied, and the pick has one right weight and ends at 0.062%.
5. **What is left over 0.01% has four causes.** Under refit-coarse, 8 cases stay over 0.01%. In 3 (260, 269, 287)
   the repair edits the right three weights but their values are off. In 3 (258, 261, 296) the committee lacks the
   true triple. In 255 the tie-break picks wrong, and in 259 the true triple's fit still falls short.

## Interpretation

- Among the members that fit every label, the coarse order is a better tie-break than the total edit for finding
  the exact broken weights, and five more fitting starts help a little more (inferred from 50 cases; the gap is
  4 to 5 exact repairs, with no case moving over 0.01%). It is a cheap change to the repair's last step.
- For how closely the repaired network behaves like the original, the choice hardly matters: most tied members
  are as good as each other on a million inputs, which fits #39's finding that "wrong" weights can still repair
  the network.
- The labels themselves cannot separate tied members. Seed 255 is the case where that costs something, so the
  next step asks labels where the tied members disagree. That is #68.
