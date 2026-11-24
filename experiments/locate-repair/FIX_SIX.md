# Fix the six cases still over 0.01% (roadmap #70)

**Date:** November 24, 2026 · **Status:** done · **Cost:** 5.9 core-hours

#69 showed that every member tied at the top is within 0.01% of the winner, so ties cost nothing. What keeps the
repair off the 0.01% line on #67's 50 width-16 three-fault cases (seeds 250 to 299) is 6 cases #68 left over it: 258,
259, 260, 261, 269 and 296. This experiment tries three fixes on them, each scored against #68's final pick on the
usual 1M-point probe.

```bash
python experiments/locate-repair/fix_six.py --output runs/six --workers 4
python experiments/locate-repair/fix_six.py --score runs/six
```

For each case the script first reruns #68 exactly (it reproduces #68's final pick in all 6), then runs:

- **Bigger committee** (only the 3 cases whose committee lacks the true triple: 258, 261, 296): twice the committee
  and twice the pairs kept, #67's restarts, then the usual coarse pick. It also records where the best pair of two
  true faults ranks in the committee's own pair order.
- **Restarts:** refit the winner, and the true triple when it is in the committee, from 20 random starts each, and
  pick again.
- **Boundary labels:** 4 rounds, each labelling the 16 inputs closest to the fitted network's decision boundary out of
  a uniform pool of 20,000 on [-2, 2]², then refitting the committee (64 labels in all).
- **True check** (a yardstick, not a fix): fit the true triple from its broken values, then 20 restarts, on #68's
  labels, to see how well the right weights can do on those labels at all.

Results are in `evidence/fix-six/`.

## Results (measured)

Probe disagreement with the original network, and faults found out of 3:

| Seed | #68 | Bigger committee | Restarts | Boundary labels | True triple on #68's labels |
|---|---|---|---|---|---|
| 258 | 0.139% (2) | **0.0019% (3)** | 0.139% (2) | 0.102% (2) | 0.0047%, fits every label |
| 259 | 0.024% (2) | | **0.0027% (3)** | 0.018% (2) | 0.0035%, fits every label |
| 260 | 0.061% (3) | | 0.061% (3) | 0.044% (3) | 0.065%, fits every label |
| 261 | 0.031% (2) | 0.021% (2) | 0.031% (2) | 0.021% (2) | 0.37%, fits 97.1% of labels |
| 269 | 0.014% (3) | | 0.014% (3) | **0.0017% (3)** | 0.023%, fits every label |
| 296 | 0.809% (2) | 0.807% (2) | 0.809% (2) | 0.745% (2) | 0.037%, fits every label |

1. **Three of the six are now under 0.01%, each by a different fix.** In 258 the bigger committee brings the true
   triple in (its best true pair ranks 1,117th of 3,321, inside the 2,000 now kept) and it wins: 0.139% to 0.0019%.
   In 259 the true triple was already in the committee but lost the fit; 20 restarts let it win, 0.024% to 0.0027%.
   In 269 the repair was already exact, and 64 boundary labels move its values from 0.014% to 0.0017%.
2. **No fix made any case worse.** Every arm's final value is at or below #68's in all the cases it ran on. Boundary
   labels help in all 6, though only 269 crosses the line.
3. **In 260 the labels are the limit.** Even the true triple fitted to #68's labels reaches only 0.065%, no better
   than the repair's own 0.061%. Boundary labels take it to 0.044%, still over the line.
4. **In 261 the true triple does not fit the labels.** Fitted from its broken values and 20 restarts, it fits only
   97.1% of #68's labels and sits at 0.37% on the probe, worse than the two-fault repair's 0.021%. The bigger
   committee still lacks it.
5. **In 296 the right weights would fix it but the committee never sees them.** The true triple fits every label and
   would reach 0.037%, but its best pair ranks 168th and the doubled committee of 400 still lacks the triple. It is
   the most expensive case: 2.3 core-hours.

## Interpretation

- The six leftovers are not one problem. 258 and 296 are committee problems (the true triple is not in it), 259 is a
  fit problem, 269 and 260 are value problems, and 261 looks like a fit problem for the true triple itself, since it
  cannot fit #68's labels from its broken values (inferred; the script does not show why).
- Taking the best arm per case would put 47 of the 50 cases under 0.01%, but that picks the arm by the probe, which a
  real repair cannot see. It is an upper bound on these three fixes, not a result. A rule that chooses among the arms
  from labels alone is not tested here; that is #71.
- Boundary labels are the cheapest fix that helps everywhere (measured on these 6); the bigger committee is the most
  expensive and helps only where the true pair ranks inside the doubled cut.
