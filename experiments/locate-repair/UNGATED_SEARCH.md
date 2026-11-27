# Search 611 even though its pick fits every label (roadmap #87)

**Date:** November 27, 2026 · **Status:** done on 611; asking the labels where the pick and the true triple's fit
disagree, and refitting both, is #88 · **Cost:** 1.0 core-hours recorded for 9,481 fits, against the roadmap's
estimate of about one.

611 is the last of the 200 cases over the 0.01% line after #86. Its pick edits hidden weights 10 and 20, two of the
three broken weights, and a wrong third (hidden weight 2), fits all 1,096 of #65's final labels and lands at 0.0118%,
so #84's search, gated on a missed label, never ran there. `ungated_search.py` runs that search anyway with #85's
fitter (every triple that keeps one of the pick's weights, on #65's final labels, split over 4 processes), then #71's
disagree chooser on the pick and every perfect fit, then #82's flat-line step on the chooser's network. As a check
the repair cannot make, it also runs the line step on every perfect fit and measures how far each is from the
original values.

```bash
python experiments/locate-repair/ungated_search.py --output runs/u87 --part 0 --parts 4   # parts 0 to 3, one per core
python experiments/locate-repair/ungated_search.py --score runs/u87
```

Evidence is in `evidence/ungated-search/case-611.json`: the perfect fits with their edits' distance from the original
values, the chooser's duel, the line step on the chosen network and on each perfect fit, and seconds.

## Results (measured)

| 611 | broken weights found | disagreement | fits every label |
|---|---|---|---|
| The pick (the result before and after) | 2 | **0.0118%** | yes |
| Perfect fit: the pick's own triple refitted on #65's labels | 2 | 0.0282% | yes |
| Perfect fit: the true triple (hidden weights 10, 20, 28) | 3 | 0.0160% | yes |
| The true triple's fit after the line step (not chosen) | 3 | 0.0161% | |

1. **The search finds the true triple, and it is over the line too.** 2 of the 9,481 triples fit every label: the
   pick's own and the true one. The fitter's values for the true triple are 0.058, -0.132 and -0.0007 spreads from
   the original values and land at 0.0160%, worse than the pick.
2. **The chooser cannot tell them apart.** The pick and the two perfect fits disagree on 67 of the 20,000 pool
   inputs; on the 32 it asks, the pick and the true triple's fit each get 20 right and the pick's refit 5. The tie
   keeps the pick, at 0.0118%.
3. **The true triple's fit is on a flat line, but not the one through the original values.** Its least-constrained
   direction is flat and the line step runs, but the segment's middle is at 0.0144% and after 32 labels 0.0161%. At
   the original values #86 found a flat segment whose networks are all at 0.0031% or less, so the labels leave a
   region around the true values that is wider than one line (inferred from the two segments and the fit's distance).
4. **The pick moves a weight that is not broken by 2.78 spreads.** Its hidden weight 2 sits 2.78 spreads from its
   original value, and its two broken weights 0.21 and -0.50 spreads from theirs; it still fits every label.
5. **Cost.** 1.0 core-hours for 9,481 fits; the step after the fitter ran on 2 and fixed none.

## Interpretation

- Searching 611 regardless of the gate does not help, so there is no gate to look for here: the right triple is
  found and fits every label, but #65's 1,096 labels do not pin its values close enough to the original, and the
  pick and its rival score the same on the chooser's labels.
- What separates them is where they disagree. A chooser that asks every one of those 67 split inputs, adds them to
  the labels and refits both triples before comparing them would give the true triple's values a chance to move
  toward the original (inferred, not measured); that is #88.
