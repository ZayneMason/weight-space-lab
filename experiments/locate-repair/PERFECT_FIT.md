# Make the fitter reach a perfect fit when one exists (roadmap #85)

**Date:** November 27, 2026 · **Status:** done on #84's 27 cases; putting #82's flat-line step on top of this result
for all 200 cases is #86 · **Cost:** 22.6 core-hours recorded for the rerun (255,987 fits), against #84's 21.1; the
check on the 10 cases took seconds, against the roadmap's estimate of under an hour.

In 10 of #84's 27 cases (312, 317, 328, 348, 419, 443, 448, 609, 630, 647) the broken weights' original values get all
1,096 of #65's final labels right, but the fitter's best values for them miss 1 or 2. `perfect_fit.py` adds one step
after the fitter, run only when a fit misses 1 to 3 labels:

- linearise every label's margin in the three values (finite differences);
- solve a small linear program for the smallest change, in tensor spreads, that puts every linearised margin above
  2e-4, inside a trust region of 0.05 spreads (where that is infeasible, the step that raises the smallest margin
  most);
- keep the step when it leaves fewer labels wrong or raises the smallest margin, otherwise halve the region; up to 40
  rounds. The fit with more labels right is kept, as before.

It then reruns #84's search (every triple that keeps one of the pick's weights, then #71's disagree chooser on the
pick and every perfect fit) on the same 27 cases with the step in every fit.

```bash
python experiments/locate-repair/perfect_fit.py --true   # the 10 cases' true triple, and the 27 picks moved (control)
python experiments/locate-repair/perfect_fit.py --output runs/p85 --part 0 --parts 4   # parts 0 to 3, one per core
python experiments/locate-repair/perfect_fit.py --score runs/p85
```

Evidence is in `evidence/perfect-fit/`: `true.json` (the true triple before and after the step on the 10 cases, and
the control), `perfect-fit.json` (every case's perfect fits, the chooser's duel and seconds) and `check.json` (the
score, with the comparison to #84).

## Results (measured)

| 27 cases where the chosen network misses a label | true triple a perfect fit | chooser takes a perfect fit | under 0.01% | exact | over the line |
|---|---|---|---|---|---|
| The chosen network | | | 26 | 20 | 641 (0.0366%) |
| #84's search | 17 | 5 | 27 | 22 | none |
| + the step in every fit (the result) | **27** | **8** | **27** | **22** | none |

| All 200 cases | under 0.01% | exact | over the line |
|---|---|---|---|
| The chosen network | 196 | 156 | 331, 400, 611, 641 |
| #84's search | 197 | 158 | 331, 400, 611 |
| + the step | 197 | 158 | 331, 400, 611 |

1. **The step closes the fitter's gap.** On the 10 cases the true triple goes from 1 or 2 labels wrong to none in all
   10, ending within 0.038 spreads of the original values (the fitter's were within 0.043). In the rerun the true
   triple is a perfect fit in 27 of 27 (from 17). 12 cases gain perfect fits and none lose any (median 4 a case, from 2; at most 412, in 645).
2. **It makes 3 cases closer and none worse, but changes no outcome.** The chooser now also takes a perfect fit in
   348 (0.0023% to 0.0017%), 630 (0.0019% to 0.0011%) and 647 (0.0015% to 0.0005%). All three picks were already
   exact. The other 5 it takes match #84's. The counts under the line and exact are #84's, here and over all 200.
3. **Where the true triple is now a perfect fit and the pick is not exact, the labels cannot choose.** In 317 the pick
   (two broken weights) and all 6 perfect fits agree on every one of the 20,000 pool inputs, so the duel asks nothing
   and keeps the pick; both are at 0.0019%. In 419 the pick (one broken weight) gets 8 of 12 duel labels right and the
   true triple 7, at 0.0032% against 0.0034%.
4. **A perfect fit can still be over the line, and the duel catches it.** In 443 the only perfect fit is the true
   triple, at 0.0143%. The pick, which misses one label, is at 0.0033%; the duel goes 22 to 1 its way and keeps it.
5. **The step cannot be used on the picks themselves.** Moving each of the 27 picks onto #65's final labels makes 26
   fit every label (641 still misses), but 21 get further from the original network, 4 closer, and 308, 641 and 648
   end over the line. The picks were fitted on more labels than #65's, as #82's control found.
6. **The step runs rarely.** 146 of 255,987 fits miss 1 to 3 labels; the step makes 79 of them perfect.
7. **Cost.** 22.6 core-hours for the 27 cases (median 50 minutes a case), 7% more than #84's 21.1, almost all of it
   the fitter itself. The duel asks 356 labels against #84's 289.

## Interpretation

- #84's 10 gaps were the fitter's, and a small linear step on the margins fixes all of them. With it, the search's
  perfect fits always include the true broken weights in these 27 cases, so a missed label at the pick now means a
  wrong triple, not a fitter shortfall (inferred from 27 cases).
- It does not move the repair's outcome: where it adds the true triple, either the pick is already exact or the true
  triple and the pick behave almost the same on the probe (317, 419), and in 443 it would have been worse. Keep the
  step in the search's fitter, since it is cheap and safe there, but not on the picks.
- The three cases over the line across all 200 (331, 400, 611) were never in the search, because their picks fit
  every label. #82's flat-line step put 331 and 400 under the line on #81's picks; #86 puts it on top of this result
  for all 200 cases, so the whole pipeline's count is measured in one place.
