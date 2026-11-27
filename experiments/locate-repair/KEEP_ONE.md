# When the chosen fit misses a label, search the triples that keep one of its weights (roadmap #84)

**Date:** November 27, 2026 · **Status:** done on the 27 of 200 cases whose chosen network misses a label; a fitter
that reaches a perfect fit when one exists is #85 · **Cost:** 21.1 core-hours recorded for 255,987 fits, against the
roadmap's estimate of a few (a fit takes about 0.3 seconds, but a case has about 9,500 triples, not 3,240).

In #83's seed 641 the true broken weights fit all of #65's final labels and land under the 0.01% line, while every one
of #71's candidates edits a wrong triple that misses some. `keep_one.py` takes the chosen network of all 200 cases
(#81's pick on #79's 100 and #80's 50, the gated pick on #83's 50), counts the labels of #65's final set it gets
wrong, and in each of the 27 cases where it misses at least one:

- fits every triple that keeps one of the pick's three weights and adds any two of the other 81 parameters, on #65's
  final labels (median 9,481 triples a case);
- keeps the triples that fit every label and scores each one's network on the probe;
- runs #71's disagree chooser (up to 32 labels where the networks disagree, from #71's pool) on the pick and every
  perfect fit, with the pick first so a tie keeps it.

```bash
python experiments/locate-repair/keep_one.py --prepare runs/l83   # copy #83's final #65 labels to the evidence
python experiments/locate-repair/keep_one.py --output runs/k84 --part 0 --parts 4   # parts 0 to 3, one per core
python experiments/locate-repair/keep_one.py --score runs/k84
python experiments/locate-repair/keep_one.py --gap   # the true triple's fit where it is not among the perfect fits
```

Evidence is in `evidence/keep-one/`: `keep-one.json` (every case's pick, perfect fits with their edits and
disagreement, the chooser's duel and seconds), `check.json` (the score) and `gap.json` (the fitter check). `--prepare`
copied #83's final #65 labels into `evidence/fresh-line/labels/`.

## Results (measured)

| 27 cases where the chosen network misses a label | under 0.01% | exact | over the line |
|---|---|---|---|
| The chosen network | 26 | 20 | 641 (0.0366%) |
| + search and chooser (the result) | **27** | **22** | none |
| Best perfect fit on the probe (not available to the repair) | 27 | 22 | none |

| All 200 cases | under 0.01% | exact | over the line |
|---|---|---|---|
| The chosen network | 196 | 156 | 331, 400, 611, 641 |
| + the search where it misses a label | **197** | **158** | 331, 400, 611 |

1. **641 is fixed exactly.** One triple of the 9,481 fits every label, and it is the true broken weights. The chooser
   asks 32 labels where it and the pick disagree (62 split inputs); the pick gets none right and the true triple all
   32. Disagreement goes from 0.0366% to 0.0027%.
2. **The chooser takes a perfect fit in 5 cases and makes none worse.** 315 (0.0017% to 0.0014%), 444 (0.0056% to
   0.0010%, two broken weights to three), 641, 642 (0.0028% to 0.0011%) and 648 (0.0077% to 0.0037%). In the other 22
   it keeps the pick. Its picks match the best perfect fit on the probe in count under the line and exact.
3. **Every pick keeps at least one broken weight**, so the true triple is in every search (27 of 27). It is among the
   perfect fits in 17. The perfect fits per case: median 2, at most 379 (645, where the pick is already at 0.0001%).
4. **In the other 10 the fitter, not the labels, stops it.** In 312, 317, 328, 348, 419, 443, 448, 609, 630 and 647
   the broken weights' original values get all 1,096 labels right, but the fitter's best values for the true triple
   get 1 or 2 wrong. 8 of these 10 picks already edit the three broken weights and miss 1 or 2 labels the same way.
   9 of the 10 have no perfect fit at all.
5. **Cost.** 21.1 core-hours for 27 cases (0.78 a case, median 47 minutes, almost all fitting), against 0.45 to
   0.48 core-hours a case for the whole repair on #80's and #83's seeds. 289 labels asked in all.

## Interpretation

- The search does what 641 needed and is safe: 1 case moves under the line, 2 more become exact, none get worse. It
  runs only where the pick misses a label, which is 27 of 200 cases here.
- "The pick misses a label" is mostly a fitter signal, not a search one: in 10 of 27 even the true weights cannot be
  fitted to every label, though their original values fit them all (inferred: `fine_fit`'s local grids and polish
  stop short of a thin region; the true values are not on the grid). A fitter that reaches a perfect fit when one
  exists would make the gate sharper and could make more picks exact; that is #85.
- The cost is high for one fix. Most of it is triples with two random extra weights; ranking the extra pairs by the
  pick's gradients, or stopping at the first perfect fit that includes the pick's best two weights, should cut it
  (inferred, not measured).
