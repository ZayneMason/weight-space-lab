# Put the flat-line step on top of the search, on all 200 cases (roadmap #86)

**Date:** November 27, 2026 · **Status:** done on all 200 cases; finding 611's true broken weights is #87 · **Cost:**
0.13 core-hours recorded (median 1.4 seconds a case), against the roadmap's estimate of under an hour.

After #85 the cases over the 0.01% line across #79's, #80's and #83's 200 cases are 331, 400 and 611, and none of
them was in #84's search, because their picks fit every one of #65's final labels. `flat_stack.py` runs #82's
flat-line step on every case's network after #85, from #65's final labels: the chooser's network in the 27 cases #84
searched (a perfect fit in 8, the pick in 19), the chosen network elsewhere. Where the least-constrained direction of
the margins is flat (singular value up to 1e-6 of the largest), it takes the middle of the segment of perfect fits
along it, then asks 4 rounds of 8 labels along it. The result was fixed in advance as the network after those 32
labels where the direction is flat, and the network before the step elsewhere, as in #83.

```bash
python experiments/locate-repair/flat_stack.py --output runs/f86 --workers 4
python experiments/locate-repair/flat_stack.py --score runs/f86
python experiments/locate-repair/flat_stack.py --truth   # the true broken weights' segment where a case stays over
```

Evidence is in `evidence/flat-stack/`: `flat-stack.json` (every case's network, singular values, segment, rounds and
seconds), `check.json` (the score) and `truth.json` (611's true broken weights at their original values).

## Results (measured)

| All 200 cases | under 0.01% | under 0.1% | exact | median disagreement | over the line |
|---|---|---|---|---|---|
| After #85 | 197 | 200 | 158 | 0.00115% | 331 (0.0621%), 400 (0.0132%), 611 (0.0118%) |
| + the segment's middle | 199 | 200 | 158 | 0.0012% | 611 |
| + 32 labels along the line (the result) | **199** | **200** | **158** | 0.00115% | 611 |

1. **331 and 400 go under the line, as on #81's picks.** 331 goes from 0.0621% to 0.0067% and 400 from 0.0132% to
   0.0012%. No case goes over the line, and the count exact does not change, since the step keeps each network's
   weights.
2. **33 of 200 networks have a flat direction.** The step makes 7 closer (331, 400, 402, 443, 526, 542, 547) and 4
   further (349 0.0018% to 0.0020%, 418 0.0018% to 0.0026%, 428 0.0015% to 0.0034%, 644 0.0010% to 0.0012%), all
   under the line. 443 is one of #84's searched cases whose pick misses a label; the step takes it from 0.0033% to
   0.0016%. Of the 8 perfect fits the search chose, only 444's is flat, and the step leaves it at 0.0010%.
3. **The step reproduces exactly.** On the 194 cases whose network has the same weights as #82's or #83's, its result
   is the same in all 194.
4. **611's chosen network is not on a flat line, and cannot reach the truth.** It edits two of the three broken weights
   (hidden weights 10 and 20) and a wrong third (hidden weight 2). Its smallest singular value is 0.24 of the largest
   and its segment is 0.03 spreads long, so the step leaves it at 0.0118%.
5. **611's true broken weights are on a flat line.** At their original values (hidden weights 10, 20 and 28) they get
   every label right; the smallest singular value is 2e-13 of the largest, the segment runs from -0.056 to 0.168
   spreads, and every network checked along it is under the line (0.0009% at one end, 0.0006% in the middle,
   0.0031% at the other). #83 found the fitter's values for the same triple at 0.0118% (inferred: they lie off this
   segment).
6. **Cost.** 0.13 core-hours for all 200 cases; the step asks 1,056 labels (32 on each of the 33 flat cases).

## Interpretation

- The whole pipeline (#65's stage, #71's chooser, #81's gated duel, #84's search with #85's fitter, #82's flat-line
  step) puts 199 of 200 cases under 0.01% and 158 exact. #82's 150 of 150 was on the cases its step was shaped on;
  here 49 of #83's 50 fresh ones are under the line too, with 641 fixed by #84's search.
- 611 is the last case over the line, and it is a search problem as well as a labels problem: the right triple fits
  every label and sits on a flat line whose networks are under the line, but the search never tries it, because the
  pick fits every label and #84 runs only where it misses one. Whether searching 611 anyway, then the chooser and the
  line step on the true triple's fit, would take it under the line, and what gate could fire there, is #87.
