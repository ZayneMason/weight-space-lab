# Run the fixed repair at width 32 (roadmap #92)

**Date:** November 28, 2026 · **Status:** done on #60's 12 seeds (200 to 211) at H = 32; a rate needs more seeds,
which is #93 · **Cost:** 15.8 core-hours recorded, under the roadmap's estimate of 25 to 50.

Every result since #61 is at H = 16. #60 ran an older repair at H = 32 on seeds 200 to 211: 12 of 12 under 0.1%, 8
under 0.01% and 5 exact. `width_32.py` runs #90's order, fixed in advance, on the same 12 seeds at H = 32:

- #83's first part (#65's stage with its network saved, #71's stage and #81's gated duel), giving the pick;
- #84's keep-one search where the pick misses one of #65's final labels, #89's keep-two search elsewhere, both with
  #85's fitter, then #88's refit duel;
- #82's flat-line step on the duel's winner. The one change from #90, taken from #91: the line step runs on any flat
  winner, not only on a kept pick.

Every stage builds its case as `t3.corrupt(16, seed)`; the script swaps that for one that builds the same seed at
H = 32, so each stage runs unchanged. Seed 200 was the pilot (exact, about 1 core-hour), so all 12 ran.

```bash
python experiments/locate-repair/width_32.py --output runs/w92 --only 200                 # the pilot
python experiments/locate-repair/width_32.py --output runs/w92 --part 0 --parts 4         # first part, parts 0 to 3
python experiments/locate-repair/width_32.py --repair runs/w92 --workers 4                # search, duel, line step
python experiments/locate-repair/width_32.py --score runs/w92
```

Evidence is in `evidence/width-32-repair/`: `width-32-repair.json` (every case's search, candidates, duel, chosen
network, line step and seconds), `check.json` (the score), `first-part/` (each case's first part), `duel/` (each
duel's rounds and labels right) and `labels/` (#65's final labels for each case).

## Results (measured)

| 12 cases at H = 32 | under 0.01% | exact | median disagreement | over the line |
|---|---|---|---|---|
| #60's two-step repair | 8 | 5 | not reported | 4 cases |
| #65's own repair (this run) | 9 | 5 | 0.0029% | 202, 204, 205 |
| The pick (#71, #81) | 12 | 8 | 0.0015% | none |
| Search and refit duel | 12 | 9 | 0.00095% | none |
| Then the line step on any flat winner (the result) | **12** | **9** | **0.00095%** | none |
| #90's order, line step only on a kept pick | 12 | 9 | 0.00095% | none |

| The same order elsewhere | cases | under 0.01% | exact | core-hours a case |
|---|---|---|---|---|
| #90, fresh seeds at H = 16 | 50 | 50 (100%) | 36 (72%) | 0.62 |
| This run, H = 32 | 12 | 12 (100%) | 9 (75%) | 1.32 |

1. **Every case ends under the line, and the pick already does.** #65's own repair leaves 3 over (202 at 0.0140%,
   204 at 0.0230%, 205 at 0.0277% with one broken weight found); #71's and #81's stages bring all three under, and
   205 to exact at 0.0001%. The worst result is 206 at 0.0024%. 211 ends with no disagreement on the million probe
   inputs.
2. **The search and duel move 5 cases closer, 1 slightly further, and gain one exact.** Against the pick, 206, 207,
   208, 209 and 210 get closer (the largest 208, 0.0018% to 0.0006%, and 209, 0.0023% to 0.0005%) and 204 goes from
   0.0010% to 0.0011%. 202 gains the third broken weight; none loses one.
3. **The three results short of exact are two ties and one unreachable triple.** 201 and 209 end with two broken
   weights: the true triple is a perfect fit and ties at the top of the duel with 147 and 156 other candidates, and
   loses on order, as all 14 did in #90. 210's pick finds only one broken weight but gets every label right, so the
   keep-two search runs; a triple keeping two of the pick's weights cannot be the true one, and the result keeps one
   broken weight at 0.0013%.
4. **No case is gated or flat.** No pick misses a label, so the keep-one search never runs, and no duel winner has a
   flat direction, so #91's change to the line step does not show here (the gated order gives the same 12 networks).
5. **The search is twice as wide, the duel no harder.** Each keep-two search fits 478 triples (238 at H = 16), a
   median 207 seconds. 12 of 12 have at least one perfect fit (median 3.5, most 158 for 201). The duels ask 466
   labels in all (median 34.5, most 165 for 209; 200 and 203 have one candidate and ask none).
6. **Cost.** 15.8 core-hours: 14.6 for the first part (median 0.71 core-hours a case; 202, 205 and 210 run all
   three stages and take 2.4 to 2.9 each), 0.7 for the keep-two searches and 0.5 for the duels. That is 2.1 times
   #90's cost a case, and less than #60's about 25 core-hours for the same 12 seeds. The container restarted once
   during the first part; the stages resumed from their saved files, and the recorded seconds may leave out the time
   lost (inferred: under an hour).

## Interpretation

- The fixed repair carries to a network twice as wide without retuning: 12 of 12 under the 0.01% line against #60's
  8, and 9 exact against 5, at less CPU than #60. Exact at 75% is in line with #90's 72% at H = 16, but on 12 seeds
  either rate is loose (inferred: a 95% interval on 9 of 12 runs from about 47% to 91%).
- What stops exact is the same as at H = 16: the duel's ties among refits that are the same function (#91), plus a
  new kind, 210, where the pick has one broken weight and passes every label, so the gate never sends it to the
  keep-one search. At H = 16, #90 had one such result (727, 0.0008%). Neither changes behaviour much (both under 0.002%).
- Width roughly doubles the cost a case, almost all of it in #65's and #71's label stages, not the search.
- Twelve seeds are enough to show the order holds at H = 32, not to give a rate. Running it on 38 more fresh seeds,
  to 50 as in #90, is #93.
