# Search the triples that keep two of the pick's weights on every case (roadmap #89)

**Date:** November 27, 2026 · **Status:** done on all 200 cases; the same repair, fixed in advance, on fresh seeds
is #90 · **Cost:** 8.0 core-hours recorded, 4.6 of them the search, against the roadmap's estimate of about 4 for the
search plus the duels.

#88's refit duel fixed 611, but only because #87 ran #84's search there by hand: 611's pick fits every one of #65's
final labels, so the gate (a missed label) never fires. `keep_two.py` runs a search cheap enough for every case, then
the duel and the flat-line step, on all 200 of #79's, #80's and #83's cases:

- where the pick misses a label (#84's 27 cases), the candidates are #85's perfect fits from #84's search, as in #88;
- elsewhere (173 cases), it fits every triple that keeps two of the pick's three weights and changes any third
  parameter (238 a case, the pick's own triple among them) with #85's fitter on #65's final labels, and keeps those
  that fit every label;
- it runs #88's refit duel on the pick and the candidates;
- it runs #82's flat-line step on the duel's winner, as #86 did.

The result was fixed in advance as the network after both steps.

```bash
python experiments/locate-repair/keep_two.py --output runs/t89 --workers 4
python experiments/locate-repair/keep_two.py --score runs/t89
```

Evidence is in `evidence/keep-two/`: `keep-two.json` (every case's search, candidates, duel, chosen network, line
step and seconds) and `check.json` (the score against the pick and against #86).

## Results (measured)

| All 200 cases | under 0.01% | exact | median disagreement | over the line |
|---|---|---|---|---|
| The pick (#79, #80, #83) | 196 | 156 | 0.0012% | 331, 400, 611, 641 |
| After #86 (search where gated, then the line step) | 199 | 158 | 0.00115% | 611 (0.0118%) |
| Keep-two search and refit duel, before the line step | 198 | 159 | 0.0007% | 331, 400 |
| Then the line step (the result) | **200** | **159** | **0.0007%** | none |

1. **611 is found and fixed without a hand.** Its keep-two search fits 238 triples and finds the same 2 perfect fits
   #87's 9,481 did, the pick's own and the true triple; the duel takes the true triple at 0.0008%, as in #88, and the
   line step leaves it at 0.0020%. The worst case after both steps is 331, at 0.0067%.
2. **The 27 gated cases reproduce #88.** On #84's 27 cases and 611 the duel's winner and result match #88's on all
   28.
3. **The search finds candidates almost everywhere.** 170 of the 173 keep-two cases have at least one perfect fit
   (median 3, most 153); the true triple is among them in 159. In 30 the pick has two broken weights, and the true
   triple fits every label in 27 of them; in 534, 606 and 639 it does not, and those stay at two broken weights,
   under the line.
4. **The duel takes another triple often, and mostly gets closer.** It takes a candidate over the pick in 142 of 200
   cases (123 of them keep-two). Against #86, 103 cases get closer and 31 further, none crosses the line; the largest
   rises are 544 (0.0023% to 0.0041%), 522 (0.0010% to 0.0025%) and 606 (0.0018% to 0.0030%). The largest falls
   after 611 are 325 (0.0056% to 0.0002%), 426 (0.0063% to 0.0019%) and 430 (0.0051% to 0.0007%).
5. **Exact moves both ways.** 10 cases gain the third broken weight (306, 311, 433, 438, 510, 524, 528, 548, 611,
   614) and 9 lose one (332, 439, 440, 517, 522, 535, 543, 633, 640). In 5 of the 9 the new network is closer to the
   original than the exact one it replaced (for example 517 from 0.0010% to 0.0001%), and in 4 further.
6. **The line step helps the pick and hurts the refits.** It runs on the 38 networks with a flat direction, and
   moves 15: 5 closer and 10 further. It is what takes 331 (to 0.0067%) and 400 under the line, both picks the duel
   kept. 8 of the 10 it pushes further are refitted winners, among them 547 (0.0016% to 0.0053%), 438 and 611.
7. **Labels.** 7,702 asked by the duels (197 cases; median 14, most 256 for 645, 4 ask none) and 1,216 by the line
   step (32 on each of 38).
8. **Cost.** 8.0 core-hours: 4.6 for 41,174 keep-two fits (about 96 seconds a case) and 3.4 for the duels and line
   steps, 0.7 of that 645's 412 candidates.

## Interpretation

- A search cheap enough to run everywhere removes the hand from #88: all 200 cases are under the 0.01% line with no
  case picked by eye, at 159 exact against #88's 160 by hand (the difference is the 9 lost and 10 gained in finding
  5, not 611).
- "Exact" and "closest" part ways here. The duel picks by labels, and in 5 of the 9 cases that lose a broken weight
  the network it picks is closer to the original than the exact one, the same pattern as #39 and #69: a wrong triple
  can match the original about as well as the right one.
- Read after the fact, the line step should run only where the duel keeps the pick (inferred, not run as a fixed
  arm): on these 200 that would leave all 200 under the line, 8 cases closer and 2 further (449, 526). The line step
  was built for the pick's values, and a refit on asked labels already sits where the labels pin it.
- Everything since #84 was shaped on these 200 cases. Running the whole repair, with this order fixed in advance, on
  fresh seeds is #90.
