# A better order among perfect practice fits (roadmap #57)

**Date:** November 19, 2026 · **Status:** done · **Cost:** CPU, about 11 core-hours (three hours on four cores)

In every case where the rules lose the true triple, it fits every practice label and is cut on the tie-break
among hundreds to thousands of other perfect fits: #51's nine cases (seeds 52 to 96) and #55's three
(124, 126 and 149). The committee keeps 200 triples, ordered by practice accuracy and then by rank in the
coarse 7-point fit. This experiment records features of every fit in those 12 committees and asks whether a
different tie-break would put the true triple inside the first 200.

```bash
python experiments/locate-repair/tie_breaks.py 52 runs/ties 1000   # one run per case, four side by side
python experiments/locate-repair/tie_rank.py
```

`tie_breaks.py` rebuilds a committee the same way `screen_ranks.py` does and saves, for every extended triple:
practice accuracy and total edit size, the coarse-fit rank, its place in the pair screen, its largest single
edit, how many bias entries it edits, and (for perfect fits) the smallest and mean scaled margin over the
practice labels. The 12 lost cases are rebuilt at 1,000 pairs, where #51 and #55 found the true triple is
always extended. Eight seeds the default keeps (100, 101, 103, 105, 106, 108, 112, 113) are rebuilt at 200
pairs as controls. `evidence/tie-breaks/` holds the 20 feature files and `summary.json`.

## Results (measured)

True triple's rank among the committee's order, 1,000 pairs (kept if 200 or better):

| Seed | Perfect fits | Coarse rank (current) | Smallest largest edit | Largest min margin | Best of the seven |
|---|---:|---:|---:|---:|---:|
| 52 | 7,092 | 1,011 | 4,066 | 6,821 | 848 (fewest biases) |
| 55 | 469 | 208 | **38** | **93** | 38 |
| 62 | 6,495 | 430 | 2,832 | 669 | 430 (current) |
| 63 | 1,218 | 253 | **20** | 905 | 20 |
| 66 | 1,646 | 781 | 660 | 347 | **149** (screen order) |
| 68 | 10,783 | 5,298 | 2,031 | 6,971 | 1,329 (fewest biases) |
| 76 | 1,901 | 1,009 | **64** | 774 | 64 |
| 93 | 1,490 | 370 | **86** | **125** | 86 |
| 96 | 3,979 | 506 | **194** | 554 | 194 |
| 124 | 3,736 | 1,441 | 1,242 | 3,510 | 548 (fewest biases) |
| 126 | 9,486 | 9,277 | 6,787 | **114** | 114 |
| 149 | 336 | 298 | 203 | 287 | **80** (largest mean margin) |

Lost cases kept within 200, by tie-break (all eight controls stay kept under every one):

| Tie-break | Lost cases kept (of 12) |
|---|---:|
| Coarse rank (current) | 0 |
| **Smallest largest single edit** | **5** |
| Largest smallest margin | 3 |
| Smallest total edit | 1 |
| Largest mean margin | 1 |
| Pair-screen order | 1 |
| Fewest bias entries | 1 |

1. **The largest single edit is the best single tie-break.** Preferring the perfect fit whose biggest one-weight
   change is smallest keeps 5 of the 12 lost triples (55, 63, 76, 93, 96). The current coarse-fit order keeps
   none, by construction of the case list.
2. **No tie-break keeps all 12.** Taking the best of the seven for each case would keep 8. Seeds 52, 62, 68 and
   124 are cut by every one: the best rank there is 430 to 1,329.
3. **Sharing the 200 places between orders does worse.** Filling the committee in turn from the current order
   and the largest-edit order keeps 4; from four orders, 2; from all seven, 2. This is analysis on the saved
   files, not a rerun.
4. **The controls do not separate the tie-breaks.** All eight keep the true triple within 200 under every
   tie-break. The worst control rank is 79 (seed 105, largest smallest margin). Seed 113 drops from 6 to 15-63.
   So the controls cannot tell which tie-break costs least on cases the default already gets right.

## What this means (inferred)

The true triple is a "small, spread-out" fix. Three faults of size 1.0 to 2.2 are repaired by three edits of
about that size. Wrong perfect fits often lean on one large edit to cover for a missing weight, so capping the
largest edit is a better prior than the coarse fit's opinion. That explains 5 of the 12. The other cases have
thousands of perfect fits that look like the true one on every feature recorded here. Those need more labels
(or better-placed ones), not a different order.

Being kept in the committee is necessary, not sufficient: the repair still has to pick the true triple, or a
fit that behaves like it, from the 200. So the next step reruns the repair with the new tie-break and scores
disagreement. That is [roadmap #58](../../docs/ROADMAP.md).
