# Run the cheap repair on 50 more fresh seeds (roadmap #78)

**Date:** November 26, 2026 · **Status:** done on 50 fresh cases; keeping #65's own repair in #71's chooser is #79 ·
**Cost:** 22.5 core-hours recorded (20.9 for the cheap run on 50 cases, 1.6 for the original rerun on 2 of them),
against the roadmap's estimate of 25 to 30.

#72 ran the whole repair, as it was then, on 50 fresh seeds (300 to 349). #77 showed that the cheap repair (columns
for every refine and the batched committee fits, in #65's stage and #71's) does the same job on 12 more seeds for
under half the compute. This item runs only the cheap repair, with `fresh_cheap.py`, on 50 seeds nothing had been run
on (400 to 449), and compares the totals with #72's. The seeds differ, so the comparison is of rates, not case by
case. The two cases the cheap run left over the 0.01% line were then rerun with the original repair, to see whether
the cheap refines put them there.

```bash
python experiments/locate-repair/fresh_cheap.py --kind cheap --output runs/f78 --workers 4 --only $(seq 400 449)
python experiments/locate-repair/fresh_50.py --score runs/f78
python experiments/locate-repair/fresh_cheap.py --kind original --output runs/f78-orig --workers 2 --only 400 414
python experiments/locate-repair/fresh_50.py --score runs/f78 --original runs/f78-orig
```

Evidence is in `evidence/fresh-50/`: `cheap.json` (each case's #65 stage and #71 stage, with candidates, pick and
seconds), `check.json` (the score next to #72's), `original.json` (the original repair's rows for 400 and 414) and
`vs-original.json` (the comparison).

## Results (measured)

| Whole repair, 50 fresh cases | #72 (original, seeds 300 to 349) | cheap (seeds 400 to 449) |
|---|---|---|
| Core-hours | 37.56 | 20.88 (10.80 in #65's stage, 10.08 in #71's) |
| Core-hours per case, mean | 0.75 | 0.42 (min 0.17, median 0.26, max 1.70) |
| Chosen network under 0.01% / under 0.1% | 49 / 50 | 48 / 50 |
| Exact (all three broken weights edited) | 39 | 39 |
| Median disagreement on a million probe inputs | 0.0010% | 0.0014% |
| #65's repair alone: under 0.01% / exact | 41 / 37 | 41 / 28 |
| Cases #71's stage brought under the line | 8 | 8 |
| Cases #71's stage took over the line | none | 1 (seed 414) |
| Bigger committee ran / labels asked | 5 / 244 | 8 / 322 |

1. **The rate holds.** 48 of 50 chosen networks are under 0.01% and all 50 under 0.1%, against 49 and 50 in #72; 39
   are exact in both. The cost is 1.80x lower in total (1.83x in #71's stage, 1.77x in #65's).
2. **Neither miss comes from the cheap refines.** On seeds 400 and 414 the original repair ends #65's stage with the
   same outcome, gives #71's chooser the same candidates to the sixth digit, makes the same pick and returns the same
   network. It takes 2.17x as long to get there (5,905 seconds against 2,727).
3. **Seed 400 finds the right weights with the wrong values.** All three candidates edit exactly the three broken
   weights, but they disagree with the original on 0.041% to 0.059% of inputs. The chooser asked 25 labels, all
   answered right only by the boundary fit, and took it at 0.041%.
4. **Seed 414 had a good repair and threw it away.** #65's stage ended with an exact repair at 0.0035%. #71's gate
   then ran the bigger committee, and every candidate it builds finds only one of the three broken weights (0.069%, or
   0.19% for the boundary fit). #65's repair is not one of #71's candidates, so the chooser could only pick among
   these. It is the only case, here or in #72's 50, where #71's stage made the repair worse.
5. **A few cases carry the cost.** The 6 cases that needed #65's third stage took 7.5 core-hours (median 1.10), the
   34 that stopped after one took 9.0 (median 0.24). The 8 with the bigger committee took 6.2 (median 0.49).

## Interpretation

- The cheap repair is now checked on 62 fresh cases (#77's 12 and these 50) and on 8 retuned ones. On these 50 it
  matches #72's rates at 1.8x lower cost, and on the two cases it misses the original repair misses in exactly the
  same way. It can be the default.
- #77 measured 2.19x on the same seeds; 1.80x here is across different seeds. This run's cases were cheaper on
  average than #77's (0.42 core-hours against 0.52), so the lower ratio probably reflects #72's seeds costing less
  per case than #77's rather than a smaller speedup (inferred: no original run covers seeds 400 to 449).
- #65's repair alone is exact in 28 of these cases against 37 of #72's. #77 showed the cheap refines can change #65's
  labels (same outcome in 9 of 12), so some of that gap may be the refines and some the seeds; this run cannot
  separate them. #71's stage closes it: the final count is 39 exact in both.
- Seed 414 points to a simple fix: keep #65's own repair as a candidate in #71's chooser. The labels it would be
  scored on already exist. Whether it would have won on 414 is untested, since #65's repair was never scored on the
  32 labels #71 asked. That is #79.
