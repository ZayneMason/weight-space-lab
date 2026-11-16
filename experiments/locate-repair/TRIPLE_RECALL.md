# Getting the true triple onto the suspect list (roadmap #30, part a)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 50 minutes (two runs of four in parallel)

[Roadmap #29](WARM_TRIPLES.md) found every visible fault whenever the true triple was a
candidate and its fit could track it. In 8 of 25 cases, though, the true triple never made #27's
coarse top 40. This experiment asks how to rank #27's extended triples so the true one ranks
higher. It found a better ranking and also a problem with the old one: #27's coarse grid works
mostly because one of its points nearly equals the size of the corruption.

```bash
python experiments/locate-repair/triple_recall.py --seeds 25 --output runs/triple-recall
python experiments/locate-repair/triple_recall.py --span-check --seeds 25 --output runs/triple-recall
```

## Method

- #27's cases (H = 16, 25 seeds) and its extended triples: the joint pair screen's best 200 pairs,
  each extended by every other entry, about 10,000 to 14,000 triples per case.
- Each triple is scored on the 1,024 practice examples by:
  - **coarse 7:** #27's 7³ grid, each value within 2.5 tensor spreads of its current one. This is
    what #28 and #29 used;
  - **coarse 9:** a 9³ grid over the same range;
  - **refined:** the coarse-7 top 1,000, each refined with #29's Nelder–Mead margin fit;
  - **span check:** the 7³ grid over ±2.0 and ±3.0 spreads instead of ±2.5.
- Triples are ordered by practice accuracy. Ties go to the smallest change, as in #28, or, as a
  variant, to the coarse-7 order. The table reports how often the true triple, mirrors
  included, is in the top 40, 100 and 200. One case (seed 14) has no true triple among the
  extended triples at all.

## Results (measured)

| Ranking | Top 40 | Top 100 | Top 200 | Median position |
|---|---:|---:|---:|---:|
| coarse 7 (#28's committee) | 17 | 20 | 21 | 23 |
| coarse 9 | 8 | 9 | 11 | 364 |
| coarse 7, span ±2.0 | 4 | 5 | 8 | 652 |
| coarse 7, span ±3.0 | 3 | 6 | 8 | 624 |
| refined, smallest change breaks ties | 14 | 20 | 22 | 19 |
| **refined, coarse-7 order breaks ties** | **19** | **21** | **22** | **5.5** |

Counts are out of 25 cases. The median is over cases where the true triple was ranked.

1. **A finer coarse grid ranks the true triple far worse.** With 9 points instead of 7, the true
   triple makes the top 40 in 8 cases instead of 17. Its own coarse accuracy is lower with 9
   points in 20 of 24 cases.
2. **The 7-point grid works because it nearly contains the corruption.** Each fault shifts its
   entry by exactly 1.6 spreads. The 7-point grid over ±2.5 has a point at ±1.67, 0.07 spreads
   away. The 9-point grid's nearest points are at ±1.25 and ±1.88. A 7-point grid over ±2.0
   (nearest ±1.33) or ±3.0 (nearest ±2.0) puts the true triple in the top 40 in only 4 and 3
   cases.
3. **Refining the top 1,000 helps, if ties keep the coarse order.** After refinement a median of
   29 triples per case fit the practice set perfectly (none did on the coarse grids). Among these
   ties, the smallest-change rule prefers look-alikes and drops top-40 recall to 14. Keeping the
   coarse-7 order among ties raises it to 19, with a median position of 5.5.

## Interpretation

- *Inference:* #27–#29's coarse screen was tuned to this benchmark by accident. A coarse grid
  that happens to contain the corruption size scores the true triple higher than a look-alike
  can reach. With any other spacing the advantage mostly disappears. Real faults have no fixed
  size, so the 17-of-25 recall that #28 and #29 started from should not be expected elsewhere.
- The same coincidence probably helps earlier experiments too. #26's 13-point joint pair screen
  over ±2.5 has a point at ±1.67, and so do #21's 61-point grid and #20's 200-point told fit (to
  within 0.01). Whether #26's 25-of-25 result survives a different corruption size has not been
  tested. That is now roadmap #31.
- The refined ranking depends on the coarse-7 top 1,000 too, so it inherits part of the same
  advantage.

## Limits

- H = 16, 25 cases, one corruption size (1.6 spreads), which is the point of result 2.
- The refinement starts from the coarse-7 grid point and is run only on the coarse-7 top 1,000.
- The targeted-example run with the refined top 100 (roadmap #30, part b) was not run. It would
  measure a ranking that result 2 shows is partly an artefact.

Evidence: [`evidence/triple-recall/triple-recall.json`](../../evidence/triple-recall/triple-recall.json)
(per-case positions under every ranking) and [`span-check.json`](../../evidence/triple-recall/span-check.json).
Code: `triple_recall.py`; 3 tests in `tests/test_triple_recall.py`.
