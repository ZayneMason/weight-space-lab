# Denser screens for bugs of any size (roadmap #32)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 80 minutes (four runs in parallel)

[Roadmap #31](RANDOM_SIZES.md) found that the coarse screens worked partly because their grids
had a point next to the 1.6-spread corruption. With sizes drawn between 1.0 and 2.2, #26's pair
pipeline found every visible fault in 20 of 25 cases instead of 25, and the 7-point triple ranking
put the true triple in the top 40 in 5 of 25 instead of 17. This experiment keeps #31's cases and
makes the grids denser, so that every size in the range sits near a grid point.

```bash
python experiments/locate-repair/random_sizes.py --mode pairs --points 25 --seeds 25 --output runs/dense-screen
python experiments/locate-repair/dense_screen.py --seeds 25 --output runs/dense-screen
```

## Method

- #31's random-size cases: the same entries, signs and seeds as #26 (two faults, H = 32) and #27
  (three faults, H = 16), each fault between 1.0 and 2.2 spreads.
- **Pairs:** #26's pipeline with a 25 × 25 joint screen instead of 13 × 13. No size in the range is
  more than 0.10 spreads from a grid point, against 0.21 before.
- **Triples:** the 25-point pair screen picks the 200 pairs that #27 extends. The extended triples
  are ranked by the coarse 7³ fit (#28's), a coarse 11³ fit (worst gap 0.25 spreads, against
  0.42), and the coarse-11 top 1,000 refined with #29's margin fit.
- Everything else, including #22's targeted examples for pairs, is unchanged.

## Results (measured)

**Two faults, H = 32, 25 cases, after 64 targeted examples:**

| Pair screen | True pair in top 40 | Found pair | Found every visible fault | Pass | Median seconds per case |
|---|---:|---:|---:|---:|---:|
| 13 points (#31) | 17 | 16 | 20 | 23 | 221 |
| **25 points** | 19 | 19 | **23** | 23 | 322 |
| #26, size 1.6 (flattered) | 21 | 21 | 25 | 25 | |

**Three faults, H = 16, 25 cases, true triple in the top 40:**

| Ranking | #31 (13-point pairs) | 25-point pairs |
|---|---:|---:|
| coarse 7 | 5 | 5 |
| coarse 11 | | 8 |
| refined top 1,000 (coarse-7 order in #31, coarse-11 here) | 11 | **14** |
| #30a, size 1.6 (flattered): refined | 19 | |

1. **For pairs, the denser screen recovers most of the loss.** Every visible fault is found in 23
   of 25 cases, against 20 with the 13-point screen and 25 on the flattered cases. The cost is about
   45% more time per case. The two misses (seeds 11 and 23) have the true pair at ranks 84 and 89,
   outside the 40 candidates.
2. **For triples, density helps less.** The 11-point grid puts the true triple in the top 40 in 8
   cases, against 5. Refining its top 1,000 reaches 14, against 11, still short of the flattered 19.
3. **The coarse triple rank is erratic in both directions.** In 8 of 25 cases the coarse-11 grid
   ranks the true triple below 1,000, so refinement never sees it. In seed 4 the 7-point grid ranks
   it 108th and the 11-point grid 8,204th. A grid has to come close to all three sizes at once, and
   a denser grid doesn't guarantee that.
4. **The pair stage can drop a true triple.** In seed 14 the true triple is not among the triples
   extended from the 25-point screen's best 200 pairs. It was there with the 13-point screen.

## Interpretation

- *Inference:* for two faults, a 25-point joint screen makes the pipeline roughly independent of
  the bug size, at 23 of 25. That is the number a post can use.
- For three faults, the coarse grid is the wrong tool. Its ranking depends on how close the grid
  happens to come to all three sizes together, and making it denser costs a lot without fixing
  that. The next step is to drop the coarse grid and give every extended triple #29's continuous
  margin fit (roadmap #33).

## Limits

- 25 cases per run, one random draw of sizes per case.
- The triple suspect lists were not taken through #29's targeted examples.
- The 25-point pair screen also changes which pairs feed the triples, so the triple comparison
  mixes two changes.

Evidence: [`evidence/dense-screen/`](../../evidence/dense-screen/) (`dense-screen-pairs.json`,
`dense-screen-triples.json`). Code: `dense_screen.py` and `random_sizes.py --points`; 2 tests in
`tests/test_dense_screen.py`.
