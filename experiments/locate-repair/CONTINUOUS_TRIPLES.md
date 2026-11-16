# A continuous fit for every three-fault suspect (roadmap #33)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 95 minutes (four runs in parallel)

[Roadmap #32](DENSE_SCREEN.md) found that the coarse grid limits the three-fault suspect list. On
[#31](RANDOM_SIZES.md)'s random-size cases, the true triple ranked below 1,000 on an 11³ grid in 8 of
25 cases, so refining the top 1,000 never reached it. This experiment drops the grid as a filter and
gives every extended triple a continuous fit.

```bash
python experiments/locate-repair/continuous_triples.py --seeds 25 --output runs/continuous-triples
```

## Method

- #31's random-size cases (H = 16, 25 seeds, each of the three faults between 1.0 and 2.2 spreads).
- #32's 25-point pair screen picks the 200 pairs that #27 extends by every other entry: about
  10,000 to 13,000 triples per case. Seed 14 has no true triple among them.
- Every triple gets #29's Nelder–Mead margin fit twice, once from its current values and once from
  its coarse 7³ optimum. The better of the two by practice accuracy, then smallest change, is kept.
- Triples are ranked by practice accuracy. Ties go to the smallest change, or, as a variant, to the
  coarse-7 order. The table reports how often the true triple, mirrors included, is in the top 40,
  100 and 200.

## Results (measured)

True triple's position among the extended triples, 25 cases:

| Ranking | Top 40 | Top 100 | Top 200 |
|---|---:|---:|---:|
| coarse 7 | 5 | 8 | 10 |
| continuous fit, ties by smallest change | 9 | 14 | 16 |
| continuous fit, ties by coarse-7 order | 13 | 17 | **21** |
| #32: coarse 11 then refined (for comparison) | 14 | 17 | 17 |

1. **The fit is no longer the limit.** With the continuous fit, the true triple fits all 1,024
   practice examples in 23 of the 24 cases that contain it. In seed 13 it misses one example.
2. **But so do many others.** A median of 160 triples per case fit practice perfectly (quartiles
   60 and 459, up to 6,332). The true triple is one of them, and practice accuracy can't pick it
   out. That is why top-40 recall stays at 9–13 while top-200 recall reaches 21.
3. **Smallest change is a bad tiebreak here.** In 11 of the 24 cases, the fit that reproduces
   practice on the true entries needs noticeably less total change than the actual bugs (by more
   than 0.1 spreads). Under smallest-change ties, the true triple ranks behind perfect triples that
   need less change still. Breaking ties by the coarse-7 order works
   better: 21 of 25 in the top 200, against 16.
4. **It costs more.** A median of 826 seconds per case, about twice #32's 412.

## Interpretation

- *Inference:* for three faults the remaining problem is the one #21 found for two: the practice
  set leaves the repair underdetermined. Dozens to hundreds of triples reproduce every practice
  label. Ranking them on practice data alone can't find the true one. #22's answer still applies:
  ask the teacher for labels where the candidates disagree. The candidate list should now be the
  perfect-fit set, or the top 200, not the top 40 (roadmap #34).
- The fit's values for the true entries are often not the bug's values, since a smaller change on
  the same entries also reproduces practice. "Found the entries" and "found the values" are
  separate questions here.

## Limits

- 25 cases at H = 16, one random draw of sizes. The suspect lists were not taken through targeted
  examples.
- Two starts per triple. Some fits may still stop short; the true triple failed to fit in 1 case.
- The pair stage still caps recall: seed 14 has no true triple to find.

Evidence: [`evidence/continuous-triples/continuous-triples.json`](../../evidence/continuous-triples/continuous-triples.json)
(per-case positions, perfect-fit counts, the true triple's fitted change and its real total size). Code:
`continuous_triples.py`; 2 tests in `tests/test_continuous_triples.py`.
