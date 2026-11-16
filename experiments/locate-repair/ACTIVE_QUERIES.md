# Asking where the suspects disagree (roadmap #22)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 30 minutes (four runs in parallel)

[Roadmap #21](PAIR_SEARCH.md) searched every pair of entries for #20's two broken weights at
H = 8. It picked the broken pair in 14 of 25 cases. In most misses another pair fit all
1,024 practice examples just as well and needed a smaller change. This experiment adds
examples chosen where the candidate repairs disagree, labelled by the teacher, and
compares them with the same number of random examples.

```bash
python experiments/locate-repair/active_queries.py --seeds 25 --output runs/active-queries
```

## Method

- #20's cases at H = 8, seeds 0–24. #21's all-pairs search gives 20 candidate pairs per
  case, each finely re-fitted.
- Eight rounds. In each round every candidate pair is re-fitted (200 × 200 joint grid) on
  the practice set plus all queries so far. The winner has the best accuracy, and the
  smallest total change breaks ties.
- **Targeted** queries: of 5,000 fresh uniform inputs, the 8 on which the candidates within
  0.5% of the best accuracy split their votes most evenly. **Random** queries: 8 uniform
  inputs. Both get their labels from the teacher, up to 64 queries in all.
- Round 0 is #21's result, and it reproduces #21's 14 of 25.

## Results (measured)

| Queries | Targeted: found pair / pass | Random: found pair / pass |
|---:|---|---|
| 0 | 14 / 23 | 14 / 23 |
| 8 | 15 / 24 | 14 / 23 |
| 16 | 16 / 24 | 14 / 23 |
| 32 | 18 / 25 | 14 / 24 |
| 48 | **20** / 25 | 14 / 24 |
| 64 | **20** / 25 | 14 / 24 |

Counts are out of 25 cases. "Found pair" means both broken entries, counting exact mirrors.

1. **64 targeted examples do what 1,024 random ones did not.** The broken pair wins in 20 of
   25 cases after 64 targeted queries, up from 14. With 64 random queries it stays at 14.
   Of the 6 cases that flipped, the flip came after 8, 16, 24, 32, 48 and 48 queries.
2. **The remaining misses are mostly outside the candidate list.** In 4 of the 5 misses,
   the broken pair was not among the 20 candidates, so no query could select it. In the
   other one (seed 14), a pair of a different hidden bias and an output bias kept fitting
   every query.
3. **What stays tied is the exact mirror symmetry.** In every case that found the pair,
   the number of pairs tied at the best accuracy after 64 queries is exactly 2 to the
   power of the number of output-layer faults (1, 2 or 4). Each output weight or bias can
   be swapped for its mirror into the other logit, and no input can tell them apart.
4. **The disagreements are rare inputs.** In the first round the targeted inputs split the
   committee a median 33% to 67%. From round 3 on, in the median case no input among the
   5,000 fresh ones split the candidates within 0.5% of the best accuracy.

## Interpretation

- *Inference:* #21's underdetermination is a sampling problem, not an identifiability
  limit, apart from the mirror symmetry. The candidate repairs differ only on a small set of
  inputs that uniform practice data seldom hits, and a few dozen labels in that set settle
  most cases.
- A benchmark or an editor that may query the original model can therefore score
  localization, not just repair, at modest cost. The cost moves to finding the candidates:
  an all-pairs search, which scales with the square of the parameter count.

## Limits

- H = 8 only, 25 cases. The all-pairs candidate search is too slow for H = 32 in this script.
- The committee is limited to 20 candidates from #21's coarse search, and 4 of 25 cases
  lacked the broken pair there.
- The query pool is 5,000 uniform inputs per round. Late rounds often find no
  disagreement in it, so a larger pool or a search along the decision boundary might
  settle more cases.

Evidence: [`evidence/active-queries/active-queries.json`](../../evidence/active-queries/active-queries.json)
(per-case histories for both query modes). Code: `active_queries.py`; 2 tests in
`tests/test_active_queries.py`.
