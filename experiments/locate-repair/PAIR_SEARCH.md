# Searching pairs of weights for two faults (roadmap #21)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 50 minutes (three runs in parallel)

[Roadmap #20](TWO_FAULTS.md) broke two entries in different hidden units. Told which two,
a joint search over both values passed the benchmark in 96% of cases. Untold, two greedy
edits found both in 4–8%. This experiment ranks pairs of entries instead of one entry
at a time, to see whether the practice data can identify the broken pair at all.

```bash
python experiments/locate-repair/pair_search.py --widths 8 32 --seeds 50 --output runs/pair-search
python experiments/locate-repair/pair_search.py --widths 8 --seeds 25 --all-pairs --output runs/pair-search-all
```

## Method

- Same cases, 1,024 practice examples and success rule as #20.
- **Single-edit ranking:** every entry is scored by its best single-edit practice accuracy.
  A fault's rank is where it, or its exact mirror, falls in that list.
- **Shortlist search:** every pair among the top 8 or top 16 entries gets a joint 61 × 61
  grid (each value within 2.5 tensor spreads of its current one). The 20 best pairs are
  re-fitted on #20's 200 × 200 grid. The winner has the best practice accuracy, with the
  smallest total relative change breaking ties.
- **All pairs:** the same search over every pair of the 42 entries at H = 8 (861 pairs,
  25 cases; larger widths are too slow for this script).

## Results (measured)

| Search | H = 8: pass / found both / median damage undone | H = 32 |
|---|---|---|
| top 8 entries | 46% / 12% / 0.89 | 54% / 0% / 0.89 |
| top 16 entries | 64% / 24% / 0.97 | 60% / 0% / 0.92 |
| all pairs (25 cases) | **92%** / 56% / 0.99 | not run |
| #20, told, joint | 96% / 100% / 0.99 | 96% / 100% / 0.99 |

1. **A fault's single-edit score barely predicts that it is a fault.** At H = 32 the
   median broken entry ranks 39th of 162, and both broken entries make the top 16 in 1 of
   50 cases. At H = 8 the median rank is 6th of 42, and both make the top 16 in 17 of 50.
   A shortlist built from single edits therefore rarely contains the pair, and the
   shortlist searches found both faults in 0 of 50 cases at H = 32.
2. **Searching every pair passes almost as often as being told, but finds the pair half
   the time.** At H = 8 the all-pairs search passes 23 of 25 cases (told: 96%) and picks
   the broken pair in 14.
3. **The practice data does not single out the broken pair.** Of the 11 cases where the
   all-pairs search missed, the broken pair fit the practice set as well as the winner
   in 9. Another pair tied it and needed a smaller change. In the other 2, a wrong pair
   fit practice strictly better than the broken pair could on the grid. Among the 20
   finely re-fitted pairs, a median of 3 fit practice perfectly, and in 3 cases all 20 did.
4. **Shortlist searches pass as often as #20's greedy repair without finding more.** At
   H = 32, top-16 pair search passes 60% against greedy's 56%, finding both faults in 0
   cases against 2.

## Interpretation

- *Inference:* with two faults, 1,024 behavioural examples leave the repair
  underdetermined. Several pairs of entries reproduce the teacher on every practice
  example, and the smallest-change rule, which #7 used to choose among single edits,
  prefers a look-alike pair whenever it needs less change than the true repair (each true
  shift is 1.6 spreads).
- Finding broken weights from input–output data alone needs either more targeted data
  (examples in the region where the candidate repairs disagree) or a prior other than
  smallest change. Passing the benchmark rule does not need either.

## Limits

- All pairs only at H = 8 and on 25 cases; at H = 32 a full search (13,041 pairs) was not run.
- Only the 20 best coarse pairs are re-fitted, so "perfect pairs" is a lower bound capped at 20.
- One corruption size; model editors were not run.

Evidence: [`evidence/pair-search/`](../../evidence/pair-search/) (`shortlist-h8.json`,
`shortlist-h32.json`, `all-pairs-h8.json`, with per-case rows). Code: `pair_search.py`;
2 tests in `tests/test_pair_search.py`.
