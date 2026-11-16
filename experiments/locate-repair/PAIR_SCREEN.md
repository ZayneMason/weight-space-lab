# A cheap screen for pairs of broken weights (roadmap #23)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 50 minutes (four runs in parallel)

Roadmaps [#21](PAIR_SEARCH.md) and [#22](ACTIVE_QUERIES.md) located #20's two broken
weights by fitting every pair of entries, which only scales to H = 8 (861 pairs). At
H = 32 there are 13,041 pairs. This experiment ranks every pair with a cheap screen and
fits only the best-ranked ones.

```bash
python experiments/locate-repair/pair_screen.py --widths 8 32 --seeds 50 --top 200 1000 --output runs/pair-screen
```

## Method

- #20's cases, H = 8 and 32, 50 seeds each, with 1,024 practice examples.
- Two screens score every pair:
  - **Newton:** the loss drop that a Newton step on the two entries predicts, using a
    logistic loss on the output margin and its Gauss–Newton matrix on the practice set;
  - **coverage:** each entry alone gets the grid value that corrects the most wrong practice
    examples without breaking a right one. A pair scores the share of wrong examples its
    two entries correct between them.
- The true pair's rank is the first pair that counts both faults, mirrors included.
- The coverage screen's top 200 or top 1,000 pairs then go through #21's search: a coarse
  61 × 61 joint fit, the 20 best re-fitted on a 200 × 200 grid, and the winner picked
  by best practice accuracy, then smallest change.

## Results (measured)

| | H = 8 (861 pairs) | H = 32 (13,041 pairs) |
|---|---|---|
| Newton: true pair's median rank / in top 1,000 | 216 / 100% | 2,952 / 26% |
| Coverage: true pair's median rank / in top 1,000 | **40** / 100% | **338** / 72% |
| Coverage top 200, then fits: pass / found pair | 78% / 40% | 70% / 8% |
| Coverage top 1,000, then fits: pass / found pair | 92% / 56% | 72% / 10% |
| #20's two greedy edits: pass / found pair | 30% / 8% | 56% / 4% |

1. **The coverage screen ranks the broken pair about nine times better than the Newton
   screen at H = 32.** The median rank is 338 against 2,952 out of 13,041, and its top
   1,000 contains the broken pair in 36 of 50 cases. The Newton screen is worse than
   coverage at H = 8 too. A local second-order estimate does not predict what a shift
   of 1.6 standard deviations does to a ReLU network.
2. **At H = 8 the screen loses nothing.** Coverage top 1,000 covers all 861 pairs and finds
   the broken pair in 28 of 50 cases, the same rate as #21's exhaustive search (14 of 25).
3. **At H = 32 the screen is no longer the bottleneck.** Of the 36 cases where the
   broken pair passed the screen, 16 lost it at the coarse fit, where other pairs fit
   practice perfectly on the coarse grid and the broken pair just missed. Of the 20 cases
   where it reached the fine fit, it won 5. In all 20, the winner fit the practice set
   perfectly.
4. **Passing keeps rising while finding barely moves.** Screen-then-fit passes 72% of
   H = 32 cases against #20's 56%, and finds the broken pair in 10% against 4%.

## Interpretation

- *Inference:* at H = 32, finding candidates is cheap enough: about 6 seconds per case
  for the coverage screen. The remaining problem is #21's, at a larger scale. Many pairs
  fit 1,024 practice examples perfectly, so ranking by practice accuracy and then smallest
  change picks a look-alike. #22's targeted queries are the obvious next step.
- The coverage screen works because each fault causes its own errors, and each true entry
  alone corrects most of its own. Between them, the two true entries' single edits
  correct a median 94% of the wrong practice examples at H = 8 and 95% at H = 32
  (interquartile range 60–100% and 76–99%, from a separate check of the same 50 cases).
  The Newton screen sees only the gradient at the corrupted weights.

## Limits

- Only the 20 best coarse pairs are re-fitted. A wider refinement would keep more true
  pairs (16 of 36 were lost at that step).
- Coverage uses #7's 1-D grid and a hard no-regression rule for each single edit. Other
  thresholds were not tried.
- 50 cases per width, and one corruption size.

Evidence: [`evidence/pair-screen/pair-screen.json`](../../evidence/pair-screen/pair-screen.json)
(per-case ranks under both screens and fit results). Code: `pair_screen.py`; 3 tests in
`tests/test_pair_screen.py`.
