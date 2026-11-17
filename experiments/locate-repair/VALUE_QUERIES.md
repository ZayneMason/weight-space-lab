# Queries that pin down the winning repair's values (roadmap #38)

**Date:** November 17, 2026 · **Status:** done · **Cost:** CPU, about an hour and a half (four runs
in parallel, four arms per case)

[#37](BEHAVIOUR_SCORES.md) found four two-fault repairs that edit the exact broken weights and still
disagree with the original network on 0.13% to 0.17% of inputs, because their values are a few
hundredths of a spread off. This experiment tries to close that gap on the same 25 random-size
two-fault cases.

```bash
python experiments/locate-repair/value_queries.py --output runs/value-queries
```

## Method

#35's setup throughout: H = 32, random fault sizes in [1.0, 2.2] spreads, the 25-point pair screen,
then 8 rounds of 8 labels. After every round the winner is probed on a million fresh inputs, as in
#37. Four arms, each a change to #37's loop:

| Arm | Labels each round | Pair fit |
|---|---|---|
| `value` | 4 value queries, 4 targeted | 200 x 200 grid, ties broken toward the corrupted value (as #37) |
| `midtie` | 8 targeted (as #37) | 200 x 200 grid, tied point nearest the middle of the tied region |
| `refine` | 8 targeted | grid fit, then a 61 x 61 grid within 1.5 grid steps of it, middle of the tie there |
| `refine_value` | 4 value queries, 4 targeted | as `refine` |

- A **value query** is an input where the current winner and a nudged copy of it disagree (one of
  its two values moved 0.01 spreads up or down, taking the four copies in turn).
- **Why refine.** A first look at seed 5 showed that the grid point nearest the true values still
  disagrees on 0.17%. The grid's step is 0.025 spreads, so no labels could get the grid fit closer.
- The `value` arm's first round equals #37's (checked for all 25 cases), so #37 is the baseline.

## Results (measured)

**After all 64 labels, 25 cases:**

| Arm | Exact broken entries | Every visible fault | **Under 0.1%** | **Under 0.01%** | Median disagreement | Better than #37 (cases) |
|---|---:|---:|---:|---:|---:|---:|
| #37 baseline | 19 | 23 | 20 | 9 | 0.015% | |
| `value` | 18 | 22 | 21 | 10 | 0.015% | 9 |
| `midtie` | 19 | 23 | 20 | 7 | 0.027% | 8 |
| `refine` | 19 | 23 | 22 | 13 | 0.0089% | 15 |
| `refine_value` | 18 | 22 | **24** | **20** | **0.0016%** | **24** (0 worse) |

**Round by round, under 0.1% and under 0.01%:**

| Labels | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `value`: under 0.1% | 11 | 17 | 17 | 19 | 20 | 20 | 20 | 22 | 21 |
| `refine`: under 0.1% | 10 | 17 | 18 | 22 | 22 | 22 | 22 | 22 | 22 |
| `refine_value`: under 0.1% | 10 | 19 | 21 | 23 | 24 | 24 | 24 | 24 | 24 |
| `refine_value`: under 0.01% | 1 | 3 | 9 | 17 | 18 | 20 | 22 | 22 | 20 |

1. **Neither change works alone; together they do.** Value queries on the grid fit end where #37
   did (21 against 20). The refined fit alone helps a little (22, 13). Both together reach 24 of 25
   under 0.1% and 20 of 25 under 0.01%, and the median falls about ninefold, to 0.0016%.
2. **The paired comparison is one-sided.** `refine_value` ends below #37's disagreement in 24 of 25
   cases and above it in none (seed 18 is level at 0.0002%).
3. **#37's four right-weight misses are fixed.** Seeds 5, 8, 12 and 20 end at 0.029%, 0.0016%,
   0.018% and 0.011%, all on the exact pair.
4. **One case still fails.** Seed 23 picks a wrong weight in every arm and ends at 0.10% to 0.16%.
5. **The middle of the tie is worse on the coarse grid.** With 1,024 labels and no queries, only 4 of
   25 `midtie` winners are under 0.1%, against 11 at the edge. It never catches up (20 at the end).
6. **Value queries cost a little weight-finding.** Arms that give half the labels to value queries
   find every visible fault in 22 of 25, against 23. Seed 3's `refine_value` winner is on a wrong
   weight yet disagrees on only 0.0059%.
7. **Cost:** median 198 seconds per arm per case for the refined arms, against 179 for the grid fit,
   including the million-input probe after every round.

## Interpretation

- *Inference:* the leftover two-fault error in #37 was mostly the fit's resolution, not the labels.
  Once the fit can resolve finer than 0.025 spreads, labels aimed at the winner's values are what
  move it. On the coarse grid they have nothing to move.
- *Inference:* the edge-of-tie rule works better than the middle at the start because the corrupted
  value is a useful prior: with few labels the tied region is large, and its middle can sit far from
  the truth. On the fine grid the region is small enough that the middle helps.
- *Inference:* value queries and targeted queries answer different questions (which values, which
  weights), and splitting the labels between them trades a little of one for a lot of the other here.
  A schedule that switches to value queries only once the committee agrees may keep both. This was
  not tested.
- These are 25 cases from one run per arm, so differences of one or two cases are within noise. The
  24-to-0 paired result is not.

## Evidence

- `evidence/value-queries/value-queries.json`: every case, every arm, every round.
