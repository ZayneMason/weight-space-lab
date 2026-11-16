# Mixing the author's guesses with the under-column table (roadmap #19)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 25 minutes

[Roadmap #18](UNDERFIT.md) refitted the rising-ball guesses to the under column. That fixed
the brick rate, but the refitted table still scores 357 against the author's 482. The
author's lead sits in rows 3–5 ([#12](ROWSWAP.md)). This experiment swaps rows between the
two tables to see whether a mix beats both.

```bash
python experiments/breakout-anatomy/combine.py CHECKPOINT --output runs/breakout-combined
python experiments/breakout-anatomy/combine.py CHECKPOINT --output runs/breakout-combined-holdout \
    --episodes 1000 --seed-start 200 --only author "author, row 8 from under"
```

## Method

- Both pure tables, every single-row swap in both directions, three groups of the author's
  rows inside the under-column table (3–5, 2–5, 3–4), and the author's table with rows 3 and
  8 from the under table: 24 variants.
- 200 episodes each at the published sticky 0.1, seeds as in the published protocol, plus the
  no-sticky cycle. All variants use the same seeds, so differences are paired by episode.
- The one swap that beat the author's table by more than two standard errors was rerun on
  1,000 fresh episodes (indices 200 to 1,199).
- Rows 0 and 1 hold the same guesses in both tables (within 0.01 column), so their swaps
  change nothing.

## Results (measured)

Every variant scores 834 without sticky actions. At sticky 0.1, differences are paired against
the table the row came into:

| Row | Author's table, this row from the under table | Under table, this row from the author's |
|---:|---:|---:|
| 2 | −13.2 ± 9.7 | +3.8 ± 9.3 |
| 3 | +5.7 ± 12.9 | +20.4 ± 15.1 |
| 4 | −8.9 ± 21.3 | +25.6 ± 21.7 |
| 5 | **−58.3 ± 20.6** | −12.9 ± 17.9 |
| 6 | +2.2 ± 18.7 | +8.0 ± 18.8 |
| 7 | +0.4 ± 13.0 | −12.0 ± 16.7 |
| 8 | +8.5 ± 3.7 | −4.4 ± 4.2 |

1. **No mix beats the author's table.** The best single swap, row 8 from the under table, was
   +8.5 ± 3.7 on 200 episodes. On 1,000 fresh episodes it was +0.9 ± 1.8 (508.4 against
   507.4), so the first result was most likely chance among 16 comparisons. Rows 3 and 8 together gave
   +9.1 ± 12.8.
2. **Rows 3–5 only work together.** The author's rows 3–5 lift the under table by 88.2 ± 22.3
   points (to 445), against a sum of +33 for the three single swaps. Row 5 alone *lowers* the
   under table (−12.9) yet is the row the author's table can least afford to lose (−58.3).
3. **The under fit adds nothing once rows 3–5 are the author's.** The under table with the
   author's rows 3–5 scores 445.0 ± 18.2. #12's accurate table with the same rows scored 448
   on 300 episodes. Outside rows 3–5, neither fit improves on the author's guesses.
4. Rows 2–5 from the author's table give the same as rows 3–5 (444.7); rows 3–4 alone give
   428.3.

## Interpretation

- *Inference:* the author's rows 3–5 act as one unit. A guess in row 5 positions the paddle
  for a flight whose outcome depends on where rows 3 and 4 then send it, so a row-5 guess
  only pays off with matching guesses below it. This fits the earliest-bounce reading (#9):
  all three rows assume the same full wall.
- The under-column fit's gain over #8's table (#18) is not additive with the author's rows
  3–5. Both improvements fix the same flights.
- None of the 24 variants is better than the author's network. *Inference:* row by row, its
  table looks close to a local optimum among brick-blind tables.

## Limits

- 200 episodes per variant; paired standard errors are 4 to 22 points, so single-row effects
  under about 40 points are not resolved.
- Only the held-out check of row 8 used fresh seeds; the other results share seeds 0–199 with
  #16 to #18.
- #12's 448 comes from a different episode count (300), so the comparison in result 3 is
  approximate.

Evidence: [`evidence/breakout-combined/combined.json`](../../evidence/breakout-combined/combined.json)
(all 24 variants, per-episode scores, paired differences) and
[`evidence/breakout-combined/holdout.json`](../../evidence/breakout-combined/holdout.json).
Code: `combine.py`; 2 tests in `tests/test_breakout_combined.py`.
