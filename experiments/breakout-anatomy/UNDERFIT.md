# Fitting the guesses to the right column (roadmap #18)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 10 minutes

[Roadmap #17](RETURNS.md) found that the accurate ascending table loses bricks through corner
catches, where the paddle is under the column the ball is about to enter rather than the one
it is in. Roadmap #8 fitted that table by distance to the nearest *catch column*, and
MinAtar's paddle survives in either column. So the fit counted a corner catch as a hit. This
experiment refits the table with three definitions of the right column and scores each.

```bash
python experiments/breakout-anatomy/underfit.py CHECKPOINT --output runs/breakout-underfit
python experiments/breakout-anatomy/underfit.py CHECKPOINT --output runs/breakout-underfit --errors
```

## Method

- The 184,502 rising-ball states of 50 published-network episodes (#8's fitting episodes).
- For every cell, the target column (on a half-column grid) with the least mean distance to:
  - **nearest**: either catch column, which is #8's fit. Refitted here, it matches #8's table
    on all 180 cells;
  - **under**: the column the ball is in just before it reaches the paddle row;
  - **corner**: the column it moves into.
- Each table replaces the author's guesses in every row of the published network. Scored
  without sticky actions (one episode, since every episode follows one cycle) and at sticky
  0.1 (200 episodes, seeds as in the published protocol), with #17's catch recorder.

## Results (measured)

| Table | Score, sticky 0 | Score, sticky 0.1 | Corner catches | Bricks per return | Deaths per 10,000 steps |
|---|---:|---:|---:|---:|---:|
| Author's guesses | 834 | 481.6 ± 17.3 | 2.0% | 0.90 | 0.79 |
| Nearest catch column (#8) | 569 | 296.4 ± 14.7 | 6.7% | 0.80 | 1.59 |
| Under column | **834** | **356.7 ± 16.3** | 3.7% | 0.88 | 1.41 |
| Corner column | 602 | 305.8 ± 14.7 | 12.4% | 0.82 | 1.49 |

Sticky-0.1 columns are means over 200 episodes with standard errors; corner share, bricks and
deaths are at sticky 0.1.

1. **Fitting to the under column fixes the game without noise.** It scores 834 without
   sticky actions, the author's number, and makes no corner catches there.
2. **With sticky actions it closes a third of the gap.** The under fit scores 357 against
   296 for #8's table and 482 for the author's: 60 points, about 2.7 standard errors
   (unpaired), and a third of the distance to the author's table.
3. **What is left is safety on short flights.** On returns that turn at row 4, the under fit
   dies 5.13 times per 1,000 against the author's 2.36. At row 3 the gap is smaller (1.00
   against 0.69).
4. **Corner catches come from a parked paddle.** For every table, 95% to 97% of corner
   catches had the paddle standing still one column ahead of the ball on the step before.
   99% of under catches had it standing still on the ball's column.
5. **Fitting to the corner column nearly doubles the corner catches without lowering the score**
   (306 against 296). Corner share alone does not predict the score; which cycle the ball
   falls into matters too.

6. **The author's table is still the least accurate by the new measure.** On #8's held-out
   episodes (32,125 rising-ball states), its guesses are 1.90 columns from the under column on
   average, against 1.56 for #8's table and 1.46 for the under fit. The under fit is only 0.1
   column closer than #8's table by its own measure, yet scores 60 points more.

## Interpretation

- *Inference:* #8's "accurate" table was accurate about the wrong thing. Its error measure let
  the paddle wait at either catch column, and waiting at the corner column breaks the brick
  cycle. A better objective (the under column) is worth 60 points, with no hand-tuning.
- The author's table is still 125 points ahead. That gap is the safety bet of #12, #13 and #15:
  the paddle parked early for the shortest flights. A table can be accurate about the right
  column and still not make that bet.
- The series now separates the author's advantage into two parts that can be fixed one at a
  time: aim for the under column (brick rate), and anticipate the earliest bounce (safety).

## Limits

- 200 episodes per table; the 60-point gain rests on standard errors of about 15 per table.
- All three fits use the published network's own trajectories; the under table might do
  better refitted on its own play (#11 found that refitting did not help #8's table).

Evidence: [`evidence/breakout-underfit/underfit.json`](../../evidence/breakout-underfit/underfit.json)
(the three tables and every result) and
[`evidence/breakout-underfit/errors.json`](../../evidence/breakout-underfit/errors.json). Code: `underfit.py`; 2 tests in
`tests/test_breakout_underfit.py`. `anatomy.ball_future` now also returns the under and
corner columns.
