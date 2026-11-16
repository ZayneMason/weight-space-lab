# The hand-written network's wrong guesses were a bet

*Draft, November 16, 2026. Not yet published.*

A language model once wrote a Breakout-playing neural network by hand: 815 nonzero
weights, typed out directly, no training. It scores 502.9 on the standard MinAtar
protocol. It also explained how it works. Part of that explanation looked wrong when I
checked it: its guesses for where a rising ball will land are off by almost two columns on
average.

Over ten experiments I tried to fix those guesses, and every fix made the network worse.
Part of the damage turned out to come from my own yardstick for "accurate". The rest came
from something the guesses do on purpose: they bet on the one bounce that kills.

## The network and what its author said

The network comes from Takehi et al.'s weight-space programming experiments. It is a
600-32-32-6 tanh policy. Its author described it as a program:

- four first-layer units compute "landing-position errors", one for each ball direction:
  where the ball will reach the paddle row, minus where the paddle is;
- two units detect the ball's horizontal and vertical direction;
- the second layer picks one of four contexts from those two units and compares, so the
  paddle moves left or right toward the landing column.

The landing column is written straight into the weights. Each input pixel for "ball at
row *r*, column *c*" connects to a landing unit with weight 0.02 × (target column), and
the paddle's position subtracts 0.02 × (paddle column). I can read the network's
prediction for every ball position directly off the weights.

I wrote an exact replica of MinAtar's ball physics, tested against the real environment
on every step of six episodes, to get the true landing columns. Then I played 20 episodes
(158,482 states) and compared.

The description holds up well:

- The selected context matches the ball's true next direction 99.4% of the time.
- For a **falling** ball, the predicted landing column is within half a column of a true
  catch column 99.2% of the time, and 100% from row 3 down.
- One step before the ball reaches the paddle, the paddle is already on a catch column
  99.4% of the time.

The exception is the **rising** ball. There the prediction is within half a column only
42% of the time, with a mean error of 1.76 columns. The network has no weights at all on
the brick channel. It cannot see bricks, so it cannot know where a rising ball will
bounce. Its rising predictions are guesses.

One more fact frames everything that follows. With the environment's sticky actions
turned off, the network never misses: 1,000 of 1,000 episodes reach the 10,000-step
ceiling, each scoring exactly 834. Every one of the 483 deaths at the published setting
needs a sticky action, where the environment repeats the previous move instead of the
one the network chose. The network's only enemy is that noise.

## Better guesses, worse network

The obvious experiment: replace the rising guesses with better ones and see whether the
network survives longer. I changed only the weights of the two rising-ball units and left
everything else alone. All scores below are means over 1,000 episodes at the published
protocol.

| Rising guesses | Mean error on held-out episodes (columns) | Score |
|---|---:|---:|
| the author's | 1.78 | **502.9** |
| perfect (fed the true landing column) | 0 | 480.6 |
| best brick-blind table, fitted on held-out play | 1.12 | 294.1 |

Perfect prediction did not help: 22.3 ± 9.2 points *below* the author's guesses on the
same seeds. A brick-blind lookup table that cuts the error by a third lost 209 points and
more than doubled the death rate. Part of that loss, it turned out, came from how I
measured error; I come back to this near the end.

Almost every death was a last-moment arrival: in 97 of 100 deaths traced across these
variants, the policy chose a move on the very step the ball reached the paddle row. A sticky action on that
step is fatal.

## The guesses are physics

So what did the author's guesses get right? Its description had said it: the rising units
"anticipate the earliest possible brick bounce." Every episode starts with bricks in rows
1–3. I computed, from the physics alone, where a rising ball would land if that wall were
still complete, so the ball bounces off the first brick it meets.

All 180 of the author's rising guesses (90 ball positions, two directions) sit on one of
those catch columns. A table built from this rule without looking at a single episode
scores 493.0 ± 7.9, 98% of the author's score.

My first hypothesis was that this hedges a deadline: the earliest bounce leaves the least
time, so guessing it keeps the paddle safe in the worst case. I fitted a table to minimize
how late the paddle could arrive, assuming it waits at the guessed column until the ball
turns down. It came out identical, cell for cell, to the accurate table that scores 294.
Under that model lateness never binds. A table fitted to hit the exact landing column more
often did better, at 398.2, but still left a gap of about 100 points.

## It is not distribution shift

The accurate table was fitted on states the author's network visits. Installing it
changes how the game goes, so it might simply be fitted to the wrong states. I closed the
loop: install a table, play, refit on that play, repeat three times from three starting
tables.

The shift is real. The accurate table is 1.11 columns off on the author's play but 1.42
off on its own. It isn't the explanation, though. No refit scored above 340 on 300
episodes, and refitting the author's own table once on its own play dropped it from 497
to 297. Fitting the rising guesses for landing accuracy, as I measured it, is what costs the
score, whatever the play distribution.

## Three rows

Next I swapped guesses one ball row at a time between the author's table and the
accurate one, scoring each mix on 300 episodes.

Rows 0–2 and 6–8 hardly matter. No single swap there moves the score more than 22 points.
Rows 3–5, the bottom row of the brick wall and the two rows below it, carry almost
everything:

- the accurate guesses in rows 3–5 only (54 of 180 cells) drop the author's network from
  497 to 284, no better than the all-accurate table (297);
- the author's guesses in rows 3–5 only lift the accurate table from 297 to 448, about
  three quarters of the way back.

## The bounce that kills

Finally I traced every falling flight, about 295,000 of them across 300 episodes per
table. For each one I recorded where the ball turned down, how far the paddle was from the
catch at that moment, and whether the flight ended in a death.

| Ball turned down at | Author's: deaths per 1,000 flights | Accurate: deaths per 1,000 flights |
|---|---:|---:|
| the ceiling (row 0) | 0.06 | 0.02 |
| under row 1 or 2 bricks | 0.32 to 0.65 | 0.59 to 2.24 |
| under the wall's bottom row | **2.14** | **5.44** |

About 70% of all deaths under both tables follow a bounce off the bottom row of the wall,
which leaves the paddle about five steps. Ceiling bounces almost never kill: 3 deaths in
75,519 flights. On those short flights the accurate table starts the paddle 1.52 columns
from the catch, against 0.99 for the author's, and dies 2.5 times as often.

That is the bet. A ball rising through rows 3–5 will bounce straight back if the brick
above it is still there. That gives the shortest flight in the game, the one where a
single sticky action is fatal. The author's guesses assume that bounce. When they are
wrong, it is because the brick is gone, the ball flies on to the ceiling, and the paddle
has plenty of time. The accurate table splits the difference and is out of position for
exactly the flight that kills.

## Which way the paddle is moving

One result above did not fit. On flights that turn down under row-2 bricks (row 3), the
accurate table starts the paddle closer to the catch (0.76 against 0.94 columns) yet dies
3.4 times as often, with five or more steps to spare. Following each flight step by step
explained it: direction, not distance. Under the accurate table, 17% of those flights begin
with the paddle moving away from the catch, against 2% for the author's. Those flights die
7.4 times per 1,000 and account for 36 of the 64 row-3 deaths.

The same trace shows the bet from the other side. At row 4, 43% of the author's flights
start with the paddle already parked on the catch, and only one of those 21,757 flights
ended in a death. The two tables also die differently. 92 of the author's 155 deaths are
out of reach: the ball turned down with fewer steps left than the paddle's distance. 189 of
the accurate table's 240 deaths had a sticky action change the paddle's move in the last
two steps. The author's guesses lose a few flights outright and keep the rest safe; the
accurate ones turn many flights into close calls.

## Not only a hedge

If the guesses were only a hedge against sticky actions, the accurate table should match
the author's when sticky actions are switched off. It does not. Without sticky actions
nothing dies, yet in 10,000 steps the author's paddle returns the ball 834 times and every
return breaks a brick, while the accurate table's paddle returns it 734 times and only 569
returns break one. Every episode follows the same cycle, for both tables.

With sticky actions on, the accurate table dies about twice as often as the author's at
every low setting (0.02, 0.05 and 0.1), a steady factor rather than a widening gap. At the
published setting about 37% of the score gap, on a log scale, is slower scoring rather than
shorter lives. Perfect landing prediction tracks the author's table on both counts. So the
guesses do two jobs: they keep the paddle safe from the noise, and they send the ball back
into bricks.

## Accurate about the wrong column

The second job turned out not to need the bet at all. MinAtar's paddle saves a falling ball
in two places: under the column the ball is in, which bounces it back up still moving the
same way sideways, or under the column it is about to enter, a corner catch, which also
reverses its sideways direction. My accurate table
was fitted to whichever of the two was nearer, so it counted a corner catch as a hit.

That was the brick problem. Without sticky actions the two tables move the ball identically
until the accurate table's first corner catch. The author's table never makes one, and its
returns break a brick every time. Almost every corner catch (95% to 97%) comes from a paddle
standing still one column ahead of the ball.

Refitting the table to the under column alone gives 834 without sticky actions, the
author's score, and 357 ± 16 at the published setting, against 296 ± 15 for my old fit and
482 ± 17 for the author's (200 episodes each). So about a third of the gap I blamed on
accuracy was my yardstick. The rest is the bet: the refitted table still dies 5.1 times per
1,000 returns that turn under the wall's bottom row, against 2.4 for the author's. And the
author's guesses are still the least accurate of all by the new measure.

Measured on average error, the guesses look like the network's weakness. Measured on
what kills it, they are its best design decision, and the author said so in one line that
I first read as a bug report.

## Caveats

- All of this is one network and one environment, mostly at one sticky-action probability
  (0.1); the sticky sweep uses 200 episodes per setting. The row-swap and death-tracing
  experiments use 300 episodes, and effects under about 30 points there are within noise.
- Paddle motion is measured over the single step before the ball turns down, and the
  per-motion death counts outside rows 3 and 4 are small.
- The under-column table is one fit on 50 episodes, scored on 200; its 60-point gain is
  about 2.7 standard errors.
- The "bet on the worst case" reading is my interpretation of the measurements, not
  something the network or its author states in those words. What the author did write,
  "anticipate the earliest possible brick bounce", matches the weights exactly.

## Code and evidence

Every number here comes from scripts and evidence committed in
[weight-space-lab](https://github.com/ZayneMason/weight-space-lab), under
`experiments/breakout-anatomy/`: `README.md` (the anatomy), `TARGETS.md`, `HEDGE.md`,
`CLOSEDLOOP.md`, `ROWSWAP.md`, `DEATHS.md`, `MOTION.md`, `STICKY.md`, `RETURNS.md` and
`UNDERFIT.md`, each with its evidence file under `evidence/`. The network and its
description belong to Takehi et al.
