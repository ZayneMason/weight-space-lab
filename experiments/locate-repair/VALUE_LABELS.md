# Ask more value labels where the values are not settled (roadmap #82)

**Date:** November 26, 2026 · **Status:** done on #79's 100 cases and #80's 50; running the whole repair with the new
step on fresh seeds is #83 · **Cost:** 1.5 core-hours recorded (0.09 for the line step on all 150 cases, the rest for
the region arm on 60), against the roadmap's estimate of a few.

After #81 the two cases over the 0.01% line are 331 (0.062%) and 400 (0.013%). In both, the chosen network edits the
three broken weights (400's third is the mirror of a broken one), so the weights are settled and the values are not.
`value_labels.py` keeps #81's pick's weights and tries two ways to move its values, on #65's final 1,096 labels plus
any new ones:

- **Region labels**, as the roadmap item proposed: walk the region of values that fit the labels as well as the pick
  (#63's hit-and-run walk), ask the 8 pool inputs where the walk's networks split most evenly, refit the values; four
  rounds, 32 labels. Run on the 60 cases where every candidate edits the same weights, with a control that refits on
  #65's labels alone.
- **Flat line**: find the direction in which the labels constrain the values least (the smallest singular vector of
  the margin gradients of the 24 labels nearest the boundary) and how far the fit reaches both ways along it, up to 4
  spreads. Where that direction is flat (singular value zero to 1e-6 of the largest), take the middle of the segment,
  then four rounds of 8 labels where 9 networks spaced along it split most evenly, each followed by a new segment and
  its middle. Run on all 150 cases.

```bash
python experiments/locate-repair/value_labels.py --prepare runs/s80
python experiments/locate-repair/value_labels.py --output runs/v82 --workers 4
python experiments/locate-repair/value_labels.py --score runs/v82
```

Evidence is in `evidence/value-labels/`: `value-labels.json` (every case's arms, segments, singular values and network
edits) and `check.json` (the score). `--prepare` copied #80's final #65 labels into `evidence/saved-65/labels/`.

## Results (measured)

| 150 cases (#79's and #80's), #81's pick | under 0.01% | closer / further than the pick | over the line |
|---|---|---|---|
| #81's pick | 148 | | 331 (0.062%), 400 (0.013%) |
| Flat line, middle only (no labels) | **150** | 5 / 8 | none |
| Flat line, after 8 labels | 149 | 5 / 8 | 349 (0.020%) |
| Flat line, after 32 labels | **150** | 7 / 3 | none |

| 60 cases where every candidate edits the same weights | under 0.01% | closer / further |
|---|---|---|
| #81's pick | 58 | |
| Refit on #65's labels (control) | 52 | 3 / 45 |
| Region labels, after 8 / 32 | 55 / 58 | 13 / 46, 32 / 27 |

1. **On 331 the fit is a line the walk cannot travel.** The true values fit all 1,096 labels, and every point on the
   straight line from the pick to them fits every label too, while the networks along it go from 0.062% to 0% of
   inputs wrong. The walk's mean chord is 0.0003 spreads and its points stay within 0.008 of the pick along that line.
   The least-constrained direction has singular value 0 and points exactly at the truth (cosine -1.0).
2. **The segment's middle fixes both cases over the line, with no new labels.** 331 goes from 0.062% to 0.0071% and
   400 from 0.013% to 0.0016%. All 150 cases are under the line for the first time. 27 of the 150 picks have a flat
   direction (16 of them edit all three broken weights).
3. **The middle alone also makes 8 cases worse, and 32 labels along the segment undo most of it.** After 32 labels the
   line arm is closer to the original than the pick in 7 cases (331 to 0.0067%, 400 to 0.0012%, 542 from 0.0096% to
   0.0017%, and 402, 443, 526, 547) and further in 3 (349 0.0018% to 0.0020%, 418 0.0018% to 0.0026%, 428 0.0015% to
   0.0034%), all under the line. After 8 labels 349 was briefly over it (0.020%).
4. **Most flat directions change nothing.** In 14 of the 27 the disagreement is the same to 0.0001% at every step, and
   in 6 of those the segment reaches the 4-spread cap at one end.
5. **Where the line matters, it moves two units that cancel.** In 331, 400 and 547 the line moves parameters of two
   hidden units, and the networks along it differ only on inputs where exactly one of the two units is active (201 of
   205 differing inputs out of 400,000 in 331, all 50 in 400, all 3 in 547). Where both are active the two changes
   cancel exactly, which is why the margins of nearby labels do not move.
6. **Region labels are the wrong tool here.** Refitting on #65's labels alone makes 45 of 60 picks worse and puts 6
   over the line, because #71's candidates were fitted on more labels. 32 region labels bring the count back to 58 but
   leave 331 (0.060%) and 400 (0.057%) where they were: the walk's networks split on 0 to 3 of 20,000 pool inputs per round in
   those two cases, since the walk stays near the pick (measured on 331, finding 1).
7. **The line step is cheap.** 0.09 core-hours for all 150 cases (median 1.3 seconds per case); the region arm took
   1.4 core-hours on 60.

## Interpretation

- #65's middle step and region labels both rest on the hit-and-run walk, and a walk with random directions cannot
  find a region that is a long, thin line: almost every random chord is cut short at once. Looking for the flat
  direction from the labels' gradients finds it in seconds. This is why #63 to #65 never moved 331 or 400.
- The new step should run after #81's chooser: take the flat segment's middle where there is one, then 32 labels along
  it. On these 150 cases that puts all 150 under 0.01% and makes 7 closer for 3 slightly further. The rule was shaped
  after looking at 331 and 400, so its rate needs fresh seeds; that is #83.
- The cancelling pair suggests a cheaper test than the gradients: the flat line exists when the pick edits
  parameters of two hidden units whose contributions are linear in the same input wherever both are active
  (inferred: checked on 3 cases, not on all 27).
