# Run the whole cheaper repair on fresh seeds (roadmap #77)

**Date:** November 25, 2026 · **Status:** done on 12 fresh cases · **Cost:** 20.4 core-hours recorded (13.7 for the
original repair, 6.2 for the cheap one, 0.5 of probe scoring), against the roadmap's estimate of 20 to 35. Seeds 360
and 361 of the original run were also started in a second process near the end, so some of seed 360's work ran
twice and is not counted.

#76 checked columns on the whole repair and the batched refine on #71's stage, but only on 8 cases those refines had
already been tried on. `fresh_cheap.py` runs the whole repair, #65's stage and then #71's, on 12 seeds nothing has
been run on (350 to 361), two ways:

- **original:** the repair as #72 ran it, with `warm_triples.refine` as it is;
- **cheap:** columns for every refine (`cut_refine.install("columns")`) and the committee's continuous fits batched
  (`batched_refine.install()`), in #65's stage as well as #71's, since both call `triple_committee.committee`.

Each case runs #65's stage (`early_values.two_step` with the middle step and the extra round, as #72), saves its
labels, then runs #71's stage from them with #73's timed runner. Both runs are timed by #73's clock.

```bash
python experiments/locate-repair/fresh_cheap.py --kind original --output runs/fc-original --workers 2
python experiments/locate-repair/fresh_cheap.py --kind cheap --output runs/fc-cheap --workers 2
python experiments/locate-repair/fresh_cheap.py --score runs/fc-original runs/fc-cheap
```

Evidence is in `evidence/fresh-cheap/`: `original.json` and `cheap.json` (each case's two stages, with seconds by
step, refine calls, #65's repair outcome and #71's candidates and pick) and `check.json` (the score).

## Results (measured)

| Whole repair, 12 fresh cases | original | cheap |
|---|---|---|
| Core-hours | 13.67 | 6.24 |
| of which #65's stage / #71's stage | 7.02 / 6.65 | 3.57 / 2.67 |
| Margin refine share | 86% | 69% |
| Speedup (per case: min, median, max) | 1 | 2.19x (1.71x, 2.27x, 3.95x) |
| Chosen network under 0.01% / under 0.1% / exact | 12 / 12 / 9 | 12 / 12 / 8 |
| #65's repair under 0.01% | 9 | 9 |
| Same pick as the original | | 11 of 12 |
| Same chosen network's disagreement | | 9 of 12 |
| Crossed the 0.01% line | | none |

1. **The cheap repair is 2.2x cheaper on cases it was never tried on.** It cuts #65's stage 1.97x and #71's stage
   2.49x. Excluding seeds 360 and 361, which partly ran next to a second process in the original run, the speedup is
   2.34x. 84% of the cheap run's refine calls went through the batched path.
2. **It holds the line.** Every chosen network is under 0.01% disagreement with the original on a million probe
   inputs in both runs, and #65's repair alone is under the line in the same 9 cases.
3. **Four cases change somewhere, all well under the line.**
   - **355:** #65's stage ends with different labels (its repair goes from 0.0043% to 0.0026%), and the same pick
     moves from 0.0016% to 0.0024%, still finding all 3 faults.
   - **360:** the same kind of change; #65's repair goes from 0.0011% to 0.0024%, and the pick from 0.0011% to
     0.0008%.
   - **356:** the one changed pick. In the original run the coarse pick's fit missed some labels, so #71's stage
     also built the bigger committee, and three of its four fixes (pick68, restarts, bigger) were at 0.0005% with all
     3 faults; the chooser saw no disagreement and took pick68. In the cheap run the coarse pick differs and fits the
     labels, so no bigger committee is built; its pick68 and restarts fixes are at 0.0019%, the chooser asks for one
     label where they disagree with the boundary fix, and takes the boundary fix (2 of 3 faults, 0.0006%), the best
     of its three. That is the exact repair the cheap run loses.
   - **358:** #65's repair moves slightly (0.0085% to 0.0080%); #71's pick and chosen network are unchanged.
4. **The largest speedup is 356's, and it is not the refine's doing alone.** Its #71's stage took 3,131 s in the
   original run, 2,707 s of it in the margin refine, mostly for the bigger committee the cheap run never built,
   against 534 s cheap (inferred from which candidates each run built; the steps were not timed per candidate). The
   smallest is 360's #65's stage, 1.31x (3,679 s against 2,818 s), not profiled further.

## Interpretation

- On 12 fresh cases the cheap repair (columns plus batching, in both stages) does the original's job for under half
  the compute: same line, same pick in 11 of 12, one exact repair lost. This is in line with #76's 8 tuned cases,
  where columns alone kept every pick and the batched refine added 1.39x on #71's stage. The extra batching in #65's
  stage is what moves #65's labels in 355 and 360 (inferred: columns alone already changed labels in 4 of 8 cases in
  #75, through the points the stage asks about as it goes).
- So the cheap repair is a reasonable new default. Being cheap, it makes a larger held-out test affordable: at about
  0.5 core-hours a case, 50 fresh seeds cost about 26 core-hours against about 57 for the original. That is #78.
- The refine is still 69% of the cheap repair. What is left in it is the refits and the told triple, still one call
  at a time (16% of calls), and the lockstep loop's bookkeeping (inferred from the call counts, not profiled).
