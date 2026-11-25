# Check the cheaper refines end to end, and batch the refine (roadmap #76)

**Date:** November 25, 2026 · **Status:** done on 8 cases; a fresh-case run of the whole cheap repair is #77 ·
**Cost:** 5.6 core-hours recorded (1.0 for #71's stage on the 4 cases whose columns labels changed, 2.2 on all 8
loose ones, 2.1 for the batched run and 0.2 of probe scoring), against the roadmap's estimate of 6 to 12.

#74 cut #71's stage with two cheaper margin refines and found the same picks, but it ran from #65's original labels.
#75 then showed that the same refines change the labels #65's stage ends with (columns in 4 of 8 cases, loose in all
8). So #74's check did not cover the whole repair. This item does two things on #75's 8 cases (250, 264, 291, 297,
307, 320, 328, 338):

1. **End to end.** `end_to_end.py` runs #71's stage with columns and loose on the labels #75's columns and loose runs
   saved, and scores the whole repair (#65's stage plus #71's) against #73's run of the original. Where #75's columns
   labels match #65's bit for bit (264, 307, 320, 328), #71's stage is #74's columns rerun, so its rows are reused
   rather than run again.
2. **Batching.** `batched_refine.py` runs the committee's Nelder-Mead fits for many triples at once, in arrays, in
   place of one SciPy call per triple, and reruns #71's stage with it (from #65's labels, as #74 did), scored against
   #73 and #74's columns run.

```bash
python experiments/locate-repair/end_to_end.py --kind columns --labels runs/t65-columns --output runs/e2e-columns \
    --workers 4 --only 250 291 297 338
python experiments/locate-repair/end_to_end.py --kind loose --labels runs/t65-loose --output runs/e2e-loose --workers 4
python experiments/locate-repair/end_to_end.py --score columns=runs/e2e-columns loose=runs/e2e-loose
python experiments/locate-repair/batched_refine.py --output runs/batched --workers 4 \
    --only 250 264 291 297 307 320 328 338
python experiments/locate-repair/batched_refine.py --score runs/batched
```

Evidence is in `evidence/end-to-end/` (`columns.json` and `loose.json` with each case's candidates, pick and seconds,
and `check.json`, the score) and `evidence/batched-refine/` (`batched.json` and `check.json`).

## Results (measured)

### The whole repair on the new labels

| Whole repair, 8 cases | original (#75 + #73) | columns | loose |
|---|---|---|---|
| Core-hours | 9.20 | 5.98 | 4.70 |
| Speedup | 1 | 1.54x | 1.96x |
| Under 0.01% / exact | 8 / 7 | 8 / 7 | 8 / 6 |
| Same pick as #73 | | 8 of 8 | 5 of 8 |
| Same chosen network's disagreement as #73 | | 7 of 8 | 3 of 8 |
| Crossed the 0.01% line | | none | none |

1. **columns holds end to end.** On the labels it changed, #71's stage still makes #73's pick in all 8 cases. The one
   chosen network that moves is seed 250's, from 0.0006% to 0.0003% disagreement with the original.
2. **loose holds the line but not the picks.** All 8 stay under 0.01%, and none crosses it, but 3 picks change and 5
   chosen networks move. The largest move is 328 (0.0010% to 0.0033%). Seed 250's new pick finds 2 of its 3 faults,
   against 3 before, at 0.0019%, so loose loses one exact repair.
3. **The estimates in #75 were close.** #75 inferred 6.0 core-hours (1.54x) for columns and 4.9 (1.90x) for loose by
   combining its runs with #74's; the measured runs are 5.98 and 4.70.

### Batching the refine

| #71's stage, 8 cases | #73 (original) | #74 columns | batched |
|---|---|---|---|
| Stage core-hours | 4.22 | 2.97 | 2.13 |
| of which margin refine | 3.64 | 2.25 | 1.48 (70%) |
| Speedup on the stage | 1 | 1.42x | 1.98x (1.39x over columns) |
| Same pick / same disagreement as #73 | | 8 / 8 | 8 / 8 |
| Under 0.01% / exact | 8 / 7 | 8 / 7 | 8 / 7 |

4. **The batched loop is SciPy's.** On 40 test problems it reaches the same point in the same number of evaluations as
   `scipy.optimize.minimize(method="Nelder-Mead")` (SciPy 1.18.1), and its loss matches the columns loss to about
   3e-17 (both checked in the tests). On a 300-problem sample, 282 fits are identical and all 300 reach the same
   accuracy on the labels.
5. **It is the same repair, faster.** The batched run makes #73's pick with the same disagreement in all 8 cases. It
   cuts #71's stage 1.39x over columns (1.18x to 1.54x per case) and 1.98x over the original. 93% of refine calls go
   through the batched path; the rest are the refits and the told triple, still one call at a time.
6. **A first version was no faster.** Recomputing all three touched units at every step ran at 0.88x to 0.99x of the
   columns loss. The version run here keeps each problem's fixed hidden columns, recomputes only the units whose own
   input weights or bias an edit changes, and works in place, which made it 1.70x per call on the sample. A timing of
   the columns loss on its own shows why batching pays: one evaluation takes 32.5 µs on 1,096 labels and 22.7 µs on
   16, so about 70% of it is Python overhead.

## Interpretation

- columns is now checked on the whole repair: same picks, same line, 1.54x cheaper on these 8 cases. It is the safe
  default. loose is cheaper again (1.96x) and kept every case under the line, but it changed 3 of 8 picks and lost
  one exact repair, so it stays an option, not the default.
- The batched refine gives the same answers as columns and a further 1.39x on #71's stage. Combining it with #75's
  columns run of #65's stage gives about 5.1 core-hours for the whole repair on these cases, 1.8x under the original
  (inferred: the batched refine has not been run in #65's stage, which also calls the committee).
- The refine is still 70% of the stage with batching. What is left is the per-step bookkeeping of the lockstep loop
  and the unbatched refits (inferred from the call counts, not profiled).
- Next is #77: install the batched refine in #65's stage too, and run the whole cheaper repair (columns plus batched)
  on fresh seeds, to check the picks and the line hold on cases nothing here was tuned on.
