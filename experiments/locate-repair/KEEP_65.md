# Keep #65's own repair in the running (roadmap #79)

**Date:** November 26, 2026 · **Status:** done on 100 fresh cases; saving #65's own networks so the added candidate
is #65's repair and not a rebuild of it is #80 · **Cost:** 20.3 core-hours recorded, against the roadmap's estimate
of about 20.

On seed 414 (#78) #65's stage ended with an exact repair at 0.0035% disagreement, and #71's stage replaced it with a
one-fault fit at 0.069%, because #71's disagree chooser picks only among its own candidates (pick68, restarts,
boundary and, where the gate opens, the bigger committee). `keep_65.py` reruns #71's stage with the cheap repair on
#72's 50 fresh seeds (300 to 349) and #78's (400 to 449) and adds a fifth candidate, `repair65`: the triple #65's
stage picked, fitted on #65's final labels with the committee's own fit. Each case runs the disagree chooser twice on
the same pool, without `repair65` (the baseline) and with it, and the score tries two tie rules for it: last (ties
keep #71's order) and first (ties go to `repair65`).

```bash
python experiments/locate-repair/keep_65.py --prepare runs/f78
python experiments/locate-repair/keep_65.py --output runs/k65 --workers 4
python experiments/locate-repair/keep_65.py --score runs/k65
```

Evidence is in `evidence/keep-65/`: `keep-65.json` (every case's candidates with their network edits, both choosers'
correct counts and picks, and seconds) and `check.json` (the score). `--prepare` copied #78's final #65 labels and
#65 rows into `evidence/fresh-50/` (`labels/` and `repair-65.json`), as #72's already were.

## Results (measured)

| 100 fresh cases | without (#71 as it is) | with, ties last | with, ties first | #65's repair alone | best on the probe |
|---|---|---|---|---|---|
| Chosen network under 0.01% / under 0.1% | 97 / 100 | **98 / 100** | 98 / 100 | 80 / 97 | 98 / 100 |
| Exact (all three broken weights edited) | 78 | **79** | 78 | 65 | 80 |
| Took #65's repair | | 5 | 24 | 100 | 9 |
| Brought under the line / took over it | | 414 / none | 414 / none | 414 / 18 cases | 414 / none |
| Closer to the original / further from it | | 4 / 1 (319) | 7 / 8 | 11 / 78 | 32 / none |
| Labels asked | 579 | 1,155 | 1,155 | | |

"Best on the probe" picks whichever candidate disagrees least with the original on the million probe inputs; no
chooser can see that, so it is a ceiling for these candidates.

1. **The baseline is #78's run.** On seeds 400 to 449 the chooser without `repair65` makes #78's pick with #78's
   disagreement in all 50 cases and asks the same 322 labels. On #72's seeds it reaches #72's rates (49 under the
   line, 39 exact) with the cheap repair.
2. **Ties last fixes seed 414 and loses nothing.** It takes #65's repair in 5 cases: 414 (0.069% to 0.0035%, now
   exact), 400 (0.041% to 0.013%), 426 (0.0063% to 0.0053%), 430 (0.0051% to 0.0014%) and 330 (no change). No case
   crosses the line the wrong way. It matches the probe ceiling on the line (98) and is one exact repair short of it.
3. **On 414 #65's repair fits every label and #71's candidates do not.** It answers 31 of the 32 asked labels right,
   against 25 for pick68, restarts and the bigger committee, and fits all 1,096 of #65's labels, against 99.5% for
   pick68 on its own set. #71's stage had replaced a repair that fit its labels with ones that do not.
4. **The one case that gets worse is 319, and the cause is the asking.** With `repair65` in, the chooser spends all
   32 labels where #65's two-fault repair differs from the rest (it gets 0 right, the other three 32), so the labels
   that told boundary (0.0016%) from pick68 (0.0035%) are never asked, and the tie goes to pick68. Both are exact
   and well under the line.
5. **Ties first is worse.** It takes #65's repair in 24 cases, 19 of them only by winning a tie, and moves 8 cases further
   from the original (310, 319, 321, 403, 413, 428, 437, 448) for 7 closer. It is exact in 78, one fewer than ties
   last.
6. **#65's repair alone is not enough.** It is under the line in 80 cases and exact in 65; the chooser with
   ties last adds 18 under the line and 14 exact over it. Keeping both is what pays.
7. **The rebuild is close, not exact.** #65's own values were not saved, so `repair65` refits #65's triple on #65's
   labels. It finds the same number of broken weights as #65 recorded in all 100 cases and lands within 0.001% of the
   recorded disagreement in 83, but on the same side of the line in only 94 (80 under against 82). The largest misses
   are 415 (0.0014% recorded, 1.07% rebuilt) and 418 (0.025% against 0.85%).
8. **Labels double.** The chooser asks 1,155 labels with `repair65` against 579 without, because #65's repair
   often disagrees with every other candidate, which opens more split inputs (only 5 cases have none with it, 17
   without). Cost is 0.20 core-hours per case on average (median 0.14, max 1.03).
9. **A bug that would have hidden the gains.** The first version raised StopIteration inside `pick_fixes.choose`
   when `repair65` alone had the most correct answers, since it looks names up in `pick_fixes.ORDER`. The worker
   pool treated that as the end of the job list and stopped without an error, so the run lost exactly the cases where
   #65's repair wins. The script now swaps in an order that includes it.

## Interpretation

- Keeping #65's repair as a fifth candidate, with ties going last, is a free gain on these 100 cases: one case
  brought under the line (414), one more exact repair, four closer to the original, none lost. It should be the
  default. The added labels are the only real cost, and they come from a candidate that is often plainly wrong.
- 319 shows the price of a bad candidate: the chooser spends its labels ruling it out. Dropping a candidate that
  disagrees with all the others on most of the pool before asking, or asking a few labels per pair of candidates,
  would probably keep 319's boundary pick (inferred: not run).
- The rebuild is the weak point. In 415 and 418 the rebuilt repair is far worse than #65's own, and neither was
  chosen, so they cost nothing here; but a better rebuild could matter where #65's repair is the best candidate.
  Saving #65's own network edits is cheap and removes the doubt. That is #80.
- The two cases left over the line are 331 and 400. In 331 every candidate edits exactly the three broken weights
  and lands at 0.062% to 0.077%. Like 400 in #78, it finds the right weights with the wrong
  values.
