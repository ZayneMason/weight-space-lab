# Save #65's own repair and choose among the saved networks (roadmap #80)

**Date:** November 26, 2026 · **Status:** done on #79's 100 cases (rescored) and 50 fresh cases; a margin for the
second round's duel is #81 · **Cost:** 22.3 core-hours recorded for the fresh run (12.4 in #65's stage, 9.8 in
#71's), against the roadmap's estimate of about 20; the rescore of #79's cases took minutes.

#79 added #65's repair to #71's disagree chooser as a fifth candidate, but #65's stage saved only its winning triple
and its labels, so #79 rebuilt the repair by refitting that triple. The rebuild was within 0.001% of #65's recorded
disagreement in 83 of 100 cases and far off in two. This item makes `early_values` save the network edits each arm
ends with (`SAVE_EDITS`, off by default), and `saved_65.py` does two things:

1. `--rescore` scores a two-round chooser on #79's 100 cases from the networks #79 saved. Round one is #71's disagree
   chooser on its own candidates, as it is. Round two is a duel between that pick and #65's repair on up to 32 labels
   where the two disagree, from a fresh pool; #65's repair is taken only when it gets more of them right. The point
   is seed 319, where #79's one-round chooser spent all 32 labels ruling out #65's repair.
2. The whole cheap repair (columns and the batched committee fits, as #78) on 50 fresh seeds (500 to 549), with #65's
   network edits saved. #71's stage then scores, on the same pool, #71's chooser alone, #79's chooser with the saved
   repair (both tie rules), #79's chooser with the rebuild instead, and the two-round chooser.

```bash
python experiments/locate-repair/saved_65.py --rescore
python experiments/locate-repair/saved_65.py --output runs/s80 --part 0 --parts 4   # parts 0 to 3, one per core
python experiments/locate-repair/saved_65.py --score runs/s80
```

Evidence is in `evidence/saved-65/`: `rescore-79.json` and `rescore-79-check.json` (the two-round chooser on #79's
cases), `saved-65.json` (every fresh case's candidates with their network edits, every chooser's correct counts and
picks, and seconds) and `check.json` (the score).

## Results (measured)

| 50 fresh cases (500 to 549) | #71 alone | with saved repair, ties last | ties first | with the rebuild | two rounds | #65's repair alone | best on the probe |
|---|---|---|---|---|---|---|---|
| Chosen network under 0.01% / under 0.1% | 50 / 50 | 50 / 50 | 50 / 50 | 50 / 50 | 50 / 50 | 42 / 50 | 50 / 50 |
| Exact (all three broken weights edited) | 39 | 39 | 40 | 39 | 39 | 36 | 39 |
| Took #65's repair | | 2 | 9 | 0 | 4 | 50 | 0 |
| Closer to the original / further from it | | 0 / 0 | 0 / 0 | 0 / 1 (508) | 0 / 3 | 0 / 41 | 8 / 0 |
| Labels asked | 296 | 449 | 449 | | 704 | | |

| #79's 100 cases, rescored | #71 alone | #79 (ties last) | two rounds |
|---|---|---|---|
| Under 0.01% / exact | 97 / 78 | 98 / 79 | 98 / 79 |
| Closer to the original / further from it | | 4 / 1 (319) | 6 / 6 |
| Labels asked | 579 | 1,155 | 1,601 |

1. **The saved repair is #65's repair.** Its disagreement equals the one #65's stage recorded in all 50 fresh cases.
   The rebuild is within 0.001% of it in 44 and on the same side of the line in 49. The worst miss is 508: saved
   0.008%, rebuilt 0.27%.
2. **The rebuild can cost a good pick, the way 319 did.** On 508 the chooser with the rebuild spends all 32 labels
   on inputs where the rebuilt repair is wrong (it gets 0 right, the other three 32), never asks the labels that told
   boundary (0.0015%) from pick68 (0.008%), and returns pick68. With the saved repair it asks 19 labels and keeps
   boundary, as #71 alone does. That is the only case where the two differ.
3. **On these 50 seeds there was nothing left to fix.** #71 alone puts all 50 under 0.01%, and its pick is the probe's
   best in 42; in the other 8 the best is 0.0001% to 0.0014% closer. The saved repair, ties last, is taken in 2 cases
   (531 and 543), each at the same disagreement as #71's pick. Ties first is exact in one more case (532, where #65's
   repair is exact and pick68 finds two of the three at the same 0.0007%) and moves nothing on the probe.
4. **#71's stage still earns its place.** #65's repair alone is over the line in 8 cases (0.011% to 0.085%) and #71's
   stage brings all 8 under it. #65's repair alone is exact in 36, against 39.
5. **Two rounds fixes 319 and trades it for small losses.** On #79's cases it keeps 319's boundary pick and makes 6
   cases closer (311, 400, 402, 414, 430, 449), but 6 further (304, 321, 325, 339, 341, 437); on the fresh cases it
   makes 3 further (503, 513, 536) and none closer. Every loss is under 0.006% and none crosses the line. In all 9
   losses the duel had 1 to 7 split inputs and #65's repair won by 1 to 4 labels; the three gains that matter most had
   37 to 133 (402, 400, 414).
6. **Two rounds asks the most.** 1,601 labels on #79's cases against 1,155 for one round, and 704 against 449 on the
   fresh ones, because the duel uses a fresh pool and asks up to 32 more.
7. **Cost matches #78.** 22.3 core-hours on 50 cases (median 0.28 per case, max 1.64) against #78's 20.9: 12.4 in
   #65's stage (10.8 in #78) and 9.8 in #71's (10.1), with more cases needing #65's later stages (8 ran three, against
   6 in #78).

## Interpretation

- Saving #65's network is cheap and removes the rebuild's doubt; it should replace the rebuild. On these 50 cases the
  saved repair never changed the outcome, while the rebuild made one case worse. With #79's 100 cases, the default is
  #79's chooser with the saved repair and ties last.
- These 50 seeds were easier than #72's and #78's: no case was left over the line by #71's stage, so they say little
  about whether #65's repair rescues hard cases. 414 in #79 is still the only case where it did.
- The two-round chooser's losses come from duels decided by one or two labels on a handful of split inputs, where
  either network is about as close to the original as the other (inferred: from the losses' sizes, not tested). Taking
  #65's repair in round two only when the duel has enough split inputs, or a large enough margin, should keep 319,
  400, 402 and 414 and drop most of the losses, and it can be scored from the saved networks of these 150 cases
  without rerunning the repair. That is #81.
