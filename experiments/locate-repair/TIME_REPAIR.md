# Find where the repair's time goes (roadmap #73)

**Date:** November 25, 2026 · **Status:** done (timing); cutting the cost is #74 · **Cost:** 21.9 core-hours recorded
for the 35-case sample (21.5 of repair, 0.4 of probe scoring), plus a one-case pilot of about 0.2, against the
roadmap's estimate of about 60 for all 100 cases.

#72 showed the whole repair holds on fresh cases (49 of 50 under 0.01%) but costs about 0.75 core-hours a case, and
nobody had measured which part costs what. `time_repair.py` reruns #71's stage (everything after #65's repair) from
the same saved labels, with each part and each step inside the parts timed. Steps are timed exclusively: a step's
time leaves out the steps it calls, so the numbers add up. Probe scoring is pulled out, since a real repair never
runs it. #65's own stage is taken from the seconds it recorded.

**The plan changed after a pilot.** The roadmap planned all 100 cases and a cheaper order (build restarts and the
bigger committee only where they might matter). The pilot on seed 264 spent 583.5 s in the committee, 0.5 s on
restarts, 0.3 s on boundary labels, 22.1 s on the final networks and 0.1 s choosing. The fixes the cheaper order would
skip cost under a second, so it could save almost nothing. The run was cut to a 35-case sample, enough to place the
time: 12 of the 15 slowest cases and 23 others, spread over seeds 250 to 349. The sample leans slow, so its
per-case cost is higher than the 100-case average.

```bash
python experiments/locate-repair/time_repair.py --output runs/time --workers 5 \
    --only 250 256 264 272 280 287 295 302 309 314 320 326 333 340 348 259 275 283 291 297 \
           307 330 344 346 251 258 263 265 270 279 281 296 315 328 338
python experiments/locate-repair/time_repair.py --score runs/time
```

Evidence is in `evidence/time-repair/`: `time-repair.json` (each case's candidates, picks, seconds by part and by
step) and `check.json` (the score).

## Results (measured)

| Where the time goes (35 cases) | Core-hours | Share |
|---|---|---|
| #65's stage (recorded by #71 and #72) | 30.0 | 58% |
| #71's stage | 21.5 | 42% |
| of which: **margin refine** | **18.6** | 87% of #71's stage |
| coarse grid fits | 2.2 | 10% |
| joint scores | 0.41 | 2% |
| middle walk (final networks) | 0.23 | 1% |
| continuous fits, ranking pairs, refits' own work, tied split | 0.05 | under 1% |

By part, #71's stage is the committee and #68's rounds (14.7), the bigger committee on the 7 cases that ran it (6.6),
the final networks (0.23), and restarts, boundary labels and choosing together (0.01).

1. **One step is the cost.** The margin refine, a Nelder-Mead search on the label margins run after every fit and
   twice in every refit, takes 86.6% of #71's stage. It is not a few slow cases: its share is 81% to 89% in every one
   of the 35, median 85%.
2. **The fixes and the chooser are nearly free.** Restarts, boundary labels and the disagree chooser cost 0.01
   core-hours together across 35 cases. The bigger committee is expensive only because it is another committee, so
   its time is mostly margin refine too.
3. **The rerun reproduces the result.** The disagree chooser makes the same pick as #71 and #72 in 35 of 35 cases: 34
   under 0.01% (296, #70's unreachable case, is over), 28 exact, 246 labels asked.
4. **The cheaper order saves nothing worth having.** Building restarts and the bigger committee only where #68's pick
   misses a label saves 0.06 of 51.5 core-hours. It reaches the same 34 under 0.01% and 28 exact; it picks a different
   candidate in 11 cases, none of which crosses the line.
5. **A case costs a median 977 s in #71's stage** (579 s to 8,314 s) and 1,958 s in #65's.

## Interpretation

- The repair's cost is mostly one optimiser's stopping rule. The margin refine runs Nelder-Mead to tight tolerances
  (`xatol` 1e-6, `fatol` 1e-9, up to 800 iterations) after every candidate fit, whatever that fit already scores.
- #65's stage was not timed by step here. It builds committees and refits with the same functions
  (`triple_committee.committee` and `refit`), so the margin refine is probably most of its 30 core-hours too
  (inferred from the code, not measured).
- Choosing among the fixes from labels, the part that carried over to fresh cases in #72, costs almost nothing. What
  to cut is the inner search, not the fixes. That is #74: loosen or skip the margin refine, and check on these 35
  cases that the picks and the 34 under 0.01% hold.
