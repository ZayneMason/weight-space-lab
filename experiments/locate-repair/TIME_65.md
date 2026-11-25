# Time #65's stage by step and cut it with #74's refines (roadmap #75)

**Date:** November 25, 2026 · **Status:** done (timing and labels); #71's stage on the new labels and a batched
refine are #76 · **Cost:** 10.7 core-hours recorded for three 8-case runs (5.0, 3.0 and 2.5 of repair, 0.3 of probe
scoring), against the roadmap's estimate of 15 to 30.

#73 timed #71's stage by step but took #65's stage, 58% of the whole repair on its cases, from the seconds #65
recorded. #74 cut #71's stage with two cheaper margin refines: columns (same Nelder-Mead search, a loss that
recomputes only the hidden units an edit touches) and loose (columns with looser stopping). `time_65.py` reruns #65's
stage on 8 of #73's cases with each refine swapped in (`cut_refine.install`), times every step with #73's clock, and
compares each run with what #65 recorded:

- the labels the stage ends with (`x`, `y` and the 8 region-split inputs that #71's stage starts from), bit for bit;
- #65's repair itself: its disagreement with the original network on a million probe inputs, faults found, stages
  run and the switch arm's disagreement.

The sample is 3 cases #65 finished in one stage (250, 264, 320), 3 in two (291, 297, 307) and 2 in three (328, 338).

```bash
python experiments/locate-repair/time_65.py --kind original --output runs/t65-original --workers 4
python experiments/locate-repair/time_65.py --kind columns --output runs/t65-columns --workers 4
python experiments/locate-repair/time_65.py --kind loose --output runs/t65-loose --workers 4
python experiments/locate-repair/time_65.py --score runs/t65-original runs/t65-columns runs/t65-loose
```

Evidence is in `evidence/time-65/`: `original.json`, `columns.json` and `loose.json` (each case's seconds by step,
refine calls and evaluations, repair outcome next to #65's, and whether the labels match) and `check.json` (the
score).

## Results (measured)

| #65's stage, 8 cases | original | columns | loose |
|---|---|---|---|
| Stage core-hours (#65 recorded 5.22) | 4.98 | 3.01 | 2.45 |
| of which margin refine | 4.27 (86%) | 2.25 (75%) | 1.67 (68%) |
| Speedup on the stage (median per case) | 1 | 1.65x (1.70x) | 2.03x (2.04x) |
| Evaluations per refine call | not counted | 266 | 191 |
| Final labels identical to #65's | 8 of 8 | 4 of 8 | 0 of 8 |
| Repair identical to #65's | 8 of 8 | 7 of 8 | 1 of 8 |
| Under 0.01% (#65: 7 of 8) | 7 | 7 | 7 |

1. **The rerun is #65's stage.** With the original refine, all 8 cases end with #65's labels bit for bit and the same
   repair. It ran 4.5% faster in total than #65 recorded (inferred: machine load, since the work is identical).
2. **The refine is most of #65's stage too.** It takes 86% of the time (85% to 88% per case). Coarse grid fits are 9%;
   everything else is under 3%.
3. **The cheaper refines cut #65's stage more than #71's.** columns cuts it 1.65x (1.45x to 1.81x per case) and loose
   2.03x (1.75x to 2.21x), against 1.40x and 1.79x on #71's stage in #74.
4. **columns does not leave the labels alone here.** In #71's stage it changed nothing, but in #65's stage 4 of 8 cases
   end with different labels. The first 1,028 label rows match in every case and run; every difference is in the last
   60 rows, and in seed 250 also in the region splits. Those are the inputs the stage chooses to ask about as it goes
   (inferred from the code), so a slightly different refine path asks about different points. The repair still has the
   same disagreement in all 8; only seed 250's switch arm moves (0.0047% to 0.0029%).
5. **loose changes every case's labels but holds the line.** 7 of 8 stay under 0.01% and seed 338 stays over in all
   three runs (0.0138%). The repair's disagreement moves in 3 cases, all well under the line: 307 (0.0013% to 0.0016%),
   320 (0.0017% to 0.0012%) and 328 (0.0027% to 0.0061%). Faults found and stages run never change.

## Interpretation

- On these 8 cases the whole repair would go from 9.2 core-hours to 6.0 with columns (1.54x) and 4.9 with loose
  (1.90x), combining these runs with #74's runs of #71's stage on the same seeds. That is inferred, not measured end to
  end: #74 ran #71's stage from #65's original labels, and 4 of 8 columns cases and all loose cases now end with
  different ones.
- So #74's "same picks" for columns does not yet cover the whole repair. The check that does is #76: run #71's stage
  with columns and loose on the labels these runs saved, and compare the chosen networks with #73's.
- The refine is still 68% to 75% of the stage with the cheaper losses, so batching Nelder-Mead over many triples at
  once is still the next cut on cost (not tried; also #76).
