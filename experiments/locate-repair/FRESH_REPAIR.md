# Run the whole repair on 50 fresh cases (roadmap #72)

**Date:** November 24, 2026 · **Status:** done · **Cost:** 37.6 core-hours recorded (19.1 for #65's stage, 18.4 for
#71's), against the roadmap's estimate of about 20. Container restarts killed runs partway and those cases were
redone, so the machine spent more than the recorded figure.

#71's gate for the bigger committee and its disagree chooser were designed after seeing #67's 50 cases (seeds 250 to
299), so its 47 of 50 under 0.01% was not a held-out result. This experiment builds 50 fresh width-16 three-fault
cases (seeds 300 to 349) and runs the whole repair as it stood after #71, with nothing changed:

1. **#65's repair:** `early_values.py --width 16 --early 2 --two-step --middle --extra` (the two-step rule, #64's middle,
   one extra round of 8 region labels). It ran one stage on 38 cases, two on 9 and three on 3.
2. **#71 on #65's final labels:** #66's rebuilt committee, #67's restarts and #68's tied-label rounds give #68's pick;
   restarts, boundary labels and, where #68's pick misses one of its own labels, the bigger committee give the other
   candidates; the disagree chooser picks among them.

The result was fixed before the run: the disagree chooser's pick, scored on the 1M-point probe. The probe's own best
candidate is shown only as the ceiling.

```bash
python experiments/locate-repair/early_values.py --width 16 --early 2 --two-step --middle --extra \
    --only 300 304 ... --output runs/fresh-a/p0          # and p1 to p3 for the other seeds
python experiments/locate-repair/fresh_repair.py --collect runs/fresh-a
python experiments/locate-repair/fresh_repair.py --source runs/fresh-a --output runs/fresh-b --workers 4
python experiments/locate-repair/fresh_repair.py --source runs/fresh-a --score runs/fresh-b
```

Evidence is in `evidence/fresh-repair/`: `repair-65.json` (stage 1), `fresh-repair.json` (stage 2), `check.json` (the
score) and `labels/` (each case's final labels from stage 1, which stage 2 starts from).

## Results (measured)

| Repair | Under 0.01% | Under 0.1% | Exact triple |
|---|---|---|---|
| #65's repair alone | 41 | 50 | 37 |
| #68's pick | 46 | 50 | 39 |
| **Whole repair (disagree chooser), the pre-committed result** | **49** | **50** | **39** |
| Best by the probe (not available to a repair) | 49 | 50 | 39 |

On #71's tuning cases the same rows read 44, 47 and 47 under 0.01% for #68, the chooser and the probe.

1. **The held-out result matches the tuned one.** The whole repair puts 49 of 50 fresh cases under 0.01% and all 50
   under 0.1%. Against #65's repair alone it gains 8 cases (301, 308, 314, 319, 335, 338, 348, 349) and loses none;
   against #68's pick it gains 3 (308, 348, 349) and loses none.
2. **The chooser again reaches the ceiling.** It lands under the line in exactly the cases the probe's best candidate
   does, and in the one case over the line, 331, it picks the probe's best too (0.068%).
3. **It is cheap in labels.** It asks 244 labels across the 50 cases, a median of 3. 10 cases have no input in the
   200,000-point pool where the candidates differ, so there is nothing to ask.
4. **Boundary labels do the rescuing here.** In all three cases gained over #68 the boundary candidate wins the vote
   outright (308: 32 of 32 against 0; 348: 32 of 32 against 0 for each other candidate; 349: 18 of 18 against 0),
   moving 0.021% to 0.0007%, 0.024% to 0.0023% and 0.010% to 0.0018%. The bigger committee ran on 5 cases (309, 312,
   317, 328, 348) and rescued none.
5. **One case gets worse, by a hair.** 341 moves from 0.0013% to 0.0014%. Boundary labels applied everywhere would
   make 9 cases worse, restarts everywhere 3; the chooser avoids all but this one.
6. **The exact-weight count does not move.** 39 of 50 edit exactly the three broken weights, with or without the
   chooser. Two committees lack the true triple (303, 324), and both cases still end under 0.01%.

## Interpretation

- #71's design holds on cases it never saw: 49 of 50 under 0.01% held out, against 47 of 50 on the cases it was
  tuned on. The fresh cases look a little easier (#68 alone gets 46 here against 44), so the honest reading is that
  the gate and chooser did not overfit, not that the repair improved.
- The mix of fixes that matter changes from one set of cases to the next (#71: one rescue each from the bigger
  committee, restarts and boundary labels; here all three from boundary labels), which is why choosing among them
  from labels, rather than always applying one, is the part that carries over.
- The cost is real: 37.6 recorded core-hours for 50 cases, nearly twice the roadmap's estimate, most of it building
  candidates, not choosing between them.
