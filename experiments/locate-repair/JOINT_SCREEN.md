# A joint screen finds every visible fault (roadmap #26)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 35 minutes (four runs in parallel)

[Roadmap #25](CANDIDATE_RECALL.md) found every visible fault in 21 of 25 width-32 cases. The 4
misses were cases where #23's coverage screen ranked the broken pair about 3,500th to 10,000th of
13,041 pairs, so no candidate rule could reach it. The coverage screen scores a pair from its
two single edits. This experiment scores each pair with a small joint fit instead.

```bash
python experiments/locate-repair/joint_screen.py --seeds 25 --output runs/joint-screen
python experiments/locate-repair/joint_screen.py --seeds 25 --queries --output runs/joint-queries
```

## Method

- #24's and #25's cases: H = 32, seeds 0–24, 1,024 practice examples, the same success rule.
- **Joint screen:** every one of the 13,041 pairs gets a 13 × 13 joint grid (each value within
  2.5 tensor spreads, the grid includes no change) on the whole practice set. Pairs are ranked
  by that practice accuracy, and the coverage rank breaks ties. No smallest-change rule.
- **Queries:** the best 40 pairs are the candidates for #22's eight rounds of 8 targeted
  queries from 20,000 fresh inputs per round, as in #24.
- #25's silent faults (corrupting the entry alone changes no practice label) are excluded from
  "visible faults found".

## Results (measured)

| | Coverage screen (#23) | Joint screen |
|---|---|---|
| Broken pair's median position (of 13,041) | 327th | **5th** |
| In top 40 / 1,000 / 3,000 | 4 / 19 / 21 | **21** / 25 / 25 |
| Time per case, all pairs | about 6 s | about 54 s |

| Targeted queries (40 joint candidates) | found pair | visible faults found | pass |
|---:|---:|---:|---:|
| 0 | 8 | 10 | 24 |
| 16 | 16 | 20 | 25 |
| 32 | 17 | 21 | 25 |
| 48 | **21** | **25** | 25 |
| 64 | 21 | 25 | 25 |

1. **The joint screen ranks the broken pair a median 5th, against 327th.** It ranks it first in
   11 of 25 cases. In every case without a silent fault it is in the top 35.
2. **It recovers all four of #25's misses.** The broken pair ranked 8,332nd, 3,501st, 10,176th
   and 7,988th under coverage, and 1st, 29th, 1st and 1st under the joint screen.
3. **With 40 joint candidates and 48 targeted queries, every visible fault is found in all 25
   cases,** and the benchmark passes 25 of 25. The broken pair itself wins in all 21 cases
   without a silent fault. In the 4 silent cases the winner touches the visible fault, and the
   broken pair ranks 135th to 315th because hundreds of pairs fit practice perfectly.
4. **The ties left are exactly the mirror symmetry,** 2 to the power of the number of
   output-layer faults, in all 21 found cases.
5. **It costs less than #25's pipeline.** About 54 seconds per case screens all 13,041 pairs,
   against about 4 minutes for #25's coarse 61 × 61 fits on the top 3,000, and 40 candidates
   suffice instead of 100.

## Interpretation

- *Inference:* coverage fails where the two faults interact, so that neither single edit can
  correct its own errors while the other fault is still in place. In two of the four misses
  both faults are in the output layer, and in the other two both are hidden-layer weights.
  A coarse joint fit sees the pair's combined effect directly.
- #25's 61 × 61 coarse fit lost broken pairs because it ordered by smallest change. A 13 × 13
  grid ordered by accuracy alone, which is cruder, keeps them near the top. The smallest-change
  prior keeps doing harm at every stage where it is used to choose.
- The full pipeline at H = 32 is now: screen every pair with a 13 × 13 joint fit, keep 40,
  and ask the teacher about at most 48 inputs where they disagree. On these 25 cases it
  locates every fault that changes behaviour. Searching every pair scales with the square
  of the parameter count, so H = 128 (about 200,000 pairs) would take at least 15 minutes a case (an estimate from
  the per-pair time here, which will grow with width).

## Limits

- 25 cases at one width and one corruption size (1.6 spreads); the grid spans 2.5 spreads, so
  larger shifts would fall outside it.
- Two faults only. Three faults would mean about 700,000 triples at H = 32.
- The screen and the candidate fits use the same practice set, so practice accuracy is not a
  held-out measure; the benchmark rule and damage undone are.

Evidence: [`evidence/joint-screen/`](../../evidence/joint-screen/): `joint-screen.json` (per-case
ranks under both screens, the true pair's joint accuracy and timing) and `joint-queries.json`
(per-round histories). Code: `joint_screen.py`; 3 tests in `tests/test_joint_screen.py`.
