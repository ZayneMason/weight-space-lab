# Three broken weights (roadmap #27)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 15 minutes (four runs in parallel)

[Roadmap #26](JOINT_SCREEN.md) located two broken weights at H = 32 with a joint pair screen
and targeted queries. This experiment breaks three entries and asks two questions. Does the
pair screen still rank a pair of true faults near the top when a third fault is present? And
can extending the best pairs by a third entry find all three from practice data alone?

```bash
python experiments/locate-repair/three_faults.py --width 16 --seeds 25 --output runs/three-faults
```

## Method

- #20's two-fault cases at H = 16 (82 entries, 3,321 pairs), plus a third corrupted entry in
  yet another hidden unit (never a mirror or the same unit as an earlier fault), shifted by 1.6
  tensor spreads. Seeds 0–24, 1,024 practice examples, the same success rule.
- **Pair screen:** #26's 13 × 13 joint fit of every pair on the practice set, ordered by
  accuracy. A pair counts as true when it holds two of the three faults, mirrors included.
- **Triple search:** the best 200 pairs are each extended by every other entry (a median of
  11,902 distinct triples per case). Each triple gets a 7 × 7 × 7 joint fit. The best 20 are
  re-fitted on a 21 × 21 × 21 grid, then two finer local grids and a coordinate polish. The
  winner has the best practice accuracy, then the smallest total change.
- **Told:** the same fine fit on the three true entries.
- No targeted queries yet. Silent faults are as in [#25](CANDIDATE_RECALL.md).

## Results (measured)

| | 25 cases |
|---|---:|
| First pair holding two true faults: median position (of 3,321) | 21st |
| True triple among the candidate triples | 24 |
| True triple in the coarse top 20 (re-fitted finely) | 11 |
| Winner holds all three faults | 5 |
| Winner holds every visible fault | 7 |
| Benchmark pass: search / told | 17 / 20 |
| Median damage undone: search / told | 98.5% / 98.9% |

1. **The pair screen still works with a third fault present.** The first pair holding two true
   faults is a median 21st of 3,321, and in the top 200 in 24 of 25 cases. The true triple
   is among the extended triples in 24 cases.
2. **The coarse triple fit loses half of them.** The true triple is in the coarse top 20 in
   11 cases; its median position among about 12,000 triples is 23rd.
3. **Even in the top 20, practice data rarely picks the true triple.** Of those 11 cases the
   winner holds all three faults in 5. In 5 of the other 6, between 3 and 20 of the 20 re-fitted
   triples fit the practice set perfectly. Across all cases a median of 4 do.
4. **Passing and locating separate again.** The search repair passes 17 of 25 cases and
   undoes a median 98.5% of the damage, while finding all three faults in only 5.
5. **Fitting three values is itself harder.** Told exactly where the three faults are, the
   fine fit passes 20 of 25. In 4 of the 5 failures it could not fit practice perfectly; in the
   other one it fit practice perfectly with values that failed on held-out examples.

## Interpretation

- *Inference:* three faults bring back #21's underdetermination one level up. Several triples,
  and several values for the true triple, reproduce 1,024 uniform practice examples. #22 and
  #24 showed that targeted queries settle this for pairs; the same should apply to triples,
  with the committee drawn from the coarse top triples.
- The pair screen is the reusable part: in 24 of 25 cases it puts a two-fault pair in its top
  200, so a third fault can be searched for around a short list of pairs instead of
  among 88,560 triples.

## Limits

- 25 cases at one width (H = 16), no queries.
- The coarse 7-point grid steps 0.83 spreads, so the true shift sits between grid points;
  a finer coarse grid would rank the true triple higher but costs more per triple.
- Two of the 25 cases have a true triple the search could not reach (seed 14, where no
  two-fault pair made the top 200) or ranked only 864th.

Evidence: [`evidence/three-faults/three-faults.json`](../../evidence/three-faults/three-faults.json)
(per-case ranks, winners and both repairs' outcomes). Code: `three_faults.py`, which reuses
`joint_screen.py`; 3 tests in `tests/test_three_faults.py`.
