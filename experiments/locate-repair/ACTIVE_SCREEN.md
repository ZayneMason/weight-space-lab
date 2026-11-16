# Targeted examples on a wider network (roadmap #24)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 45 minutes (four runs in parallel)

[Roadmap #22](ACTIVE_QUERIES.md) asked the teacher about inputs where candidate repairs
disagree, and at H = 8 the broken pair won in 20 of 25 cases instead of 14. That needed
every pair as a candidate, which does not scale. [Roadmap #23](PAIR_SCREEN.md) made
candidates cheap at H = 32 with a coverage screen, but fits on its top 1,000 pairs found
the broken pair in only 10% of cases. This experiment puts the two together at H = 32.

```bash
python experiments/locate-repair/active_screen.py --seeds 25 --output runs/active-screen
```

## Method

- #20's two-fault cases at H = 32 (162 entries, 13,041 pairs), seeds 0–24, 1,024 practice
  examples, same success rule.
- **Candidates:** #23's coverage screen ranks every pair; its top 1,000 get #21's coarse
  61 × 61 joint fit; the best 40 by practice accuracy, then smallest change, are the
  candidates (#22 used 20).
- **Queries:** #22's eight rounds of 8 targeted or 8 random teacher-labelled inputs, each
  round re-fitting all 40 candidates on a 200 × 200 grid. The targeted pool is 20,000 fresh
  uniform inputs per round (#22 used 5,000).
- Round 0 is the candidates' fine fits on practice alone.

## Results (measured)

| Queries | Targeted: found pair / pass | Random: found pair / pass |
|---:|---|---|
| 0 | 5 / 19 | 5 / 19 |
| 8 | 5 / 21 | 5 / 19 |
| 16 | 8 / 22 | 5 / 19 |
| 24 | 9 / 23 | 5 / 19 |
| 40 | 11 / 23 | 6 / 19 |
| 64 | **12** / 22 | 6 / 19 |

Counts are out of 25 cases. "Found pair" means both broken entries, counting exact mirrors.

1. **Targeted examples still work at H = 32.** 64 targeted queries raise found-the-pair from
   5 to 12 of 25 cases. 64 random queries raise it to 6. The median damage undone after
   64 queries is 99.2% targeted and 95.7% random.
2. **When the broken pair is among the candidates, queries almost always find it.** It was
   among the 40 candidates in 13 of 25 cases. In those 13, the broken pair won 5 times on
   practice alone and 12 times after 64 targeted queries.
3. **The limit is now the candidate list.** Of the 12 cases with the broken pair missing,
   6 lost it at the screen (rank past 1,000) and 6 at the coarse fit (in the top 1,000
   but not among the best 40 coarse fits). The screen ranks the broken pair a median 326th
   of 13,041 here.
4. **What remains tied is the mirror symmetry again.** In all 12 found cases, the number
   of pairs tied at the best accuracy after 64 queries is exactly 2 to the power of the
   number of output-layer faults (1, 2 or 4), as in #22.
5. **The disagreements shrink fast.** The targeted inputs split the committee a median
   45% to 55% in round 1, 31% in round 5, and 0% from round 7 on: by then the 20,000-input
   pool held no input that split the candidates near the best accuracy.
6. **More correct fitting can cost a pass.** Targeted passing peaked at 23 of 25 after 24
   queries and ended at 22. In seed 4 the queries moved the winner to a pair holding one
   broken entry, which then failed the held-out rule, undoing 85% of the damage.

## Interpretation

- *Inference:* #22's conclusion holds one width up. The repairs that fit 1,024 uniform
  examples differ on a few rare inputs, and a few dozen labels there settle the choice.
  The cost of locating two faults is now set by finding candidates, not by choosing
  among them.
- The next lever is candidate recall: half the misses are pairs the coarse grid misjudged,
  so re-fitting more of the screened pairs finely, or a finer coarse grid, should recover
  some of them. The other half need a better screen.
- Passing and locating keep separating. Targeted queries pass 22 of 25 and locate 12;
  random queries pass 19 and locate 6.

## Limits

- 25 cases at one width, so each count has a standard error of about 2.5 cases.
- The candidate list is capped at 40, and only the 1,000 best-screened pairs are fitted.
- The query pool is uniform. Late rounds found no disagreement in it; a search along the
  candidates' decision boundaries might.

Evidence: [`evidence/active-screen/active-screen.json`](../../evidence/active-screen/active-screen.json)
(per-case rank, candidate check and per-round histories for both query modes). Code:
`active_screen.py`, which reuses `pair_screen.py` and `active_queries.py`; 2 tests in
`tests/test_active_screen.py`.
