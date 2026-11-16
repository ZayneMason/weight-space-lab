# Targeted examples on a 200-triple committee (roadmap #34)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about two hours (four runs in parallel)

[Roadmap #33](CONTINUOUS_TRIPLES.md) fitted every extended triple continuously. The true triple fit
all practice labels in 23 of 24 cases, but so did a median 160 others, so the top 40 held it in only
13 of 25 random-size cases and the top 200 in 21. This experiment takes the top 200 as the
committee and asks the teacher for labels where they disagree, as [#22](ACTIVE_QUERIES.md) did for
pairs.

```bash
python experiments/locate-repair/triple_committee.py --seeds 25 --output runs/triple-committee
```

## Method

- [#31](RANDOM_SIZES.md)'s random-size cases: three faults at H = 16, each between 1.0 and 2.2
  spreads, 25 seeds.
- The committee is #33's best 200 triples, ranked by practice accuracy after the continuous fit,
  with the coarse-7 order breaking ties.
- Eight rounds of #22's targeted examples. Each round adds the 8 inputs, of 20,000 fresh uniform
  ones, on which the members within 0.5% of the best accuracy split their votes most evenly. The
  teacher labels them, up to 64 in all.
- After every round each member is re-fitted with #29's margin fit, started from its last values
  and from its current values, and the better fit is kept. The winner has the best accuracy, then
  the smallest change. "Visible" means every fault that changes a practice label on its own,
  mirrors included.

## Results (measured)

| Examples | All three found | Every visible fault found | Pass | Members tied at best (median) |
|---:|---:|---:|---:|---:|
| 0 | 3 | 3 | 15 | 159 |
| 8 | 5 | 5 | 19 | 48 |
| 16 | 8 | 9 | 21 | 17 |
| 32 | 11 | 12 | 24 | 4 |
| 48 | 16 | 19 | 24 | 4 |
| 64 | **16** | **19** | 24 | 3 |
| #29 at 64, size 1.6, 40 candidates (flattered) | 12 | 15 | 23 | |

Counts are out of 25 cases.

1. **Targeted examples settle most of the 160-way tie.** With practice data alone, the winner
   finds all three faults in 3 of 25 cases. After 64 targeted labels it finds them in 16, and every
   visible fault in 19. Before #31 the best result, on the easier fixed-size cases, was 12 and 15.
2. **Wherever the committee held the true triple, most cases ended with it.** The committee had
   it in 21 cases. Of those, 16 ended with all three found. In 3 more (seeds 4, 10 and 15), a
   lookalike still tied the true triple on every label after 64 examples; seed 15 had found all
   three at 48 and lost them on a tie. In seed 20 the winner fit one more label than the true
   triple's fit did.
3. **The remaining limit is the list.** In 4 cases the committee lacked the true triple. Two of
   them (seeds 5 and 17) still found every visible fault, because one fault is silent.
4. **The pass rate rises too, but it was never the hard part.** 24 of 25 pass after 32 examples.
   With practice data alone 15 pass.
5. **Cost:** a median of 917 seconds per case, 841 of them building the committee.

## Interpretation

- *Inference:* three faults behave like two once the fit works. The practice set leaves a few
  hundred perfect repairs. A few dozen labels where they disagree cut that to a handful, and in
  most cases to the true triple. The method holds on bug sizes the grids cannot match.
- Every visible fault in 19 of 25 three-fault cases, and 23 of 25 two-fault cases (#32), are the
  honest numbers for a post.
- What is left: the candidate list (4 misses) and lookalikes that a uniform query pool can't
  separate (3 cases). A pool drawn near the committee's decision boundaries might settle the
  second.

## Limits

- 25 cases at H = 16, one random draw of sizes per case.
- One committee size (200). A smaller committee was not tried at this fit quality.
- The query pool is uniform over the input square, as in #22.

Evidence: [`evidence/triple-committee/triple-committee.json`](../../evidence/triple-committee/triple-committee.json)
(per-case histories with the winner, ties and the told triple's fit each round). Code:
`triple_committee.py`; 2 tests in `tests/test_triple_committee.py`.
