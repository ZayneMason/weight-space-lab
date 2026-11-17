# Are the last lookalikes wrong at all? (roadmap #36)

**Date:** November 17, 2026 · **Status:** done · **Cost:** CPU, about three hours (four runs in parallel, restarted twice)

[#34](TRIPLE_COMMITTEE.md) and [#35](BOUNDARY_QUERIES.md) scored a three-fault repair by its entries:
did the winner edit the three broken weights (mirrors included)? By that score, 9 of 25 cases ended
on the wrong entries after 64 targeted labels. This experiment asks a different question of the same
runs: how often does each repaired network give a different answer from the original, uncorrupted
network?

```bash
python experiments/locate-repair/lookalikes.py --seeds 25 --output runs/lookalikes
```

## Method

- #34's cases and committee, built once per case. The same 64 targeted labels, with three query
  pools: uniform (#34), boundary (#35), and mixed (half of each, new).
- After the last round, every member tied at the best practice accuracy (up to 20) is run on a
  million fresh uniform inputs over the input square. Its **disagreement** is the share of inputs it
  labels differently from the teacher.
- For comparison: the continuous fit on the true three entries (when the committee holds them), the
  fit on the told entries, and the corrupted network.

## Results (measured)

| Pool | All three entries found | Every visible fault | Winner disagrees on under 0.1% | Under 0.01% | Median winner disagreement |
|---|---:|---:|---:|---:|---:|
| uniform | 16 | 19 | **22** | 17 | 0.0055% |
| boundary | 17 | 19 | **22** | 16 | 0.0030% |
| mixed | 16 | 19 | **22** | 17 | 0.0056% |
| corrupted network | | | 0 | 0 | 12.6% |

Counts are out of 25 cases. The uniform and boundary rows reproduce #34 and #35 exactly.

1. **Most "wrong" repairs are not wrong.** In the uniform run, 9 winners edit the wrong entries. 6
   of them (seeds 3, 4, 5, 10, 15, 21) disagree with the teacher on 0.009% or less of the input
   square, about the same as the fit on the true or told entries in the same case. On a million
   inputs they are the same function, up to about 90 points.
2. **The real misses are three cases, and only one is the search's fault.** Seed 14 (true triple
   not in the committee) ends at 0.87%. In seeds 17 and 20 the winner disagrees on 0.12% and 0.37%,
   but the fit on the told entries disagrees on 0.11% and 0.37% too: there the fit, not the choice of
   entries, sets the limit.
3. **Sometimes the lookalike is the better repair.** In seed 2 with the boundary pool, the winner
   edits one wrong entry and disagrees on 0.013%. The fit on the told entries, started from the
   current values, disagrees on 2.0%.
4. **Lookalikes are often in the same hidden unit.** In seeds 4, 10 and 17, the lookalike swaps one
   true entry for another entry of the same hidden unit. In the other cases it moves a different
   unit.
5. **The mixed pool adds nothing.** 16, 19 and 22, the same as uniform.
6. **Cost:** a median of 954 seconds per case for all three pools, 682 of them building the committee.

## Interpretation

- *Inference:* the benchmark was asking the wrong question. Small ReLU networks have many settings
  that compute nearly the same function, so "which weights were broken" often has no unique answer,
  while "does the repaired network act like the original" does. Judged that way, three-fault repair
  from 1,024 practice examples and 64 targeted labels works in 22 of 25 random-size cases.
- This changes the post: the headline is behaviour, 22 of 25, with the entry count (19 of 25) as the
  stricter measure and seed 14's pair-stage miss as the clear failure.
- The 0.1% line is a choice. Seeds 17, 20 and 14 sit at 0.12%, 0.37% and 0.87%; the corrupted
  networks disagree on 2.3% to 69%.

## Limits

- 25 cases at H = 16, one draw of sizes per case, and disagreement measured only on the input square
  [-2, 2]², where the practice data lives.
- Only the final round was probed; the earlier rounds may already match behaviourally sooner.
- The two-fault results (#32, 23 of 25) have not been rescored this way.

Evidence: [`evidence/lookalikes/lookalikes.json`](../../evidence/lookalikes/lookalikes.json) (per case
and pool: histories, tied members with their entries and disagreement, true, told and corrupted
disagreement). Code: `lookalikes.py`, plus `triple_committee.py --pool mixed`; 2 tests in
`tests/test_lookalikes.py`.
