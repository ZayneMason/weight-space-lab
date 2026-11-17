# Querying near the suspects' decision boundaries (roadmap #35)

**Date:** November 17, 2026 · **Status:** done · **Cost:** CPU, about two hours (four runs in parallel)

[Roadmap #34](TRIPLE_COMMITTEE.md) asked the teacher for 64 labels where a 200-triple committee
disagreed. It found all three faults in 16 of 25 random-size cases. In 3 more, a lookalike still tied
the true triple on every label, and by the last round the most even split that 20,000 uniform inputs
offered was a median 10% of the tied members against the rest. This experiment draws the query pool
near the tied members' decision boundaries instead, where they are most likely to disagree.

```bash
python experiments/locate-repair/triple_committee.py --pool boundary --seeds 25 --output runs/boundary-queries
```

## Method

- Everything as in #34: #31's random-size cases at H = 16, the same 200-triple committee, 8 rounds of
  8 targeted examples, the same re-fit every round.
- Only the pool changes. Each round draws 200,000 uniform inputs and keeps the 20,000 closest to a
  decision boundary of any of the (at most 20) members within 0.5% of the best accuracy. #22's rule
  then picks the 8 that split those members most evenly.

## Results (measured)

| Examples | All three found (uniform → boundary) | Every visible fault | Pass | Members tied at best (median) |
|---:|---:|---:|---:|---:|
| 0 | 3 → 3 | 3 → 3 | 15 → 15 | 159 → 159 |
| 16 | 8 → 6 | 9 → 6 | 21 → 19 | 17 → 16 |
| 32 | 11 → 8 | 12 → 10 | 24 → 23 | 4 → 8 |
| 48 | 16 → 14 | 19 → 16 | 24 → 24 | 4 → 4 |
| 64 | 16 → **17** | 19 → **19** | 24 → 24 | 3 → 3 |

Counts are out of 25 cases. Uniform is #34.

1. **The boundary pool keeps finding disagreements.** The median split of the chosen queries stays
   near even for longer: 0.31 in the last round against 0.10 with the uniform pool. Rounds whose best
   split was under 5% fell from 17 to 9.
2. **But the disagreements it finds teach less early on.** Up to 48 examples it trails the uniform
   pool on every count. By 64 it ends one case ahead on all three faults and level on the rest.
3. **It trades cases rather than closing them.** It fixes seeds 4 and 20 and settles seed 7's
   eight-way tie to one. It loses seeds 2 and 21. Seeds 10 and 15 end on the same two- and three-way
   ties as before.
4. **Labels near a boundary are harder to fit.** After 64 examples the true triple's fit gets every
   label right in 16 of the 21 cases whose committee holds it, against 18 with the uniform pool. Seed 21
   is lost this way: its true triple's fit misses labels that a lookalike gets right.
5. **Cost:** a median of 979 seconds per case, against 917.

## Interpretation

- *Inference:* where to ask is not what limits the last cases. A uniform pool learns faster at first,
  because spread-out labels pin down the broad shape. A boundary pool keeps the committee divided,
  but each label settles a thin sliver near an edge and is harder for the margin fit to honour. Net,
  the two are about equal.
- *Inference:* the two ties that survive both pools (seeds 10 and 15) may be lookalikes that agree
  with the teacher almost everywhere on the input square. If so they are not wrong repairs, and the
  fair count should judge a repair by how often it disagrees with the teacher, not by which entries
  it edits (roadmap #36).
- The post numbers stand: every visible fault in 19 of 25 three-fault cases, and 23 of 25 two-fault
  cases.

## Limits

- 25 cases, one draw of sizes per case, one pool size and one draw ratio (10 to 1).
- Closeness to a boundary is measured on the tied members' fitted values, which change every round.
- A mixed pool (part uniform, part boundary) was not tried.

Evidence: [`evidence/boundary-queries/boundary-queries.json`](../../evidence/boundary-queries/boundary-queries.json)
(same schema as #34, with `pool_mode`). Code: `triple_committee.py --pool boundary`; 3 tests in
`tests/test_triple_committee.py`.
