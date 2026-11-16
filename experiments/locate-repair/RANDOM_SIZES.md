# Do the screens only work because the grid matches the bug? (roadmap #31)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about three and a half hours (up to four runs in parallel)

Every fault since #7 shifts its entry by exactly 1.6 tensor spreads. [#30a](TRIPLE_RECALL.md)
found that #27's 7-point triple grid has a point at ±1.67 spreads, next to the corruption, and that
grids without such a point rank the true triple much worse. #26's 13-point pair screen has the same
point. This experiment checks the two- and three-fault results with fault sizes the grids cannot
match.

```bash
python experiments/locate-repair/random_sizes.py --mode pairs --seeds 25 --output runs/random-sizes
python experiments/locate-repair/random_sizes.py --mode triples --seeds 25 --output runs/random-sizes
python experiments/locate-repair/random_sizes.py --mode pairs --fixed 1.6 --points 14 --seeds 25 --output runs/random-sizes-control
```

## Method

- The same entries, signs and seeds as #26 (two faults, H = 32) and #27 (three faults, H = 16).
  Each fault's size is drawn uniformly between 1.0 and 2.2 spreads instead of 1.6. The practice
  and held-out sets are rebuilt from the new corrupted network exactly as #20 builds them.
- **Pairs:** #26's pipeline unchanged: coverage screen, the 13 × 13 joint screen, the best 40 pairs,
  then #22's targeted examples, 8 per round up to 64.
- **Triples:** #30a's rankings unchanged: coarse 7³ (#28's), coarse 9³, and the coarse-7 top 1,000
  refined with #29's margin fit.
- **Control for pairs:** fixed size 1.6, as in #26, but the joint screen uses 14 points per axis
  instead of 13. Its nearest points to 1.6 are 1.35 and 1.73. With `--fixed 1.6` the rebuilt cases
  match #26's to within 1e-6 (a test checks this), so any change comes from the grid alone.
- For triples, #30a already has a control in the other direction: the 9-point grid never had a
  point near 1.6 (its nearest are 1.25 and 1.875).

## Results (measured)

**Two faults, H = 32, 25 cases, after 64 targeted examples:**

| Run | True pair in top 40 | Found pair | Found every visible fault | Pass |
|---|---:|---:|---:|---:|
| #26: size 1.6, 13-point screen | 21 | 21 | **25** | 25 |
| Control: size 1.6, 14-point screen | 17 | 17 | 21 | 25 |
| Random sizes, 13-point screen | 17 | 16 | 20 | 23 |

**Three faults, H = 16, 25 cases, true triple in the top 40:**

| Ranking | #30a (size 1.6) | Random sizes |
|---|---:|---:|
| coarse 7 (#28 and #29 used this) | 17 | **5** |
| coarse 9 (no point near 1.6) | 8 | 7 |
| refined coarse-7 top 1,000 (coarse-7 ties) | 19 | 11 |

1. **#26's 25 of 25 depended on the grid.** Moving the screen's grid points away from 1.6, with
   nothing else changed, finds every visible fault in 21 cases instead of 25. All four losses are
   cases where the true pair fell out of the top 40 (ranks 9 to 34 under #26, 50 to 80 under the
   control). Random sizes give about the same, 20 of 25.
2. **The pass rate barely moves.** Every pair run passes 23–25 of 25. The screen's weakness shows
   up only in whether the broken weights are found, as in #20.
3. **For triples, the coarse-7 ranking loses about two-thirds of its recall.** The true triple
   reaches the top 40 in 5 of 25 cases instead of 17. The 9-point ranking, which never had a
   matching point, stays at 7 instead of 8. The bug sizes did not get harder in general; the
   7-point grid lost the advantage it had on 1.6-sized bugs.
4. **Refinement keeps some of it.** The refined ranking reaches 11, but it can only refine what
   the coarse-7 grid puts in its top 1,000. In 9 of 25 cases the true triple is not there: 8
   ranked below 1,000, and in 1 the pair stage never produced it. In #30a this happened once.
5. **Small faults are the hard pairs.** With random sizes, the true pair reached the top 40 in
   7 of the 13 cases whose smallest fault is at most 1.15 spreads, and in 10 of the other 12.

## Interpretation

- *Inference:* the headline results from #26 to #30a were measured on bugs whose size the coarse
  grids happened to contain. With sizes they cannot match, the pair pipeline finds every visible
  fault in about 20 of 25 cases, not 25, and the triple suspect list from #28 and #29 holds the
  true triple about a third as often. #29's 12 of 25 for triples was therefore also measured on a
  favourable list.
- What still holds: targeted examples still find the right repair among the candidates, and
  repairs still pass the benchmark without finding the bugs. The part that was flattered is the
  cheap screen that builds the candidate list.
- A fair screen should not depend on where its grid points land. Several grids at offset points,
  or a continuous fit of each candidate before ranking, are the obvious repairs (roadmap #32).

## Limits

- 25 cases per run, so each count has a standard error of about 2.
- One random draw of sizes per case, uniform between 1.0 and 2.2. The size range is a choice.
- The pair control moves the grid in one way (14 points); other offsets were not tried.
- The triple run was not taken through #29's targeted examples; only the suspect list was
  measured.

Evidence: [`evidence/random-sizes/`](../../evidence/random-sizes/) (`random-sizes-pairs.json`,
`random-sizes-triples.json`, `fixed-1.6-points-14-pairs.json`, with per-case rows and sizes). Code:
`random_sizes.py`; 3 tests in `tests/test_random_sizes.py`.
