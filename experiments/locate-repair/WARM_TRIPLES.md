# A fitter that keeps the true triple perfect (roadmap #29)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 42 minutes (four runs in parallel)

[Roadmap #28](TRIPLE_QUERIES.md) added targeted examples for three broken weights at H = 16 and
found all three in 10 of 25 cases. It also found why it stopped there: after 64 examples, no
candidate fit every label in 15 of the 17 cases that had the true triple among the candidates,
even though the teacher's own values fit every label. The grid fit was too coarse where the
examples landed. This experiment keeps everything from #28 and changes only the fit.

```bash
python experiments/locate-repair/warm_triples.py --seeds 25 --output runs/warm-triples
```

## Method

- #28's cases, candidates (#27's coarse top 40 triples), query rule and rounds.
- Each triple, in each round, keeps the best of three fits, by practice accuracy and then
  smallest change:
  - #27's fine grid fit, as in #28;
  - a **warm start**: the triple's best values from the previous round, refined with #27's two
    local grids and three coordinate passes;
  - a **continuous refinement** of the better of those two: Nelder–Mead on the mean hinge loss
    max(0, 10⁻⁴ − signed output margin), in units of each tensor's spread.
- The same fit is applied to the true triple every round ("told"), whether or not it is among
  the candidates, to check the fitter itself.

## Results (measured)

| Examples added | All three found | Visible faults found | Pass | Winner fits every label | Told fit fits every label |
|---:|---:|---:|---:|---:|---:|
| 0 | 6 (#28: 5) | 9 (8) | 20 (18) | 22 (19) | 24 |
| 16 | 10 (8) | 13 (11) | 22 (21) | 22 | 25 |
| 32 | 10 (7) | 13 (10) | 23 (22) | 20 | 24 |
| 64 | **12** (10) | **15** (13) | 23 (22) | 18 (2) | 22 |

Counts are out of 25 cases; #28's counts are in brackets.

1. **The fix works on the fit.** After 64 examples the true triple fits every label in 16 of
   the 17 cases where it is a candidate. In #28 no candidate did in 15 of those 17. The told fit
   fits every label in 22 of 25 cases. The other three end at 99.7–99.9%, so the fitter is
   better but not exact.
2. **The true triple no longer loses on fit.** In #28, seeds 11 and 15 found all three and then
   lost them as the true triple's fit fell. Here the true triple fits every label in every round
   in 16 of the 17 cases. Seed 15 is found from 8 examples on. Only seed 1 loses it for two rounds,
   on a tie with smallest change breaking it, and finds it again.
3. **All three found rises from 10 to 12 of 25; visible faults from 13 to 15.** Among the 17
   cases with the true triple available, every visible fault is found in 14. Two of those 14
   have a silent fault, which no example can reveal.
4. **The three remaining misses are real ties.** In seeds 4, 11 and 20 both the true triple
   and the winner fit all 1,088 labels, and the winner needs a smaller change. In seeds 4 and 11
   the last round's examples still split the near-best candidates about 27% to 73% on average,
   so more rounds should help. In seed 20 the split had fallen to 4%, so the pool rarely contains
   an input that separates them.
5. **The candidate list is now the main limit.** 8 of 25 cases never have the true triple
   among the 40 candidates. Only one of them (seed 17) ends with every visible fault found.

## Interpretation

- *Inference:* with three faults, targeted examples need a fitter that tracks each candidate
  as the examples accumulate. A fresh grid fit each round loses precision exactly where the
  examples go, and a warm-started continuous fit does not.
- For two faults, #26 found every visible fault in 25 of 25 cases; for three it is 15 of 25.
  The gap is now mostly candidate recall (8 cases) plus a few unsettled ties (3 cases), the
  same two limits #24 and #25 hit for pairs before the joint screen.

## Limits

- H = 16 only, 25 cases, one corruption size, 64 examples per case.
- The hinge margin 10⁻⁴ and the Nelder–Mead settings were set once and not tuned.
- The committee is still #27's coarse top 40; a larger committee or a 9-point coarse grid was
  not tried.

Evidence: [`evidence/warm-triples/warm-triples.json`](../../evidence/warm-triples/warm-triples.json)
(per-case histories, including the true triple's and the told fit's accuracy each round).
Code: `warm_triples.py`; 3 tests in `tests/test_warm_triples.py`.
