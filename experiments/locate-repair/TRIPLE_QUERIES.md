# Targeted examples for three broken weights (roadmap #28)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 45 minutes (four runs in parallel)

[Roadmap #27](THREE_FAULTS.md) broke three entries at H = 16. Its triple search picked all three
in only 5 of 25 cases, because several triples fit the practice set perfectly. For two faults,
[#22](ACTIVE_QUERIES.md) and [#26](JOINT_SCREEN.md) settled the same tie with examples chosen
where the candidate repairs disagree. This experiment runs those examples on triples.

```bash
python experiments/locate-repair/triple_queries.py --seeds 25 --output runs/triple-queries
```

## Method

- #27's cases: H = 16, seeds 0–24, three faults in three different hidden units, 1,024 practice
  examples, the same success rule.
- **Candidates:** #27's search up to its coarse fit (joint pair screen, the best 200 pairs
  extended by every other entry, a 7³ fit of each triple), keeping the best 40 triples.
- **Rounds:** #22's eight rounds of 8 targeted examples, each drawn from 20,000 fresh uniform
  inputs where the near-best candidates split their votes most evenly, labelled by the teacher.
  Every round re-fits all 40 triples with #27's fine fit (a 21³ grid, two finer local grids,
  then three coordinate passes). The winner has the best accuracy, with smallest change
  breaking ties.
- "Visible faults found" means the winner touches every fault that changes at least one
  practice label, mirrors included (4 cases have a silent fault).
- Seeds 11, 15 and 20 were rerun with the true triple's own fit recorded every round.

## Results (measured)

| Examples added | All three found | Visible faults found | Pass | Median damage undone |
|---:|---:|---:|---:|---:|
| 0 | 5 | 8 | 18 | 98.5% |
| 16 | 8 | 11 | 21 | 99.0% |
| 32 | 7 | 10 | 22 | 99.4% |
| 48 | 7 | 10 | 23 | 99.5% |
| 64 | **10** | 13 | 22 | 99.5% |

Counts are out of 25 cases.

1. **The targeted examples double how often all three are found, from 5 to 10 of 25.** That is
   still far below #26's two-fault result, where every visible fault was found in 25 of 25.
2. **The candidate list caps it.** The true triple is among the 40 candidates in 17 cases.
   Without it no example can help, and of the other 8 cases only one (seed 17, which has a
   silent fault) ends with every visible fault found.
3. **Inside the list, the fit fails, not the data.** Restoring the three teacher values scores
   100% on any set the teacher labels, and those values lie inside the fit's search range.
   Yet after 64 examples the winner fits every example in only 2 of 25 cases (19 of 25 before
   any example was added). In 15 of the 17 cases with the true triple among the candidates, no
   candidate, the true triple included, fits all the examples.
4. **So the right answer can lose to a wrong one.** In seeds 11 and 15 the winner held all three
   faults in some round and lost them later. In the three reruns the true triple's own fit ended
   between 97.2% and 99.4%. In seed 15 it fit every example through 16 added examples, then fell
   to 99.2% while a wrong triple reached 99.7%. In seed 11 it ended tied with a wrong triple and lost
   on smallest change. In seed 20 it never fit even the practice set perfectly (98.3%). The
   examples land where the candidates disagree, which is where a grid fit of three values is
   least precise.
5. **Passing still does not need the faults.** 22 of 25 repairs pass after 64 examples, against
   10 that hold all three faults.

## Interpretation

- *Inference:* targeted examples work only as well as the fit that scores the candidates. With
  two faults a 200 × 200 grid was fine enough; with three, a 21³ grid plus local refinement is
  not. The next step is a fitter that cannot fall short of the teacher, for example one that
  starts each round from the previous round's best values and refines locally, so that the
  true triple keeps its perfect fit as examples are added.
- The tie count after 64 examples is a median 2 (range 1–8). That is consistent with the mirror
  symmetry seen for pairs, but most winners here are not the true triple, so it is not a check
  of it.

## Limits

- H = 16 only, 25 cases, one corruption size.
- The committee is #27's coarse top 40; a 9-point coarse grid or a larger committee was not tried.
- The true triple's fit was recorded only in the three reruns; for the other cases, result 3
  follows from the winner's accuracy, since the winner's accuracy is at least the true triple's.

Evidence: [`evidence/triple-queries/triple-queries.json`](../../evidence/triple-queries/triple-queries.json)
(per-case histories) and [`true-triple-fits.json`](../../evidence/triple-queries/true-triple-fits.json)
(the three reruns). Code: `triple_queries.py`; 2 tests in `tests/test_triple_queries.py`.
