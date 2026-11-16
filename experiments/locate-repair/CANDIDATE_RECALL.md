# Getting the broken pair onto the suspect list (roadmap #25)

**Date:** November 16, 2026 · **Status:** done · **Cost:** CPU, about 1 hour 25 minutes (four runs in parallel)

[Roadmap #24](ACTIVE_SCREEN.md) found that targeted examples pick out the broken pair at H = 32
whenever it is among the candidates. It was there in only 13 of 25 cases. This experiment
compares candidate rules and runs the targeted examples again on the best one.

```bash
python experiments/locate-repair/candidate_recall.py --seeds 25 --output runs/candidate-recall
python experiments/locate-repair/candidate_recall.py --queries runs/candidate-recall/candidate-recall.json --output runs/candidate-queries
```

## Method

- #24's cases: H = 32, seeds 0–24, 1,024 practice examples.
- **Phase 1:** #23's coverage screen ranks all 13,041 pairs, and the top 3,000 get #21's coarse
  61 × 61 joint fit. Rules are then compared offline: fit the top 1,000 or 3,000; keep 40, 100
  or 200; order by coarse accuracy, then either smallest change (#24's rule) or screen rank;
  and optionally treat coarse accuracies within 0.2% or 0.5% of the best as tied.
- **Phase 2:** the best affordable rule (fit 3,000, keep 100, smallest change) gives the
  candidates for #22's eight rounds of 8 targeted queries from 20,000 fresh inputs per round.
  Random queries were not re-run; #24 already showed they barely help.
- **Silent faults:** a fault is silent when corrupting it alone changes no practice label.
  "Found the visible faults" means the winner touches every fault that is not silent.

## Results (measured)

| Candidate rule | broken pair kept (of 25) |
|---|---:|
| #24: fit 1,000, keep 40, smallest change | 13 |
| fit 1,000, keep 40, screen rank | 15 |
| fit 3,000, keep 40, screen rank | 16 |
| **fit 3,000, keep 100, smallest change** | **18** |
| fit 3,000, keep 200, smallest change | 18 |
| broken pair anywhere in the screen's top 3,000 | 21 |

| Targeted queries | #24 (40 candidates): found pair / pass | 100 candidates: found pair / visible faults / pass |
|---:|---|---|
| 0 | 5 / 19 | 6 / 8 / 20 |
| 16 | 8 / 22 | 11 / 12 / 23 |
| 32 | 9 / 23 | 13 / 16 / 24 |
| 48 | 12 / 22 | 17 / 20 / 24 |
| 64 | 12 / 22 | **18 / 21 / 24** |

1. **Keeping 100 candidates from the top 3,000 raises recall from 13 to 18 of 25.** Keeping 200
   adds nothing more. Treating near-perfect coarse fits as ties never helps: at keep 40 it cuts
   recall to between 3 and 13, because hundreds of other pairs then tie with the broken one.
2. **With 100 candidates, targeted queries find the broken pair in all 18 cases where it is a
   candidate.** Found-the-pair goes from 6 to 18 of 25 (#24: 12), and the benchmark passes 24
   of 25 (#24: 22). The last case flipped after 56 queries.
3. **Four cases have a fault that nothing can see.** In seeds 10, 14, 18 and 24 the second
   fault, corrupted alone, changes none of the 1,024 practice labels and at most 2 of 20,000
   uniform inputs. Hundreds of pairs then fit practice perfectly: 162 to 852 in the top
   1,000. The winner touches the visible fault in all four, so the repair is right
   wherever it matters.
4. **Every visible fault is found in 21 of 25 cases.** The 4 misses (seeds 4, 6, 8, 12) are
   all cases where the screen ranked the broken pair past 3,000 (3,500th to 10,175th of
   13,041). No candidate rule can reach them.
5. **What stays tied is the mirror symmetry again.** In the 17 found cases without a silent
   fault, the pairs tied at the best accuracy after 64 queries number exactly 2 to the power
   of the number of output-layer faults.

## Interpretation

- *Inference:* behavioural data plus a few dozen targeted labels locates two faults at H = 32
  whenever the faults are visible and the screen ranks them in its top 3,000. The remaining
  failures are a screen problem, not a choosing problem.
- A localization benchmark should score only the faults that change behaviour. A silent
  fault is a fault in the weights but not in the function, and no behavioural method,
  however many queries it makes, can be blamed for missing it.
- The smallest-change prior is wrong for choosing the winner (#21) but right for ordering
  coarse candidates: with 100 kept, it beats screen-rank ordering, 18 to 17.

## Limits

- 25 cases at one width; each count has a standard error of about 2.5 cases.
- Random queries were not re-run on the 100 candidates.
- "Silent" uses the 1,024 practice inputs; a fault silent there may matter elsewhere (here it
  changed at most 0.01% of uniform inputs).
- Phase 2 cost about 10 minutes per case for 100 candidates, since each round re-fits all of them.

Evidence: [`evidence/candidate-recall/`](../../evidence/candidate-recall/):
`candidate-recall.json` (each case's coarse accuracy and change for the screen's top 3,000
pairs, and the rule table) and `candidate-queries.json` (per-round histories with silent
faults). Code: `candidate_recall.py`; 5 tests in `tests/test_candidate_recall.py`.
