# Where the committee screen loses the true triple (roadmap #47)

**Date:** November 18, 2026 · **Status:** done (diagnosis on 12 cases; wider screen on the 3 cases it
targets and 5 controls) · **Cost:** CPU, about three and a half hours

[#46](FRESH_SWITCH.md) found that the committee of 200 triples misses the true triple in 12 of 50
three-fault cases, and both clear failures left (seeds 14 and 49) are among them. The committee is built
in three steps: rank all pairs ([#26](JOINT_SCREEN.md)'s joint pair screen), extend the best 200 pairs by
every other entry to get about 10,000 triples, then fit each triple and keep the 200 that fit the practice
labels best. This experiment finds the step where each of the 12 true triples drops out, then widens the
step that loses the two failures.

```bash
python experiments/locate-repair/screen_ranks.py 49 runs/screen-misses
python experiments/locate-repair/early_values.py --early 2 --pairs 1000 --only 14 --output runs/screen-wide
```

## Method

- **Ranks:** for each of the 12 cases, `screen_ranks.py` rebuilds the committee and reports the best rank
  of any of the true triple's three pairs, whether the true triple is among the extended triples, and if so
  its rank in the committee's order.
- **Wider screen:** `early_values.py --pairs 1000` extends the best 1,000 pairs instead of 200, which gives
  about 4.5 times as many triples (38,000 to 48,000). The committee is still the best 200, and everything
  else is #45's 2-early-round rule.
- **Cases for the wider screen:** the 3 whose true triple is never extended (14, 47, 49), and 5 controls:
  seed 0, fresh seeds 25 to 27 (true triple already in the committee) and seed 28 (lost in the cut).

## Results (measured)

**Where the true triple drops out, 12 cases, before any targeted label:**

| Seed | Best rank of a true pair | Triples extended | True triple extended? | Its rank (committee keeps 200) | True triple's practice accuracy | Best member | 200th member |
|---|---:|---:|---|---:|---:|---:|---:|
| 3 | 7 | 9,265 | yes | 357 | 100% | 100% | 100% |
| 5 | 2 | 11,236 | yes | 759 | 100% | 100% | 100% |
| 17 | 1 | 10,199 | yes | 430 | 100% | 100% | 100% |
| 28 | 113 | 12,279 | yes | 874 | 100% | 100% | 100% |
| 30 | 6 | 8,701 | yes | 858 | 99.8% | 100% | 100% |
| 32 | 56 | 12,548 | yes | 1,175 | 100% | 100% | 100% |
| 37 | 54 | 11,490 | yes | 863 | 100% | 100% | 100% |
| 41 | 19 | 13,498 | yes | 206 | 100% | 100% | 100% |
| 44 | 2 | 13,014 | yes | 491 | 100% | 100% | 100% |
| **14** | **204** | 9,164 | **no** | — | **100%** | **99.6%** | 98.8% |
| 47 | 817 | 13,102 | no | — | 100% | 100% | 100% |
| **49** | **930** | 13,267 | **no** | — | **100%** | **99.9%** | 99.2% |

**Wider screen (1,000 pairs), after all 64 labels, against the 200-pair screen:**

| Seed | Kind | True triple in committee (200 → 1,000) | Exact broken entries | Every visible fault | Disagreement, 200 pairs → 1,000 pairs | Seconds, 200 → 1,000 |
|---|---|---|---:|---|---:|---:|
| **14** | never extended | no → **yes** | 2 → **3** | no → **yes** | **0.61% → 0.0092%** | 685 → 2,870 |
| **49** | never extended | no → **yes** | 1 → **3** | no → **yes** | **0.34% → 0.0069%** | 976 → 3,389 |
| 47 | never extended | no → yes | 1 → 2 | no → no | 0.041% → 0.0037% | 942 → 3,276 |
| 0 | control | yes → yes | 3 → 3 | yes → yes | 0.0061% → 0.0071% | 709 → 2,508 |
| 25 | control | yes → yes | 2 → 2 | yes → yes | 0.0022% → 0.0033% | 882 → 3,439 |
| 26 | control | yes → yes | 2 → 2 | no → no | 0.0016% → 0.0011% | 1,001 → 3,455 |
| 27 | control | yes → yes | 3 → 3 | yes → yes | 0.0040% → 0.0048% | 960 → 3,390 |
| 28 | lost in the cut | no → no | 2 → 0 | no → no | 0.073% → 0.052% | 863 → 2,592 |

1. **Two different ways to lose the true triple.** In 9 of the 12 cases it is extended and fits every
   practice label (seed 30: 99.8%), but so does the 200th member, and the tie-break puts it at rank 206 to
   1,175. In the other 3 (14, 47, 49) none of its three pairs is in the best 200 pairs, so it is never
   extended at all.
2. **The two clear failures are both never-extended cases, and in both the true triple would have won
   outright.** In seeds 14 and 49 it gets every practice label right, and no extended triple does: the best
   committee member gets 99.6% and 99.9%.
3. **Seed 14 was only just missed.** Its best true pair ranks 204th, against a cut at 200. Seed 49's ranks
   930th.
4. **Extending 1,000 pairs fixes both failures.** Seed 14 goes from 0.61% to 0.0092% and seed 49 from
   0.34% to 0.0069%, both on the exact broken weights. Seed 14 is under 0.1% before any targeted label
   (0.079%). Seed 14 had failed in every three-fault arm so far.
5. **Seed 47 also gets the true triple into the committee** but still ends on a lookalike, now 10 times
   closer (0.041% to 0.0037%).
6. **The controls do not change in kind.** Seeds 0, 25, 26 and 27 keep the same weight scores and move by
   at most a thousandth of a percent. Seed 28, lost in the cut rather than never extended, still lacks the
   true triple and ends on a different lookalike (0.073% to 0.052%).
7. **Cost:** the wider screen takes about 3.5 times as long per case (median about 3,300 against 910 seconds over
   these 8 cases), almost all of it fitting the extra triples once.

## Interpretation

- *Inference:* the joint pair score that orders pairs is the weak step for the clear failures. Seed 49's best
  true pair ranks 7th on the plain pair score and 930th on the joint one; the joint screen was built in #26
  for two-fault cases, where it helped.
- *Inference:* losing the true triple in a tie (9 of 12) mostly does no harm. Those committees still hold
  triples that fit every label, and the targeted labels then pick a lookalike that behaves like the
  original (all 9 end under 0.1% with 2 early rounds).
- *Inference:* a wider screen is worth its cost if it fixes failures without new ones. Eight cases are not
  enough to say it never hurts; roadmap #48 runs it on all 50.

## Evidence

- `evidence/screen-misses/rank-{seed}.json`: the ranks for the 12 cases.
- `evidence/screen-misses/wide/case-{seed}.json`: every round of the 8 wider-screen runs, with both told fits.
- The 200-pair runs are #45's `evidence/switch-point/early-2.json` and #46's `evidence/fresh-switch/fresh-early-2.json`.
