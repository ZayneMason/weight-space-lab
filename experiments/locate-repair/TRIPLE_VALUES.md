# Value queries for three broken weights (roadmap #40)

**Date:** November 17, 2026 · **Status:** done · **Cost:** CPU, about an hour and three quarters (four runs
in parallel)

[#38](VALUE_QUERIES.md) and [#39](SWITCH_QUERIES.md) closed most of the two-fault value gap by spending
some of each round's labels on inputs aimed at the winner's own values. [#37](BEHAVIOUR_SCORES.md)'s
three-fault repairs already refit each triple with #29's continuous margin fit, so this experiment adds
only the queries, and asks whether the same trade shows up: better behaviour, slightly worse weight-finding.

```bash
python experiments/locate-repair/triple_values.py --output runs/triple-values
```

## Method

- **Setup:** #37's three-fault cases (H = 16, random sizes in [1.0, 2.2] spreads, committee of 200
  triples, #29's margin refit, uniform pool, 8 rounds of 8 labels, a million-input probe after every
  round). Round 0 reproduces #37 exactly in every case.
- **Value queries:** each round, 4 of the 8 labels go to inputs where the winning triple and a copy of
  it with one value nudged by 0.01 spreads disagree (six nudged copies, taken in turn). The other 4 go
  to the committee's most even splits, as in #37.
- **Told fit:** the true triple refitted on the same labels, as a reference for how well the labels
  can pin the values when the weights are known.

## Results (measured)

**After all 64 labels, 25 cases:**

| Arm | Value queries (of 1,600 labels) | Exact broken entries | Every visible fault | **Under 0.1%** | **Under 0.01%** | Median disagreement | Told fit: under 0.1% / 0.01% |
|---|---:|---:|---:|---:|---:|---:|---:|
| #37 targeted only | 0 | 16 | **19** | 22 | 16 | 0.0056% | 23 / 18 |
| Half value queries | 797 | 15 | 17 | **24** | **19** | 0.0044% | 24 / 20 |

**Round by round:**

| Labels | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Value queries: under 0.1% | 3 | 10 | 16 | 16 | 21 | 23 | 23 | 24 | 24 |
| #37: under 0.1% | 3 | 10 | 16 | 19 | 20 | 21 | 22 | 22 | 22 |
| Value queries: under 0.01% | 0 | 0 | 3 | 6 | 9 | 10 | 14 | 18 | 19 |
| #37: under 0.01% | 0 | 0 | 0 | 1 | 3 | 7 | 10 | 15 | 16 |
| Value queries: every visible fault | 3 | 7 | 9 | 10 | 14 | 14 | 16 | 17 | 17 |
| #37: every visible fault | 3 | 5 | 9 | 10 | 12 | 18 | 19 | 19 | 19 |

1. **The two-fault trade repeats.** Value queries bring 24 of 25 three-fault repairs under 0.1% and 19
   under 0.01%, against 22 and 16, but every visible fault drops from 19 to 17.
2. **The strict line comes earlier.** Under 0.01%, the value-query arm leads from 16 labels on, and has
   14 cases after 48 labels against 10.
3. **Case by case it is closer to the original in 16 cases and further in 9.**
4. **The two cases that move under 0.1% were value-limited.** Seeds 17 and 20 go from 0.116% and 0.37%
   to 0.033% and 0.047%. The told fit moves with them, from 0.109% and 0.37% to 0.011% and 0.009%, so in
   #37 even the true weights could not be set right from those labels.
5. **The lost weight-finding cases still behave.** Seed 11 swaps one true entry for another weight
   and ends at 0.0064%. Seed 17 ends at 0.033% without editing every visible fault.
6. **Seed 14 still fails,** with one of three entries found and 0.87% disagreement in both arms.
7. **The told fit is not a true floor.** In seed 2 the refit of the true triple stays stuck at about
   2% in every round of this run (and until 56 labels in #37), while the winner, which is the true
   triple from 48 labels on, ends at 0.0015%. That one case is the told fit's only miss under 0.1%.
8. **Cost:** median 876 seconds per case, against 776 for #37.

## Interpretation

- *Inference:* for three faults as for two, labels chosen to split the committee find the weights,
  and labels aimed at the winner's values set them. Each job takes labels from the other, so with 64
  labels the split shows up as a one- or two-case swing between the entry and behaviour columns.
- *Inference:* the told fit is a local refit started from the previous round's values, so it can sit
  in a poor optimum when the winner, started elsewhere, does not. A told fit from several starts would
  make a firmer reference. Not tested.
- *Inference:* #39's switch rule recovered the visible-fault loss for two faults. Whether it does for
  three is roadmap #41.
- One run on 25 cases, so one or two cases either way is within noise.

## Evidence

- `evidence/triple-values/triple-values.json`: every case, every round, with the told fit.
