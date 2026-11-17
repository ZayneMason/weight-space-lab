# Switch rule for three broken weights (roadmap #41)

**Date:** November 17, 2026 · **Status:** done (negative result) · **Cost:** CPU, about an hour and
three quarters (four runs in parallel, about six CPU-hours)

[#40](TRIPLE_VALUES.md) found the two-fault trade again for three faults: value queries bring more repairs
close to the original (24 and 19 of 25 under 0.1% and 0.01%, against 22 and 16) but fewer find every
visible fault (17 against 19). For two faults, [#39](SWITCH_QUERIES.md)'s switch rule kept both gains by
spending labels on committee splits while any exist and on value queries after. This experiment runs that
rule on #40's three-fault cases.

```bash
python experiments/locate-repair/switch_triples.py --output runs/switch-triples
```

## Method

- **Setup:** #37's three-fault cases (H = 16, random sizes in [1.0, 2.2] spreads, committee of 200
  triples, #29's margin refit, uniform pool of 20,000, 8 rounds of 8 labels, a million-input probe after
  every round).
- **Switch rule (#39):** while any pool input splits the committee (the triples within 0.005 of the best
  practice accuracy), all 8 labels go to its most even splits, as in #37. Once none does, all 8 go to
  value queries (#40's inputs where the winner and a copy with one value nudged by 0.01 spreads disagree).
- **Told fit, two ways:** the true triple refitted each round from the previous round's values (#40's
  way, which stuck in seed 2), and fitted afresh each round the way committee members are.

## Results (measured)

**After all 64 labels, 25 cases:**

| Arm | Value queries (of 1,600 labels) | Exact broken entries | Every visible fault | **Under 0.1%** | **Under 0.01%** | Median disagreement |
|---|---:|---:|---:|---:|---:|---:|
| #37 targeted only | 0 | 16 | 19 | 22 | 16 | 0.0056% |
| #40 half value queries | 797 | 15 | 17 | 24 | 19 | 0.0044% |
| **Switch rule** | **101** | 16 | 19 | 22 | 16 | 0.0056% |

**Round by round, switch rule:**

| Labels | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Under 0.1% | 3 | 10 | 16 | 19 | 20 | 21 | 22 | 22 | 22 |
| Under 0.01% | 0 | 0 | 0 | 1 | 3 | 7 | 10 | 15 | 16 |
| Every visible fault | 3 | 5 | 9 | 10 | 12 | 18 | 19 | 19 | 19 |
| Value queries asked that round | 0 | 0 | 0 | 8 | 8 | 0 | 24 | 61 | |

1. **The committee almost never agrees, so the rule almost never switches.** In 16 of 25 cases some pool
   input still split the committee in every round, and not one value query was asked. Of the other 9,
   one first switched after 24 labels (seed 12, which went back to splits when new ones appeared), two after 48 (seeds 1 and 7) and six only for the last round.
2. **So the result is #37's, to the case.** Every round before a case's first switch matches #37 exactly,
   and the final counts are #37's: 22 and 16 under the two lines, 19 every visible fault, 16 exact.
3. **Where it did switch, it mostly helped a little.** Against #37, 4 cases end closer to the original
   (seed 7 from 0.0054% to 0.0019%, seed 12 from 0.029% to 0.017%), 1 ends further (seed 1, from 0.0013%
   to 0.0022%), and 20 are unchanged. None crosses either line.
4. **The two value-limited cases never switch.** Seeds 17 and 20, which #40's value queries brought from
   0.116% and 0.37% down to 0.033% and 0.047%, keep a splitting committee to the end and stay at #37's
   values. Seed 14 still fails at 0.87%.
5. **Fitting the told triple afresh does not change the reference.** Both told fits end with 23 of 25
   under 0.1% and 18 under 0.01%. In seed 2 neither stuck this time (0.0026% fresh, 0.0052% chained),
   so #40's stuck told fit depended on that run's labels.
6. **Cost:** median 879 seconds per case.

## Interpretation

- *Inference:* for two faults the committee runs out of splits, because only one or two pairs fit the
  labels. For three, the committee keeps triples that behave alike almost everywhere (#36's lookalikes),
  and some of the 20,000 pool inputs still separate them. "No split left" is too strict a trigger there.
- *Inference:* a trigger tied to the behaviour line, such as switching once fewer than 0.1% of pool inputs
  split the committee, would let the rule switch when the remaining disagreement is already below what
  the repairs are scored on. That is roadmap #42.
- One run on 25 cases.

## Evidence

- `evidence/switch-triples/switch-triples.json`: every case, every round, with both told fits and the
  number of value queries per round.
