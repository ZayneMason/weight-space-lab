# Switching at the behaviour line for three broken weights (roadmap #42)

**Date:** November 17, 2026 · **Status:** done (negative result) · **Cost:** CPU, about an hour and
three quarters (four runs in parallel, about six CPU-hours)

[#41](SWITCH_TRIPLES.md) found that #39's switch rule almost never fires for three faults, because some
pool input nearly always splits the committee. This experiment lowers the trigger: switch to value queries
once fewer than 0.1% of the pool (20 of 20,000 inputs) split it, the same line the repairs are scored on.

```bash
python experiments/locate-repair/switch_line.py --output runs/switch-line
```

## Method

- **Setup:** #37's three-fault cases (H = 16, random sizes in [1.0, 2.2] spreads, committee of 200
  triples, #29's margin refit, uniform pool of 20,000, 8 rounds of 8 labels, a million-input probe after
  every round).
- **Line rule:** while 20 or more pool inputs split the committee, all 8 labels go to its most even
  splits, as in #37. Once fewer do, all 8 go to #40's value queries (inputs where the winner and a copy
  with one value nudged by 0.01 spreads disagree).
- The number of splitting inputs is recorded every round, so the trigger can be re-read at other lines.
- **Told fit:** the true triple refitted on the same labels, fresh and chained, as in #41.

## Results (measured)

**After all 64 labels, 25 cases:**

| Arm | Value queries (of 1,600 labels) | Exact broken entries | Every visible fault | **Under 0.1%** | **Under 0.01%** | Median disagreement | Told fit (fresh): under 0.1% / 0.01% |
|---|---:|---:|---:|---:|---:|---:|---:|
| #37 targeted only | 0 | 16 | 19 | 22 | 16 | 0.0056% | 23 / 18 |
| #40 half value queries | 797 | 15 | 17 | **24** | **19** | **0.0044%** | 24 / 20 |
| #41 switch when none split | 101 | 16 | 19 | 22 | 16 | 0.0056% | 23 / 18 |
| **Line rule (under 20 split)** | **472** | 16 | 19 | 22 | **13** | 0.0095% | 23 / **13** |

**Round by round, line rule (#37 in brackets):**

| Labels | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Under 0.1% | 3 (3) | 10 (10) | 16 (16) | 19 (19) | 20 (20) | 21 (21) | 22 (22) | 22 (22) | 22 (22) |
| Under 0.01% | 0 (0) | 0 (0) | 0 (0) | 1 (1) | 3 (3) | 7 (7) | 10 (10) | 12 (15) | 13 (16) |
| Every visible fault | 3 (3) | 5 (5) | 9 (9) | 10 (10) | 12 (12) | 18 (18) | 20 (19) | 18 (19) | 19 (19) |
| Value queries asked that round | 0 | 0 | 8 | 16 | 40 | 91 | 134 | 183 | |

1. **The lower trigger fires.** 23 of 25 cases cross the line: one after 16 labels (seed 12), one after 24
   (seed 8), three after 32, seven after 40, five after 48 and six only for the last round. No case went back
   to committee splits once it had switched. 472 of the 1,600 labels were value queries.
2. **It does not help, and at the strict line it hurts.** Under 0.1% stays at 22. Under 0.01% falls from
   16 to 13: seeds 1, 8, 22 and 24 cross it the wrong way and only seed 13 crosses it the right way. The
   median disagreement rises from 0.0056% to 0.0095%.
3. **Case by case against #37 it is a wash:** 9 closer, 11 further, 5 unchanged (the median ratio is 1.0).
   Against #40 it is closer in 6 and further in 19.
4. **Weight-finding is unchanged in total.** 19 find every visible fault and 16 the exact entries, as in
   #37, though seed 11 loses its true triple and seed 4 gains one.
5. **The labels themselves pin the values worse.** Refitting the true triple on this arm's labels puts 13
   cases under 0.01%, against 18 on #37's labels and 20 on #40's. So the loss is in what the labels say,
   not only in which triple wins.
6. **The value-limited cases do not recover.** Seed 20 never drops below 20 splitting inputs (116 in the
   last round) and stays at 0.37%. Seed 17 switches only for the last round and ends at 0.142%, against
   0.116% in #37 and 0.033% in #40. Seed 14 still fails at 0.87%.
7. **Cost:** median 867 seconds per case.

## Interpretation

- *Inference:* "fewer than 20 inputs split the committee" is not the same as "the committee has settled".
  The remaining splitting inputs are where the lookalike triples still differ, and those inputs also say
  something about the values. Giving up all 8 committee labels at once drops that, and value queries
  aimed at one winner do not replace it.
- *Inference:* #40's fixed half-and-half mix from the first round is the best rule so far for behaviour.
  What it loses is two visible-fault cases. Whether those losses come from value queries asked early,
  before the committee has found the weights, is the question for roadmap #43.
- One run on 25 cases, so one or two cases either way is within noise. The 16 to 13 drop at 0.01% is
  four cases one way and one the other.

## Evidence

- `evidence/switch-line/switch-line.json`: every case, every round, with the splitting count, the number of
  value queries asked and both told fits.
