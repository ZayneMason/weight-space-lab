# Half value labels from the behaviour line on, for three broken weights (roadmap #43)

**Date:** November 17, 2026 · **Status:** done (negative result) · **Cost:** CPU, about an hour and
three quarters (four runs in parallel, about six CPU-hours)

[#42](SWITCH_LINE.md) switched all 8 labels to value queries once fewer than 20 of the 20,000 pool inputs
split the committee, and lost ground at the strict 0.01% line. This experiment keeps the same trigger but,
past it, uses [#40](TRIPLE_VALUES.md)'s mix: 4 value queries and 4 of the committee's most even splits.

```bash
python experiments/locate-repair/half_line.py --output runs/half-line
```

## Method

- **Setup:** #37's three-fault cases (H = 16, random sizes in [1.0, 2.2] spreads, committee of 200
  triples, #29's margin refit, uniform pool of 20,000, 8 rounds of 8 labels, a million-input probe after
  every round).
- **Half-line rule:** while 20 or more pool inputs split the committee, all 8 labels go to its most even
  splits, as in #37 and #42. Once fewer do, 4 go to value queries (inputs where the winner and a copy with
  one value nudged by 0.01 spreads disagree) and the other 4 to the most even splits.
- **Told fit:** the true triple refitted on the same labels, fresh and chained, as in #41 and #42.

## Results (measured)

**After all 64 labels, 25 cases:**

| Arm | Value queries (of 1,600 labels) | Exact broken entries | Every visible fault | **Under 0.1%** | **Under 0.01%** | Median disagreement | Told fit (fresh): under 0.1% / 0.01% |
|---|---:|---:|---:|---:|---:|---:|---:|
| #37 targeted only | 0 | 16 | 19 | 22 | 16 | 0.0056% | 23 / 18 |
| #40 half value queries from the start | 797 | 15 | 17 | **24** | **19** | **0.0044%** | 24 / 20 |
| #42 all value queries past the line | 472 | 16 | 19 | 22 | 13 | 0.0095% | 23 / 13 |
| **Half value queries past the line** | **239** | 15 | 18 | 22 | 16 | 0.0074% | 23 / 18 |

**Round by round, half-line rule (#37 in brackets):**

| Labels | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Under 0.1% | 3 (3) | 10 (10) | 16 (16) | 19 (19) | 20 (20) | 21 (21) | 22 (22) | 22 (22) | 22 (22) |
| Under 0.01% | 0 (0) | 0 (0) | 0 (0) | 1 (1) | 3 (3) | 6 (7) | 11 (10) | 14 (15) | 16 (16) |
| Every visible fault | 3 (3) | 5 (5) | 9 (9) | 10 (10) | 12 (12) | 18 (18) | 20 (19) | 19 (19) | 18 (19) |
| Value queries asked that round | 0 | 0 | 4 | 8 | 20 | 47 | 68 | 92 | |

1. **The trigger fires exactly as in #42,** because the rule is the same until the line: 23 of 25 cases,
   first after 16 labels (seed 12), 24 (seed 8), 32 (three), 40 (seven), 48 (five) and 56 (six). Seeds 14
   and 20 never reach it. Once a case started mixing it kept mixing. 239 labels were value queries.
2. **It undoes #42's loss but adds nothing over #37.** Under 0.1% stays at 22 and under 0.01% is back at 16,
   the same cases as #37: no case crosses either line in either direction. Against #42, seeds 1, 8, 22 and
   24 cross back under 0.01% and seed 13 crosses back over it.
3. **Case by case against #37 it is a wash:** 10 closer, 8 further, 7 unchanged (median ratio 1.0). The
   median disagreement is 0.0074%, against 0.0056%.
4. **Weight-finding drops by one.** Seed 11 loses its true triple (as in #42), so 18 find every visible
   fault and 15 the exact entries. Its two-of-three repair still disagrees on only 0.003%.
5. **The labels pin the values as well as #37's do.** The true triple refitted on these labels has 23 under
   0.1% and 18 under 0.01%, the same as on #37's labels and better than #42's 13.
6. **The value-limited cases do not recover.** Seed 20 never mixes (116 splitting inputs in the last round)
   and stays at 0.37%. Seed 17 mixes only for the last round and ends at 0.116%, as in #37; #40 gets it to
   0.033%. Seed 14 still fails at 0.87%.
7. **Against #40:** 9 closer, 16 further. #40 is ahead at 0.1% on seeds 17 and 20, the two cases that reach
   the line late or never.
8. **Cost:** median 891 seconds per case.

## Interpretation

- *Inference:* half the labels past the line is enough to keep what the last splitting inputs say, so
  #42's loss came from dropping all of them. But value queries asked that late change nothing that matters.
- *Inference:* #40's behaviour gain is in the cases that never get near the line (seeds 17 and 20), so it
  must come from value labels asked early, while the committee still disagrees widely. A trigger that waits
  for the committee to settle cannot reach those cases.
- *Inference:* #40 is ahead on visible faults early (7 against 5 after 8 labels, 14 against 12 after 32)
  and falls behind only between 32 and 40 labels, where it stays at 14 while #37 jumps to 18. The next test is the reverse order: #40's mix early, then committee splits only, to see
  whether the early value labels keep the behaviour gain and the late splits win back the weights
  ([roadmap #44](../../docs/ROADMAP.md)).
- One run on 25 cases, so one or two cases either way is within noise.

## Evidence

- `evidence/half-line/half-line.json`: every case, every round, with the splitting count, the number of
  value queries asked and both told fits.
