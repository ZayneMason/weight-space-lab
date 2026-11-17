# Value labels early, committee splits late, for three broken weights (roadmap #44)

**Date:** November 17, 2026 · **Status:** done · **Cost:** CPU, about an hour and three quarters (four
runs in parallel, about six CPU-hours)

[#40](TRIPLE_VALUES.md) gave half of every round's labels to value queries and got the best behaviour score
for three broken weights (24 of 25 under 0.1%, 19 under 0.01%), but found every visible fault in only 17
cases against [#37](BEHAVIOUR_SCORES.md)'s 19. [#43](HALF_LINE.md) showed that value queries asked late
change nothing. This experiment reverses the order: #40's mix for the first four rounds, then committee
splits only.

```bash
python experiments/locate-repair/early_values.py --output runs/early-values
```

## Method

- **Setup:** #37's three-fault cases (H = 16, random sizes in [1.0, 2.2] spreads, committee of 200
  triples, #29's margin refit, uniform pool of 20,000, 8 rounds of 8 labels, a million-input probe after
  every round).
- **Early-values rule:** in rounds 1 to 4, 4 labels go to value queries (inputs where the winner and a
  copy with one value nudged by 0.01 spreads disagree) and 4 to the committee's most even splits. In
  rounds 5 to 8, all 8 go to the most even splits, as in #37.
- **Told fit:** the true triple refitted on the same labels, fresh and chained, as in #41 to #43.

## Results (measured)

**After all 64 labels, 25 cases:**

| Arm | Value queries (of 1,600 labels) | Exact broken entries | Every visible fault | **Under 0.1%** | **Under 0.01%** | Median disagreement | Told fit (fresh): under 0.1% / 0.01% |
|---|---:|---:|---:|---:|---:|---:|---:|
| #37 targeted only | 0 | **16** | **19** | 22 | 16 | 0.0056% | 23 / 18 |
| #40 half value queries every round | 797 | 15 | 17 | **24** | **19** | **0.0044%** | 24 / 20 |
| #43 half value queries past the line | 239 | 15 | 18 | 22 | 16 | 0.0074% | 23 / 18 |
| **Half value queries in rounds 1 to 4** | **400** | **16** | **19** | **24** | 18 | 0.0061% | 25 / 19 |

**Round by round, early-values rule (#37, then #40, in brackets):**

| Labels | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Under 0.1% | 3 (3, 3) | 9 (10, 10) | 16 (16, 16) | 17 (19, 16) | 21 (20, 21) | 21 (21, 23) | 22 (22, 23) | 23 (22, 24) | 24 (22, 24) |
| Under 0.01% | 0 (0, 0) | 0 (0, 0) | 4 (0, 3) | 5 (1, 6) | 7 (3, 9) | 8 (7, 10) | 11 (10, 14) | 16 (15, 18) | 18 (16, 19) |
| Every visible fault | 3 (3, 3) | 6 (5, 7) | 8 (9, 9) | 9 (10, 10) | 12 (12, 14) | 16 (18, 14) | 16 (19, 16) | 18 (19, 17) | 19 (19, 17) |

1. **It keeps both gains.** Behaviour matches #40 at the 0.1% line (24 of 25) and weight-finding matches
   #37 (19 every visible fault, 16 exact). The cases that find every visible fault are #37's exactly. At
   the strict 0.01% line it has 18, between #37's 16 and #40's 19.
2. **Half as many value queries as #40.** Every case got 4 value queries in each of the first four rounds
   and none after: 400 of 1,600 labels, against #40's 797.
3. **The two cases #40 fixed are fixed again.** Seed 20 goes from 0.37% in #37 to 0.0078% (#40: 0.047%).
   Seed 17 goes from 0.116% to 0.098%, crossing the line only in the last round (#40: 0.033%).
4. **Against #37, case by case:** 15 closer, 8 further, 2 unchanged; median ratio 0.64. Seeds 6, 15, 19
   and 20 cross under 0.01%; seeds 8 and 10 cross back over it.
5. **Against #40, case by case:** 10 closer, 14 further (median ratio 1.09). At 0.01% it gains seeds 1, 6
   and 20 and loses seeds 0, 10, 12 and 13. It finds every visible fault in seeds 11 and 17, which #40 misses.
6. **The weights arrive later than in #37 and catch up.** Every visible fault is 16 after 40 and 48 labels,
   against #37's 18 and 19, and reaches 19 only in the last round.
7. **The labels pin the values well.** The true triple refitted on these labels has all 25 under 0.1% and
   19 under 0.01%, the best fresh told fit at 0.1% so far.
8. **One clear failure is left, and labels can't fix it.** Seed 14 ends at 0.56% (0.87% in every earlier
   arm). Its committee of 200 triples never contained the true triple; the true triple refitted on the same
   labels reaches 0.025%.
9. **Cost:** median 862 seconds per case.

## Interpretation

- *Inference:* value labels help most while the committee still disagrees widely, and committee splits are
  what find the last broken weight. Ordering them that way gets #40's behaviour and #37's weights.
- *Inference:* the 0.01% gap to #40 (18 against 19) is one case and within the noise of one 25-case run.
  The four cases #40 still wins at that line (0, 10, 12, 13) all end between 0.010% and 0.022% here, so it is
  about the last few hundredths of a spread, not about which weights.
- *Inference:* the remaining failure is a screening problem, not a labelling one: seed 14's true triple is
  never in the committee. The next useful change is to the committee, not to the labels.
- Untested: whether 4 early rounds is the right switch point. Roadmap #45 tries 2 and 6.

## Evidence

- `evidence/early-values/early-values.json`: every case, every round, with the number of value queries
  asked and both told fits.
