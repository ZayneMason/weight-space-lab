# How many early rounds of value labels, for three broken weights (roadmap #45)

**Date:** November 17, 2026 · **Status:** done · **Cost:** CPU, about three and a half hours (four runs in
parallel, about twelve and a half CPU-hours)

[#44](EARLY_VALUES.md) gave 4 of every 8 labels to value queries in the first four rounds and all 8 to
committee splits after that. It got [#40](TRIPLE_VALUES.md)'s behaviour (24 of 25 under 0.1%) and
[#37](BEHAVIOUR_SCORES.md)'s weight-finding (19 every visible fault). Four rounds was a guess. This
experiment runs the same rule with 2 and with 6 early rounds.

```bash
python experiments/locate-repair/early_values.py --early 2 --output runs/switch-point-2
python experiments/locate-repair/early_values.py --early 6 --output runs/switch-point-6
```

## Method

- **Setup:** #44's, unchanged (#37's 25 three-fault cases, H = 16, random sizes in [1.0, 2.2] spreads,
  committee of 200 triples, 8 rounds of 8 labels, a million-input probe after every round).
- **Arms:** 2 early rounds (200 value queries in all) and 6 early rounds (600). #44 is the 4-round arm (400).
- The three arms ask the same labels in the first two rounds, so they differ only from 24 labels on.

## Results (measured)

**After all 64 labels, 25 cases:**

| Arm | Value queries (of 1,600 labels) | Exact broken entries | Every visible fault | **Under 0.1%** | **Under 0.01%** | Median disagreement | Told fit (fresh): under 0.1% / 0.01% |
|---|---:|---:|---:|---:|---:|---:|---:|
| #37 targeted only | 0 | 16 | 19 | 22 | 16 | 0.0056% | 23 / 18 |
| **2 early rounds** | **200** | **16** | **19** | **24** | **20** | **0.0041%** | **25 / 22** |
| 4 early rounds (#44) | 400 | 16 | 19 | 24 | 18 | 0.0061% | 25 / 19 |
| 6 early rounds | 600 | 15 | 18 | 23 | 16 | 0.0069% | 25 / 16 |
| #40 half value queries every round | 797 | 15 | 17 | 24 | 19 | 0.0044% | 24 / 20 |

**Round by round: under 0.1%, under 0.01%, every visible fault:**

| Labels | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 early rounds | 16, 4, 8 | 17, 4, 10 | 20, 7, **15** | 21, 11, **18** | 23, 12, **19** | 24, 18, 19 | 24, **20**, 19 |
| 4 early rounds | 16, 4, 8 | 17, 5, 9 | 21, 7, 12 | 21, 8, 16 | 22, 11, 16 | 23, 16, 18 | 24, 18, 19 |
| 6 early rounds | 16, 4, 8 | 17, 5, 9 | 21, 7, 12 | 22, 9, 14 | 22, 14, 16 | 24, 15, 18 | 23, 16, 18 |
| #37 | 16, 0, 9 | 19, 1, 10 | 20, 3, 12 | 21, 7, 18 | 22, 10, 19 | 22, 15, 19 | 22, 16, 19 |

1. **Two early rounds is the best arm so far.** It keeps 24 of 25 under 0.1% and has 20 under 0.01%, the
   most of any three-fault arm (#40: 19, #44: 18, #37: 16). Its median, 0.0041%, is also the lowest.
2. **It uses the fewest value queries.** 200 of 1,600 labels, half of #44's 400 and a quarter of #40's 797.
3. **The weights match #37's exactly.** 19 every visible fault and 16 exact, the same cases as #37 and
   #44. Every visible fault reaches 19 after 48 labels, as in #37; with 4 early rounds it takes all 64.
4. **Six early rounds is worse on every count.** It loses seed 17 at the 0.1% line (0.111%) and seed 11's
   weights (18 every visible fault, 15 exact). Under 0.01% it falls back to #37's 16.
5. **More early rounds, later weights.** Every visible fault after 40 labels is 18, 16 and 14 for 2, 4 and
   6 early rounds.
6. **Seeds 17 and 20 stay fixed with two rounds.** They end at 0.086% and 0.0090% (#37: 0.116% and 0.37%).
7. **Against 4 early rounds, case by case:** 13 closer, 10 further, 2 unchanged; median ratio 0.94. Under
   0.01% it gains seeds 0, 8 and 13 and loses seed 9 (0.0141%, against 0.0051%).
8. **Against #40, case by case:** 14 closer, 10 further; median ratio 0.91. It finds every visible fault in
   seeds 11 and 17, which #40 misses.
9. **The labels pin the values best too.** The true triple refitted on the 2-round labels has 25 under 0.1%
   and 22 under 0.01%, the best told fit so far (#44: 19, #40: 20).
10. **Seed 14 still fails in every arm** (0.61%, 0.56% and 0.63%). Its committee never contains the true
    triple. Seeds 3, 5 and 17 also lack it and still end under 0.1%.
11. **Cost:** median about 945 seconds per case in both arms.

## Interpretation

- *Inference:* a little value labelling goes a long way. Two rounds (16 value queries per case) give all of
  #40's behaviour gain; more rounds delay the committee splits that find the last broken weight, and by six
  rounds that costs both weights and behaviour.
- *Inference:* the differences between 2 and 4 rounds are two or three cases out of 25 at the 0.01% line
  and within the noise of one run. The direction agrees across three measures (0.01%, median, and how soon
  the weights arrive), which makes 2 the better default, not a proven optimum.
- Untested: whether one round is enough, and whether 2 rounds still beat 4 and #37 on fresh cases.
  Roadmap #46 tries both.

## Evidence

- `evidence/switch-point/early-2.json` and `early-6.json`: every case, every round, with the number of
  value queries asked and both told fits.
- The 4-round arm is #44's `evidence/early-values/early-values.json`.
