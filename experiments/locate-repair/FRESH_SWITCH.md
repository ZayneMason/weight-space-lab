# Two early rounds on fresh cases, and one early round (roadmap #46)

**Date:** November 18, 2026 · **Status:** done · **Cost:** CPU, about six hours (four runs in parallel,
about 24 CPU-hours)

[#45](SWITCH_POINT.md) found that 2 early rounds of value labels beat 4 and 6 for three broken weights, by two
or three cases out of 25 at the 0.01% line. Those 25 cases are the ones every three-fault rule so far was
tuned on. This experiment runs 0, 2 and 4 early rounds on 25 fresh cases (seeds 25 to 49), where 0 is
[#37](BEHAVIOUR_SCORES.md)'s rule, and 1 early round on the original 25.

```bash
python experiments/locate-repair/early_values.py --early 1 --output runs/fresh-switch-1
python experiments/locate-repair/early_values.py --early 0 --seed-start 25 --output runs/fresh-switch-0
python experiments/locate-repair/early_values.py --early 2 --seed-start 25 --output runs/fresh-switch-2
python experiments/locate-repair/early_values.py --early 4 --seed-start 25 --output runs/fresh-switch-4
```

## Method

- **Setup:** #44's, unchanged (H = 16, random sizes in [1.0, 2.2] spreads, committee of 200 triples, 8 rounds
  of 8 labels, a million-input probe after every round). Only the seeds change.
- **Fresh cases:** seeds 25 to 49, never run before by any three-fault arm.
- **One early round:** 4 value queries in round 1, then committee splits only (100 value queries in all).

## Results (measured)

**Fresh cases (seeds 25 to 49), after all 64 labels:**

| Arm | Value queries (of 1,600 labels) | Exact broken entries | Every visible fault | **Under 0.1%** | **Under 0.01%** | Median disagreement | Told fit (fresh): under 0.1% / 0.01% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 early rounds (#37's rule) | 0 | 13 | 15 | 23 | 15 | 0.0065% | 25 / 19 |
| 2 early rounds | 200 | 13 | 15 | **24** | 15 | 0.0060% | 25 / 19 |
| 4 early rounds | 400 | 13 | **16** | **24** | **16** | **0.0059%** | 25 / **22** |

**Original cases (seeds 0 to 24), after all 64 labels:**

| Arm | Value queries | Exact | Every visible fault | **Under 0.1%** | **Under 0.01%** | Median | Told fit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 early rounds (#37) | 0 | 16 | 19 | 22 | 16 | 0.0056% | 23 / 18 |
| **1 early round** | **100** | **16** | **19** | **23** | **18** | **0.0066%** | **25 / 18** |
| 2 early rounds (#45) | 200 | 16 | 19 | 24 | 20 | 0.0041% | 25 / 22 |
| 4 early rounds (#44) | 400 | 16 | 19 | 24 | 18 | 0.0061% | 25 / 19 |

**Both sets together, 50 cases:**

| Arm | Every visible fault | Under 0.1% | Under 0.01% |
|---|---:|---:|---:|
| 0 early rounds | 34 | 45 | 31 |
| 2 early rounds | 34 | **48** | **35** |
| 4 early rounds | 35 | **48** | 34 |

**Fresh cases, round by round: under 0.1%, under 0.01%, every visible fault:**

| Labels | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 early rounds | 12, 1, 8 | 18, 1, 10 | 20, 6, 13 | 21, 10, 14 | 22, 12, 15 | 23, 13, 15 | 23, 15, 15 |
| 2 early rounds | 12, 3, 9 | 16, 4, 11 | 22, 6, 13 | 23, 11, 14 | 22, 13, 15 | 24, 13, 15 | 24, 15, 15 |
| 4 early rounds | 12, 3, 9 | 15, 4, 10 | 23, 5, 10 | 23, 9, 12 | 22, 11, 13 | 24, 15, 15 | 24, 16, 16 |

1. **Two rounds' lead over four does not hold on fresh cases.** At 0.01% the fresh cases give 15 for 2 early
   rounds and 16 for 4. Case by case, 2 rounds is closer in 12, further in 10 and level in 3 (median ratio
   1.00). Over all 50 cases the two are 35 against 34.
2. **Early value labels still help at the 0.1% line.** On fresh cases both mixes have 24 of 25 against 23
   for no value labels. The case is seed 32, at 0.110% with none, 0.074% with 2 rounds and 0.046% with 4.
   Over 50 cases the mixes have 48 against 45.
3. **At 0.01%, value labels are a wash on fresh cases.** Two rounds against none: closer in 15, further in 10
   (median ratio 0.92), but 15 against 15 under the line (gains seeds 27 and 43, loses 35 and 39). Four
   rounds against none: closer in 17, further in 8, median ratio 0.68, 16 against 15.
4. **The weights are the same with or without value labels.** 13 exact and 15 every visible fault in all
   three fresh arms; 4 rounds adds seed 37. On the original cases every arm from 0 to 4 rounds has 19 and 16.
5. **One early round loses seed 17.** It ends at 0.108%, just over the line (2 rounds: 0.086%), so 1 round
   has 23 of 25. Under 0.01% it has 18, like 4 rounds. Seed 20, the other case value labels fixed, stays
   fixed (0.0082%).
6. **The fresh cases are harder for the weights but not for behaviour.** 15 every visible fault against 19
   on the original cases, yet 23 to 24 under 0.1% in both. With 2 early rounds, ten fresh winners miss a
   visible fault; nine of them are under 0.1% and seven under 0.05%.
7. **Seed 49 is a new kind of failure.** Every arm ends between 0.34% and 0.40%, on a wrong triple each
   time, although its committee holds the true triple. The true triple refitted on the same labels reaches
   0.0055% to 0.011%. Every fresh committee holds the true triple; seed 14's is still the only one that
   doesn't.
8. **Cost:** median about 870 seconds per case in every arm.

## Interpretation

- *Inference:* #45's 2-against-4 difference was noise. The finding that replicates is coarser: a few rounds
  of value labels early move about 3 of 50 cases under the 0.1% line and cost nothing in weights. Two
  rounds and four are equally good; one round is slightly worse; six (#45) is worse.
- *Inference:* 2 early rounds is still a reasonable default because it uses half the value queries of 4 for
  the same result, not because it behaves better.
- *Inference:* seed 49 says the remaining failures are not all screening failures. Its true triple is in the
  committee, and the winner is by construction the member that scores best on the labels, so the
  committee's own fit of the true triple scores no better than a wrong triple there, while the told fit of
  the same triple is far closer. That points to the fit, not the labels or the screen. Roadmap #47 looks at it.

## Evidence

- `evidence/fresh-switch/fresh-early-0.json`, `fresh-early-2.json`, `fresh-early-4.json`: seeds 25 to 49,
  every case and round, with both told fits.
- `evidence/fresh-switch/early-1.json`: 1 early round on seeds 0 to 24.
- The 2- and 4-round arms on seeds 0 to 24 are #45's `evidence/switch-point/early-2.json` and #44's
  `evidence/early-values/early-values.json`.
