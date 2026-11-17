# Rescoring two- and three-fault repairs by behaviour, round by round (roadmap #37)

**Date:** November 17, 2026 · **Status:** done (two- and three-fault; the one-fault benchmark is not
rescored) · **Cost:** CPU, about two and a quarter hours (four runs in parallel)

[#36](LOOKALIKES.md) found that scoring a three-fault repair by its entries undercounts: 22 of 25
winners behave like the original network, against 19 that edit every visible broken weight. This
experiment asks the same question of the two-fault repair, and asks it after every round of targeted
labels instead of only the last.

```bash
python experiments/locate-repair/behaviour_scores.py --mode pairs --output runs/behaviour
python experiments/locate-repair/behaviour_scores.py --mode pairs --fixed 1.6 --output runs/behaviour
python experiments/locate-repair/behaviour_scores.py --mode triples --output runs/behaviour
python experiments/locate-repair/value_check.py --seeds 5 8 12 20 13 15
```

## Method

- **Two faults:** #35's setup (H = 32, random sizes in [1.0, 2.2] spreads, 25-point pair screen,
  then 8 rounds of 8 targeted labels). A control re-runs it with every fault at 1.6 spreads.
- **Three faults:** #34's setup (H = 16, random sizes, committee of 200 triples, uniform pool).
- After every round, the current winner is run on a million fresh uniform inputs. Its
  **disagreement** is the share it labels differently from the uncorrupted network.
- A repair **behaves like the original** when its disagreement is under 0.1%.
- `value_check.py` fits only the two truly broken entries on the practice labels, to see whether the
  leftover disagreement comes from the choice of weights or from their values.

## Results (measured)

**After all 64 targeted labels, 25 cases each:**

| Setup | Exact broken entries | Every visible fault | **Behaves like the original (under 0.1%)** | Under 0.01% | Median disagreement | Corrupted network |
|---|---:|---:|---:|---:|---:|---:|
| Two faults, random sizes | 19 | 23 | **20** | 9 | 0.015% | 4.6% |
| Two faults, all at 1.6 (control) | 19 | 23 | **23** | 8 | 0.030% | 4.3% |
| Three faults, random sizes | 16 | 19 | **22** | 16 | 0.0056% | 12.6% |

The two-fault random row reproduces #35's 19 and 23. The three-fault row reproduces #34 and #36.

**Round by round, "behaves like the original" (under 0.1%) against "every visible fault":**

| Targeted labels | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Two faults: behaviour | 11 | 16 | 18 | 19 | 21 | 21 | 20 | 20 | 20 |
| Two faults: visible | 10 | 15 | 19 | 22 | 23 | 22 | 23 | 23 | 23 |
| Three faults: behaviour | 3 | 10 | 16 | 19 | 20 | 21 | 22 | 22 | 22 |
| Three faults: visible | 3 | 5 | 9 | 10 | 12 | 18 | 19 | 19 | 19 |

1. **For three faults, behaviour comes much sooner than the entries.** After 16 labels, 16 of 25
   repairs already behave like the original, against 9 that have every visible fault. The entry
   count needs 40 labels to reach 18. The behaviour count reaches 19 after 24 labels.
2. **For two faults, the behaviour score is stricter, not kinder.** It ends at 20 of 25, below the
   23 that find every visible fault. Four winners (seeds 5, 8, 12 and 20) edit both exact broken
   weights and still disagree on 0.13% to 0.17%. One winner on the wrong weight (seed 11) behaves
   like the original, at 0.057%. Seed 23 is wrong on both counts, at 0.21%.
3. **Those four misses are set by the value, not the weights.** Fitting only the true pair on the
   practice labels lands every value within 0.03 spreads of the original in seeds 5, 8 and 12, and
   still disagrees on 0.12% to 0.17%. In seed 20 one value is 0.17 spreads off (0.29%). In seeds 13
   and 15, for comparison, the same fit lands within 0.008 spreads and disagrees on 0.002% or less.
4. **The 1.6 control shows the same split.** Two right-weight winners (seeds 12 and 16) disagree on
   0.12% and 0.11%, and two wrong-weight winners (seeds 6 and 11) behave like the original.
5. **"Under 1%" separates nothing.** Every two-fault winner is under 1% even before any targeted
   label, and every three-fault winner from 8 labels on. The useful lines are 0.1% and 0.01%.
6. **Cost:** median 292 seconds per two-fault case and 776 per three-fault case, including the
   million-input probe after every round.

## Interpretation

- *Inference:* the two scores measure different failures. The entry score misses lookalikes that
  compute the same function, which matters most for three faults. The behaviour score also catches
  repairs that pick the right weights but set them slightly wrong, which matters most for two.
  Neither alone is the whole picture, so both columns should be reported.
- *Inference:* 1,024 practice labels can only resolve a disagreement of about 1 in 1,024, or 0.1%.
  A right-weights repair that is off by 0.1% to 0.2% is at the limit of what the labels can see, and
  the targeted labels do not help, because they are chosen to split the committee over which weights
  to edit, not over the values. That suggests queries aimed at the winner's own value next
  ([roadmap #38](../../docs/ROADMAP.md)).
- *Inference:* the three-fault repairs end closer to the original (median 0.0056%) than the two-fault
  ones (0.015%), probably because #34 refits each triple with #29's continuous margin fit, while the pair repair
  takes the best point on a 200-step grid with ties broken toward the corrupted value. This was not tested
  directly.

## Evidence

- `evidence/behaviour-scores/behaviour-pairs.json`, `behaviour-pairs-fixed1.6.json` and
  `behaviour-triples.json`: every case, every round.
- `evidence/behaviour-scores/value-check.json`: the true-pair fits for seeds 5, 8, 12, 20, 13 and 15.
