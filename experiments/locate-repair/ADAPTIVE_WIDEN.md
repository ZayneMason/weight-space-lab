# Does the adaptive widening flag hold on fresh cases? (roadmap #49)

**Date:** November 18, 2026 · **Status:** done · **Cost:** CPU, about 14 core-hours (four runs side by side)

[#48](WIDE_SCREEN.md) found that extending the best 1,000 pairs instead of 200 puts all 50 three-fault cases
(seeds 0 to 49) under 0.1%, and that a cheap flag picked out exactly the two cases that needed it: before any
targeted label, no member of the 200-pair committee gets every practice label right. This experiment builds
that rule into `early_values.py` and runs it on 50 fresh three-fault cases (seeds 50 to 99) with #45's
2-early-round rule.

```bash
python experiments/locate-repair/early_values.py --early 2 --widen-to 1000 --seed-start 50 --seeds 50 --output runs/adaptive-widen
python experiments/locate-repair/adaptive_check.py
```

`--widen-to 1000` builds the 200-pair committee, and rebuilds it from 1,000 pairs only if its best member
misses a practice label. The runs were split into four seed ranges in parallel; `adaptive-widen.json` merges
them, and `check.json` holds the scores below.

## Results (measured)

**50 fresh cases, after all 64 labels:**

| | Seeds 50 to 99 (this run) | Seeds 0 to 49, 200 pairs (#45, #46) |
|---|---:|---:|
| Flag fired (widened) | **0** | 2 |
| Under 0.1% disagreement | 45 | 48 |
| Under 0.01% | 28 | 35 |
| Exact broken entries | 26 | 29 |
| Every visible fault | 30 | 34 |
| True triple in the committee | 35 | 38 |
| Worst case | 0.44% | 0.61% |
| Median disagreement | 0.0080% | 0.0048% |
| CPU hours, all 50 | 13.9 | 12.3 |

1. **The flag never fired.** In all 50 fresh cases some member of the first 200-pair committee gets every
   practice label right, so no case was widened and this run is the plain 200-pair screen.
2. **Five fresh cases miss 0.1%, and the flag caught none of them.** Seeds 61 (0.13%), 63 (0.23%), 76
   (0.16%), 83 (0.13%) and 96 (0.44%). In every one the true triple is missing from the committee, yet some
   member fits all the practice labels. #48's flag was right on seeds 0 to 49 by luck of those two cases, not
   because a missing true triple shows up as a practice miss.
3. **The fresh cases are harder than seeds 0 to 49 on every line.** The true triple is missing from 15 of 50
   committees here against 12 of 50 before, and under 0.01% falls from 35 to 28. The 15 are 52, 55, 56, 61,
   62, 63, 65, 66, 68, 69, 76, 83, 86, 93 and 96.
4. **A different flag catches all seven failures across both sets: score the chosen repair after the
   targeted labels, not the committee before them.** After all 64 labels, the winner in each of the five
   fresh failures still gets some labels wrong (final practice accuracy 0.982 to 0.995). The same holds for
   seeds 14 and 49. Flagging a case when its finished 200-pair repair misses any label fires on 16 of the 50
   fresh cases and 11 of seeds 0 to 49, and misses no failure in either set.
5. **On seeds 0 to 49 that end flag keeps #48's headline at about 1.9 times the cost.** Scored from #48's two
   runs, rerunning the 11 flagged cases with 1,000 pairs gives 50 of 50 under 0.1%, 37 under 0.01%, 31 exact
   and 36 every-visible-fault, in 22.8 CPU hours against 12.3 for 200 pairs and 51.5 for 1,000 pairs on every
   case. It costs more than #48's start flag because it fires on 9 cases that were already fine and because a
   flagged case pays for both runs.

## Interpretation

- *Inference:* the start flag is the wrong signal. A committee can fit every practice label without holding
  the true triple, because the practice labels rarely touch the inputs where the wrong triple and the broken
  network part ways. The 64 targeted labels are chosen to find exactly those inputs, so a repair that still
  misses some of them after the targeted rounds is the one to distrust.
- Roadmap #50 runs the 16 flagged fresh cases with 1,000 pairs to see whether the end flag plus a wider rerun
  fixes all five fresh failures, as it does on seeds 0 to 49.

## Files

- `early_values.py --widen-to N` (the start-flag rule) and one test for it.
- `adaptive_check.py` scores both flags from committed evidence; one test checks the evidence.
- `evidence/adaptive-widen/adaptive-widen.json` (50 cases) and `check.json`.
