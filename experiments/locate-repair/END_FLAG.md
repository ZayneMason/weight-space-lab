# Widen when the finished repair still misses a label (roadmap #50)

**Date:** November 19, 2026 · **Status:** done · **Cost:** CPU, about 12.6 core-hours (four runs side by side)

[#49](ADAPTIVE_WIDEN.md) found that #48's start flag never fires on 50 fresh three-fault cases (seeds 50 to 99)
and misses all five that end above 0.1%. A different flag caught every failure in both sets: after all 64
labels, the chosen repair still gets at least one label wrong. This experiment reruns the 16 fresh cases that
end flag picks with the 1,000-pair screen and scores the full rule on all 100 three-fault cases: 200 pairs
everywhere, and the 1,000-pair result wherever the end flag fires.

```bash
python experiments/locate-repair/early_values.py --early 2 --pairs 1000 --only 52 55 61 62 63 64 66 68 75 76 81 83 89 92 93 96 --output runs/end-flag
python experiments/locate-repair/end_flag_check.py
```

Seeds 0 to 49 reuse #48's 1,000-pair runs for their 11 flagged cases. The fresh runs were split into four
seed groups in parallel; `end-flag.json` merges them, and `check.json` holds the scores below.

## Results (measured)

| | 200 pairs everywhere | End-flag rule |
|---|---:|---:|
| **Seeds 50 to 99:** under 0.1% | 45 | **47** |
| under 0.01% | 28 | 29 |
| Exact broken entries | 26 | 28 |
| Every visible fault | 30 | 32 |
| True triple in the committee | 35 | 37 |
| CPU hours | 13.9 | 26.5 |
| **All 100 cases:** under 0.1% | 93 | **97** |
| under 0.01% | 63 | 66 |
| Exact broken entries | 55 | 59 |
| Every visible fault | 64 | 68 |
| Worst case | 0.61% | 0.44% |
| CPU hours | 26.2 | 49.3 |

1. **The wider rerun fixes two of the five fresh failures, not all five.** Seeds 61 (0.13% to 0.010%) and 83
   (0.13% to 0.0065%) end on the exact broken weights, because 1,000 pairs bring the true triple into the
   committee. Seeds 63 (0.23%), 76 (0.16% to 0.10%) and 96 (0.44%) stay above 0.1%, and in all three the true
   triple is still missing from the 1,000-pair committee. Seeds 63 and 96 end exactly where they did.
2. **Widening never pushes a fine case over the line.** Of the 16 fresh reruns, 6 end closer to the original
   network, 3 further (62: 0.068% to 0.084%; 64 and 68 move by 0.002 points or less) and 7 the same. None of
   the 11 flagged cases in seeds 0 to 49 crosses 0.1% either, though seed 6 goes from 0.003% to 0.011%, just
   over the 0.01% line.
3. **Across all 100 cases the rule gives 97 under 0.1% at 1.9 times the cost of 200 pairs everywhere.** The
   three left are the three in finding 1.
4. **The flag stays up after the rerun in 13 of the 16 fresh cases, so it cannot say whether widening
   worked.** Only 61, 83 and 92 end on a repair that fits every label. Four cases (64, 75, 81, 89) keep the
   flag with the true triple in the committee and end under 0.011%, so a missed label does not always mean a
   wrong repair.
5. **Nine of the 16 fresh committees still lack the true triple at 1,000 pairs** (52, 55, 62, 63, 66, 68, 76,
   93 and 96). Six of them end under 0.1% anyway, on other weights.

## Interpretation

- *Inference:* the end flag is a good alarm and a poor stopping rule. It catches every failure in 100 cases,
  but in this set most of what it flags were fine already, and a second look at 1,000 pairs does not reach
  the true triple for the three that are not.
- *Inference:* for 63, 76 and 96 the loss is earlier than the committee cut. Roadmap #51 finds where each true
  triple's pairs rank in the pair screen, to tell whether a still wider screen or a better pair score is the
  fix.
- The rule does not meet the bar #50 set (all five fresh failures under 0.1%), so #48's start flag is not
  replaced yet. The end flag is the better of the two: it catches every failure the start flag catches and
  the five it misses.

## Files

- `end_flag_check.py` scores the end-flag rule from committed evidence; one test checks the evidence.
- `evidence/end-flag/end-flag.json` (16 reruns with 1,000 pairs) and `check.json`.
