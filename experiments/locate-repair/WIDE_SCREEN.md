# The wider screen on all 50 three-fault cases (roadmap #48)

**Date:** November 18, 2026 · **Status:** done · **Cost:** CPU, about 51 core-hours (four runs side by side,
about 11 hours of wall time)

[#47](SCREEN_MISSES.md) found that the two clear three-fault failures left (seeds 14 and 49) lose the true
triple at the very first step: none of its three pairs is among the best 200 pairs, so it is never extended
to a triple. Extending the best 1,000 pairs fixed both on 8 cases. This experiment runs that wider screen on
all 50 three-fault cases (seeds 0 to 49) with #45's 2-early-round rule and compares it with the 200-pair runs
from [#45](SWITCH_POINT.md) (seeds 0 to 24) and [#46](FRESH_SWITCH.md) (seeds 25 to 49). Everything except
the number of pairs extended is the same.

```bash
python experiments/locate-repair/early_values.py --early 2 --pairs 1000 --seed-start 0 --seeds 50 --output runs/wide-screen
python experiments/locate-repair/wide_compare.py
```

The runs were split into four seed ranges in parallel; `wide-screen.json` merges the 50 per-case files.

## Results (measured)

**All 50 cases, after all 64 labels:**

| | 200 pairs | 1,000 pairs | Adaptive (below) |
|---|---:|---:|---:|
| Under 0.1% disagreement | 48 | **50** | **50** |
| Under 0.01% | 35 | 37 | 37 |
| Exact broken entries | 29 | 30 | 31 |
| Every visible fault | 34 | 35 | 36 |
| True triple in the committee | 38 | 41 | 40 |
| Worst case | 0.61% | 0.079% | 0.086% |
| Median disagreement | 0.0048% | 0.0047% | 0.0048% |
| Median seconds per case | 894 | 3,497 | |
| CPU hours, all 50 | 12.3 | 51.5 | 14.0 |

1. **50 of 50 under 0.1%, and nothing lost at that line.** The two cases gained are seeds 14 and 49, the two
   clear failures, as in #47. The worst case is now seed 17 at 0.079% (it was 0.086% with 200 pairs), whose
   true triple is in neither committee.
2. **At the 0.01% line the gain is small and comes with losses.** Six cases cross it downward (10, 12, 14,
   35, 47, 49) and four upward (6, 20, 22, 33), for a net of 2. All four losses stay under 0.06% (worst: seed
   20 at 0.052%).
3. **Two of the four losses end on exactly the same weights as before.** Seeds 6 and 33 pick the same
   winning triple with both screens and differ only in its fitted values. Across all 24 cases where both
   screens pick the same winner, the final disagreement still moves by a factor of 0.09 to 3.7 (20 of the 24
   within a factor of 2). Over all 50, 18 cases end closer, 21 further, and the median ratio is 1.00.
4. **The wider screen brings the true triple into 3 more committees and out of none.** It gains seeds 14, 47
   and 49, the three cases #47 found are never extended. The 9 cases that still lack it (3, 5, 17, 28, 30,
   32, 37, 41, 44) are exactly the 9 that #47 found lose it in the tie at the cut to 200.
5. **It costs about 4 times as much** (median 3,497 against 894 seconds per case), almost all of it fitting
   about 3.5 times as many triples (median 43,000 against 12,000).
6. **An adaptive rule gets the same headline at 1.1 times the cost.** Before any targeted label, the
   200-pair committee's best member gets every practice label right in 48 of the 50 cases. The two where it
   does not are seeds 14 and 49, exactly the two failures. Keeping the 200-pair run unless that happens, and
   rerunning only those cases with 1,000 pairs, gives 50 of 50 under 0.1%, 37 under 0.01%, 31 exact and 36
   every-visible-fault, in 14.0 CPU hours against 51.5. This is scored from the two runs above, not a run of
   its own; a real run would make the same choices, since the flag depends only on the first 200-pair
   committee.

## Interpretation

- *Inference:* the wider screen does the one thing #47 showed it should: it brings the never-extended true
  triples into the committee, which fixes the two clear failures. It does nothing for the 9 lost in the tie,
  as expected, since widening the screen adds more triples to that tie.
- *Inference:* most of the movement at the 0.01% line is not the screen. A different committee asks
  different targeted labels, and cases that end on the same weights move by up to a factor of 3.7 from that
  alone. So the net +2 at 0.01% should not be read as a gain from the wider screen; the gain at 0.1% (two
  clear failures fixed) is the real effect.
- *Inference:* a committee whose best member cannot fit the practice labels is a cheap, reliable sign that
  the screen has missed the true triple. On these 50 cases it fires on exactly the two cases where it matters
  and nowhere else. Whether it stays that precise on new cases is roadmap #49.
- **Three-fault default from here:** 200 pairs, widened to 1,000 when no committee member fits every practice
  label, with #45's 2-early-round rule.

## Evidence

- `evidence/wide-screen/wide-screen.json`: all 50 cases with 1,000 pairs (`pairs_kept` 1000, `early_rounds`
  2), round-by-round disagreement, winner, weight scores and seconds.
- `evidence/wide-screen/comparison.json`: the comparison and the adaptive score, from `wide_compare.py`.
- 200-pair baselines: `evidence/switch-point/early-2.json` and `evidence/fresh-switch/fresh-early-2.json`.
