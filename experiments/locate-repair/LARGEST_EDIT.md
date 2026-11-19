# The largest-edit tie-break in the full repair (roadmap #58)

**Date:** November 19, 2026 · **Status:** done · **Cost:** CPU, about 24 core-hours (six hours on four cores)

#57 found that, among perfect practice fits, preferring the triple whose largest single scaled edit is smallest
keeps 5 of 12 lost true triples inside the 200-triple committee, where the current coarse-fit order keeps none.
This experiment puts that order into the whole repair and reruns the two-step rule (#54) on the same 50 unseen
three-fault cases, seeds 100 to 149. The only change is how the committee breaks ties.

```bash
python experiments/locate-repair/early_values.py --early 2 --two-step --tie-break largest-edit \
    --only 100 104 108 ... --output runs/largest-edit/g0          # four groups side by side
python experiments/locate-repair/early_values.py --early 2 --two-step --tie-break largest-edit \
    --merge runs/largest-edit/g*/early-values.json --output runs/largest-edit/merged
python experiments/locate-repair/largest_edit_check.py
```

`triple_committee.committee(..., tie_break="largest-edit")` orders by practice accuracy, then by the smallest
largest single edit (`largest_edit`, the same feature `tie_breaks.py` records). `--tie-break coarse` is the
default and leaves every earlier run unchanged. The comparison arm is #54's coarse-order run,
`evidence/two-step-unseen/two-step-unseen.json`. Results are in `evidence/largest-edit/`.

## Results (measured)

| 50 cases, seeds 100 to 149 | Coarse order (#54) | Largest-edit order |
|---|---:|---:|
| Under 0.1% disagreement | **50** | 49 |
| Under 0.01% | **35** | 34 |
| Exact three weights | 27 | **28** |
| Every visible fault found | **35** | 34 |
| True triple in the final committee | **40** | 35 |
| Median disagreement | 0.0045% | 0.0043% |
| Worst case | **0.069%** (126) | 0.32% (104) |
| Cases rerun (bigger committee / wider screen) | 15 / 6 | 18 / 9 |
| CPU hours | **19.8** | 24.3 |

1. **One case crosses the 0.1% line, none come back under it.** Seed 104 goes from 0.0067% on two true
   weights to 0.32% on none; its committee never holds the true triple at any of the three stages.
2. **At 0.01%** the new order gains 5 cases (102, 131, 133, 139, 148) and loses 6 (104, 110, 113, 119, 122, 144).
   Per case it is closer in 18 and further in 22; 10 end at the same disagreement.
3. **True weights.** One case gains a true weight (133, two to three). Eight lose one or more (104, 118, 119,
   122, 124, 136, 140, 143). The true triple drops out of the final committee in six cases (104, 111, 122, 128,
   141, 143) and comes in in one (131).
4. **The loss starts at the first stage.** With the plain 200-pair, 200-triple committee, the new order holds
   the true triple in 27 cases against 29, and 41 cases are under 0.1% after that stage against 46. Three
   more cases need the third, widest stage, which is where the 23% extra CPU goes.
5. **The three hard cases stay lost.** Seeds 124, 126 and 149 still end without the true triple and still miss
   a practice label. They finish at 0.068%, 0.046% and 0.043% (coarse: 0.012%, 0.069%, 0.045%). In #57 none of
   these three was among the five the new order kept; the five were all #51 cases (55, 63, 76, 93, 96).
6. **#51's nine lost cases were not rerun.** The roadmap item asked for them too. They are the only cases where
   #57 showed an upside, so the result here says nothing about them.

## Interpretation

- **The order stays off by default.** #58's own rule was to switch only if no case got worse at 0.1%. Seed 104
  does, and the new order is also more expensive and loses the true triple more often than it saves it.
- **#57's measurement did not carry over (inferred).** #57 ranked fits on committees built after the targeted
  labels, where the true triple already fits every label. In the full repair the order also picks the first
  committee, which picks the next labels, so a different early committee changes everything after it. The
  first-stage numbers (item 4) fit that reading. Which feature of the lost cases makes small-edit ties wrong is
  not clear: their largest fault sizes (1.68 to 2.05) are not different from the cases the order gained (1.77
  to 2.17).
- **The useful version, if any, is narrower.** #57's five wins were at 1,000 pairs, the stage only a few cases
  reach. Using the new order only there would leave the first two stages, and so 41 of the 50 cases here,
  exactly as they are for the 44 of these 50 cases that never reach it. That is #59.
