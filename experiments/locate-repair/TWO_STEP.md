# The two-step rule on all 100 three-fault cases (roadmap #53)

**Date:** November 19, 2026 · **Status:** done · **Cost:** CPU, about 3.3 core-hours (four runs side by side)

[#52](BIG_COMMITTEE.md) found that on the 50 fresh three-fault cases (seeds 50 to 99) a committee of 1,000
triples fixes most failures cheaply, and that rerunning the few it leaves flagged with the 1,000-pair screen
puts all 50 under 0.1%. This experiment runs the larger committee on the 11 end-flag cases of seeds 0 to 49
and scores the whole rule on all 100 cases:

1. run with 200 pairs and a committee of 200 triples;
2. if the finished repair misses any of its 64 labels, rerun keeping 1,000 triples;
3. if that repair still misses a label, rerun with 1,000 pairs.

```bash
python experiments/locate-repair/early_values.py --early 2 --committee 1000 --only 6 14 17 21 28 30 32 37 41 47 49 --output runs/two-step
python experiments/locate-repair/two_step_check.py
```

Step 3 reuses [#48](WIDE_SCREEN.md)'s 1,000-pair runs for seeds 0 to 49 and [#50](END_FLAG.md)'s for seeds 50
to 99. `old-big.json` holds the 11 new runs; `check.json` holds the scores.

## Results (measured)

**All 100 three-fault cases, after all 64 labels:**

| | 200 pairs, 200 triples | Wider screen when flagged (#50) | **Two-step rule** |
|---|---:|---:|---:|
| Under 0.1% disagreement | 93 | 97 | **100** |
| Under 0.01% | 63 | 66 | **75** |
| Exact broken entries | 55 | 59 | **69** |
| Every visible fault | 64 | 68 | **79** |
| True triple in the committee | 73 | 78 | **90** |
| Worst case | 0.61% | 0.44% | **0.088%** |
| CPU hours | 26.2 | 49.3 | 42.9 |

1. **The two-step rule puts all 100 cases under 0.1%,** with 75 under 0.01% and 69 on the exact broken
   weights, at 1.6 times the cost of the plain run. The previous best rule (#50) left three cases above 0.1%
   and cost more.
2. **On seeds 0 to 49 the larger committee alone is not enough.** It fixes neither of the two failures (14 and
   49, whose true triple is never extended from 200 pairs) and pushes seed 17 over the line (0.086% to
   0.12%). It still helps below the line: 38 under 0.01% and 32 exact, against 35 and 29.
3. **The second flag catches every case the committee leaves broken.** After the larger committee it fires on
   5 of the 11 (14, 17, 32, 47 and 49): the three whose true triple is never extended (14, 47 and 49), seed 17,
   and seed 32. Their 1,000-pair runs end between 0.0037% and 0.079%.
4. **The expensive screen runs on only 10 of 100 cases.** The first flag fires on 27 cases and the second on
   10 of those (5 in each set).
5. **The closest call is a case no flag touches.** The worst result, seed 70 at 0.088%, is a plain 200-triple
   run whose repair fits every label, so the rule never reruns it; seed 95 (0.074%) is the same. Of the 25
   cases left above 0.01%, 16 were never flagged, so the end flag is a 0.1% alarm, not a 0.01% one.

## Interpretation

- *Inference:* the two failure kinds found in #47 and #51 (a true triple cut on the tie-break, and one never
  extended) each have a fix, and the end flag, checked again after the cheap fix, picks out the cases that need
  the expensive one. On these 100 cases no failure got past both flags, but seeds 70 and 95 show the margin
  under 0.1% is thin for cases the flag never sees.
- These 100 seeds are the ones the rule was designed on. Roadmap #54 builds the rule into `early_values.py` as
  one option and runs it end to end on 50 unseen cases (seeds 100 to 149).

## Files

- `two_step_check.py` scores the rule on all 100 cases from committed evidence; one test checks the evidence.
- `evidence/two-step/old-big.json` (11 larger-committee runs, seeds 0 to 49) and `check.json`.
