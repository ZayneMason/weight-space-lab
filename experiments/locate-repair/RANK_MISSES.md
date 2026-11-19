# Where the three remaining failures lose the true triple (roadmap #51)

**Date:** November 19, 2026 · **Status:** done · **Cost:** CPU, about two core-hours (four runs side by side)

[#50](END_FLAG.md) left three fresh three-fault cases above 0.1% (seeds 63, 76 and 96) even after rerunning
them with 1,000 pairs, and in all three the true triple never reached the committee. Nine of the 16 reruns
lacked it. [#47](SCREEN_MISSES.md) found two ways to lose it: the true triple is never extended because none of
its pairs makes the cut, or it is extended, fits every practice label, and is cut from the committee of 200
among many triples that fit just as well. Only the first can be fixed by extending more pairs. This
experiment runs #47's `screen_ranks.py` on the nine to see which it is.

```bash
for s in 52 55 62 63 66 68 76 93 96; do python experiments/locate-repair/screen_ranks.py $s runs/rank-misses; done
python experiments/locate-repair/rank_misses.py
```

`screen_ranks.py` rebuilds the 200-pair committee before any targeted label and reports where the true
triple's pairs rank in the pair screen, whether the true triple is among the extended triples, and its rank
in the committee's order (accuracy on the practice labels, then the fit's cost, then the coarse fit's rank).

## Results (measured)

| Seed | Best rank of a true pair (of 3,321) | Triples extended | True triple's rank (committee keeps 200) | 200th member's practice accuracy | Disagreement, 200 → 1,000 pairs |
|---|---:|---:|---:|---:|---:|
| 52 | 47 | 13,149 | 897 | 100% | 0.038% → 0.038% |
| 55 | 51 | 10,924 | 204 | 100% | 0.041% → 0.039% |
| 62 | 24 | 11,374 | 413 | 100% | 0.068% → 0.084% |
| **63** | 13 | 10,646 | **253** | 100% | **0.23% → 0.23%** |
| 66 | 10 | 13,636 | 705 | 100% | 0.090% → 0.063% |
| 68 | 70 | 12,204 | 4,839 | 100% | 0.021% → 0.023% |
| **76** | 49 | 12,967 | **840** | 100% | **0.16% → 0.10%** |
| 93 | 13 | 11,109 | 350 | 100% | 0.026% → 0.026% |
| **96** | 5 | 12,768 | **494** | 100% | **0.44% → 0.44%** |

1. **All nine are lost in the cut, none in the pair screen.** In every case one of the true triple's pairs is
   among the best 70 of 3,321, so the true triple is extended even with 200 pairs. It fits every practice
   label, and so do at least 200 other triples; it ranks 204th to 4,839th and is cut.
2. **That is why the wider screen did not help them.** Extending 1,000 pairs adds more triples that tie on
   the practice labels; it cannot lift a true triple that was already extended and cut on the tie-break.
3. **A committee of 1,000 triples would hold the true triple in eight of the nine,** all three failures
   included (63 at 253, 76 at 840, 96 at 494). Only seed 68 (4,839) would still miss.
4. **Across both sets the cut is now the main loss.** With #47's 12 cases this makes 18 of 21 checked misses
   lost in the cut and 3 never extended (14, 47 and 49); the wider screen brought the true triple into the
   committee for those three.

## Interpretation

- *Inference:* once the true triple is extended, the screen has nothing left to say about it. The practice
  labels cannot tell hundreds of perfect fits apart, and the tie-break (the cheaper fit first) put the true
  triple 204th or lower in all nine. A committee that keeps more of the perfect fits is the direct fix.
- Roadmap #52 runs the 16 fresh end-flag cases with a committee of 1,000 triples from 200 pairs, to see
  whether it fixes 63, 76 and 96 and at what cost.

## Files

- `rank_misses.py` summarises the nine rank files; one test checks the evidence.
- `evidence/rank-misses/rank-*.json` (nine cases) and `summary.json`.
