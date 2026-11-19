# A third rerun with 1,000 pairs and 1,000 triples (roadmap #56)

**Date:** November 19, 2026 · **Status:** done · **Cost:** CPU, about 2.4 core-hours (three runs side by side)

[#55](UNSEEN_MISSES.md) found that on the three unseen cases the two-step rule leaves flagged (seeds 124, 126
and 149), the true triple is cut among perfect practice fits, and predicted that only a rerun with 1,000 pairs
and 1,000 triples together would keep one of them (149). This experiment runs that setting on the three cases.

```bash
python experiments/locate-repair/early_values.py --early 2 --pairs 1000 --committee 1000 --only 124 --output runs/both-wide-124
python experiments/locate-repair/both_wide_check.py
```

(One run per seed, side by side.) `early_values.py` gains `--three-step`, which adds this setting as a last
rerun after `--two-step`'s two. `both-wide.json` holds the three runs; `check.json` holds the scores.

## Results (measured)

| Seed | Two-step rule | 1,000 pairs and 1,000 triples | True triple kept | Fits every label | CPU hours |
|---|---:|---:|---|---|---:|
| 124 | 0.012% | 0.029% | no | yes | 0.80 |
| 126 | 0.069% | 0.050% | no | no | 0.76 |
| 149 | 0.045% | **0.0055%** | **yes** | yes | 0.87 |

1. **The prediction holds.** Seed 149's true triple is kept and the repair lands on the exact broken weights
   (0.045% to 0.0055%). Seeds 124 and 126 still lack it, as #55's ranks (1,441 and 9,277) said.
2. **On all 50 unseen cases, adding the step** gives 50 under 0.1%, 36 under 0.01% and 28 exact (two-step: 50,
   35 and 27), worst case 0.050% (0.069%), for 2.4 more CPU hours (22.2 against 19.8, 12% more).
3. **Seed 124 gets worse and the flag can no longer see it.** Its repair now fits every label but disagrees
   more (0.012% to 0.029%), so the end flag goes quiet on a worse answer.
4. **The step costs about 0.8 CPU hours per flagged case,** under #56's 1.5-hour bar, and fires on 3 of 50.

## Interpretation

- *Inference:* the third step is worth keeping as an option (`--three-step`), not as the default. It fixes the
  one case whose true triple is within reach of a 1,000-triple cut, and nothing else; on 124 it trades a
  flagged repair for an unflagged, slightly worse one.
- This confirms #55's reading: past a point, more candidates add more ties. Roadmap #57 looks for a better
  order among perfect fits instead.

## Files

- `both_wide_check.py` scores the third step against #54's finals; two tests (the stage order and the
  evidence).
- `evidence/both-wide/both-wide.json` and `check.json`.
