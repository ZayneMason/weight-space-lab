# Where the two-step rule still loses the true triple (roadmap #55)

**Date:** November 19, 2026 · **Status:** done · **Cost:** CPU, about 3 core-hours (six runs side by side)

On 50 unseen three-fault cases ([#54](UNSEEN.md)) the two-step rule put every case under 0.1%, but three
repairs (seeds 124, 126 and 149) still miss a label after both reruns, and their true triple is never in the
committee. This experiment rebuilds each case's committee from the practice labels, as [#51](RANK_MISSES.md)
did, with 200 pairs and with 1,000 pairs, and records where the true triple drops out.

```bash
for s in 124 126 149; do
  python experiments/locate-repair/screen_ranks.py $s runs/unseen-misses
  python experiments/locate-repair/screen_ranks.py $s runs/unseen-misses 1000
done
python experiments/locate-repair/unseen_misses.py
```

`screen_ranks.py` takes a new optional third argument, the number of pairs to extend (default 200).

## Results (measured)

| Seed | Best true pair | 200 pairs: true triple | 1,000 pairs: true triple | Triples tied or better (1,000 pairs) |
|---|---:|---|---|---:|
| 124 | 11 | extended, rank 1,177 | extended, rank 1,441 | 1,411 |
| 126 | 281 | never extended | extended, rank 9,277 | 7,766 |
| 149 | 217 | never extended | extended, rank 298 | 246 |

1. **The pair screen is not where any of the three is lost.** With 1,000 pairs all three true triples are
   extended. On seeds 126 and 149 the best true pair ranks just past the 200 cut (281 and 217); on seed 124 it
   ranks 11th.
2. **All three are cut among perfect practice fits.** The true triple fits all practice labels in each case,
   and hundreds to thousands of other triples fit them as well: 1,411, 7,766 and 246 at 1,000 pairs.
3. **No step of the rule keeps any of them,** because the rule never uses 1,000 pairs and 1,000 triples
   together. That combination would keep seed 149 (rank 298). It would not keep 124 (rank 1,441) or 126 (rank
   9,277).
4. **A bigger screen makes seed 124 harder, not easier.** Its true triple drops from rank 1,177 to 1,441 when
   1,000 pairs are extended, because more perfect fits come in ahead of it.

## Interpretation

- *Inference:* this is #51's failure (a tie among perfect fits) with a wider margin, not a new one. A larger
  committee or a wider screen only helps while the true triple is within reach of the cut; past that, more
  candidates add more ties. A better order among perfect fits, not a bigger list, is what would fix 124 and 126.
- All three stay under 0.1% because the repairs they do find are close substitutes (final disagreement 0.012%,
  0.069% and 0.045%).
- Roadmap #56 tests the one setting this predicts would help: a third rerun with 1,000 pairs and 1,000
  triples, on the three cases.

## Files

- `unseen_misses.py` summarises the six rank files; one test checks the evidence and the stage rule.
- `evidence/unseen-misses/rank-{seed}.json` (200 pairs), `rank-{seed}-1000.json` (1,000 pairs) and
  `summary.json`.
