# A larger committee instead of a wider screen (roadmap #52)

**Date:** November 19, 2026 · **Status:** done · **Cost:** CPU, about 4.9 core-hours (four runs side by side)

[#51](RANK_MISSES.md) found that all nine fresh cases whose 1,000-pair committee lacks the true triple lose it
in the cut to 200 triples, not in the pair screen: the true triple is extended, fits every practice label, and
ranks 204th to 4,839th among other perfect fits. This experiment keeps 200 pairs but keeps the best 1,000
extended triples in the committee, on the same 16 fresh three-fault cases (seeds 50 to 99) that
[#50](END_FLAG.md)'s end flag picks.

```bash
python experiments/locate-repair/early_values.py --early 2 --committee 1000 --only 52 55 61 62 63 64 66 68 75 76 81 83 89 92 93 96 --output runs/big-committee
python experiments/locate-repair/big_committee_check.py
```

`--committee 1000` is new in `early_values.py`; everything else is #45's 2-early-round rule. The runs were
split into four seed groups in parallel; `big-committee.json` merges them, and `check.json` holds the scores.

## Results (measured)

**Seeds 50 to 99, all 50 cases; the rule reruns only the 16 the end flag picks:**

| | 200 pairs, 200 triples | Wider screen (#50) | **Larger committee** | Larger committee, then wider screen if still flagged |
|---|---:|---:|---:|---:|
| Under 0.1% | 45 | 47 | **49** | **50** |
| Under 0.01% | 28 | 29 | 33 | 34 |
| Exact broken entries | 26 | 28 | 33 | 35 |
| Every visible fault | 30 | 32 | 37 | 39 |
| True triple in the committee | 35 | 37 | 43 | 45 |
| Worst case | 0.44% | 0.44% | 0.21% | 0.088% |
| Seconds per rerun (median) | | 2,808 | 1,103 | |
| CPU hours, all 50 | 13.9 | 26.5 | 18.8 | 22.5 |

1. **The larger committee fixes four of the five fresh failures, three of them on the exact broken weights.**
   Seeds 63 (0.23% to 0.061%), 76 (0.16% to 0.0020%) and 96 (0.44% to 0.032%) end exact; 83 (0.13% to 0.059%)
   ends under the line on other weights. The wider screen fixed two (61 and 83).
2. **#51's prediction held exactly.** The true triple reaches the committee in the eight cases #51 said it
   would (52, 55, 62, 63, 66, 76, 93, 96), and not in seed 68, whose true triple ranked 4,839th.
3. **It is cheaper than the wider screen and barely dearer than the plain run.** A rerun takes a median 1,103
   seconds, against 2,808 for the wider screen and 983 for the same 16 cases' first 200-triple run. The screen
   still fits about 12,000 triples rather than about 45,000, so keeping five times as many adds only about 12%.
4. **The one it misses is the kind the wider screen fixes.** Seed 61 gets worse (0.13% to 0.21%): its true
   triple is never extended from 200 pairs, and #50's 1,000-pair rerun put it at 0.010% on the exact weights.
5. **After the larger committee the end flag means something.** It stays up in only 5 of the 16 (61, 75, 81,
   83 and 89), against 13 of 16 after the wider screen, and the one remaining failure is among them. Rerunning
   those five with 1,000 pairs (#50's runs) puts all 50 fresh cases under 0.1% (worst 0.088%), with 35 exact,
   in 22.5 CPU hours.
6. **No flagged case crosses 0.1% upward.** Ten end closer to the original network, four further (61, 81, 92,
   93; apart from 61 all stay under 0.035%) and two the same.

## Interpretation

- *Inference:* the two failure kinds need two different fixes. A true triple cut on the tie-break needs a
  larger committee; one never extended needs a wider screen. Trying the cheap committee first and the
  expensive screen only when the flag is still up handles both, and on these 50 cases it costs 1.6 times the
  plain run.
- Only the 16 fresh cases were rerun, so the two-step rule is scored on seeds 50 to 99 alone. Roadmap #53 runs
  the larger committee on the 11 flagged cases of seeds 0 to 49 (their 1,000-pair runs exist from #48) to
  score it on all 100.

## Files

- `early_values.py --committee N` and one test for it.
- `big_committee_check.py` scores the three rules from committed evidence; one test checks the evidence.
- `evidence/big-committee/big-committee.json` (16 reruns) and `check.json`.
