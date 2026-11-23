# Rebuild the committee from the final labels (roadmap #66)

**Date:** November 23, 2026 · **Status:** done · **Cost:** 19.2 core-hours

#65 left 9 of 50 cases over 0.01%, and in 7 of them the repair edits a wrong weight from a committee that never held
the true triple. That committee was screened from the 1,024 practice labels alone. #65 saved every case's final
labels: those 1,024, the 64 asked during the repair and the 8 region-split inputs, 1,096 in all. This experiment
reruns the pair screen and the triple committee on all 1,096 labels for the same 50 cases (seeds 250 to 299), with
the options of the case's last stage, picks the winner the way the repair does and takes its middle (#64). It
compares the result with #65's region arm, the best arm there.

```bash
python experiments/locate-repair/rebuild_committee.py --output runs/rebuild --workers 4
python experiments/locate-repair/rebuild_committee.py --score runs/rebuild
```

The winner is the member that fits the most labels, ties going to the smallest fit cost, as in the repair. The
committee's own order breaks the same ties by the coarse 7-value fit instead, so its first member can differ; it
is scored too. Results are in `evidence/rebuild-committee/`.

## Results (measured)

| | Under 0.1% | Under 0.01% | Median | True triple in committee | Exact (all three right weights) |
|---|---|---|---|---|---|
| #65 region arm, then middle | 48 | 41 | 0.0026% | 35 | 30 |
| Rebuilt committee, winner, then middle | 46 | 42 | 0.0029% | **46** | 30 |
| Rebuilt committee, first member, then middle | 47 | 42 | 0.0029% | 46 | **34** |

1. **The final labels get the true triple into the committee far more often.** It enters 12 committees that
   lacked it (250, 251, 253, 254, 263, 265, 270, 279, 282, 285, 287, 292) and leaves one (258), so 46 of 50
   committees now hold it. Only 256, 258, 261 and 296 lack it.
2. **Picking it is now the bottleneck.** The winner edits all three right weights in 30 cases, the same count as
   before: 251, 263, 265 and 292 become exact, while 255, 258, 259 and 269 stop being exact. In 16 cases the true
   triple is in the committee and is not picked. In 12 of them it fits all 1,096 labels exactly as well as the
   winner, so the tie-break by fit cost chooses the wrong member. In the other 4 (255, 259, 269, 277) the true
   triple's fit reaches fewer labels than the winner's.
3. **The committee's own order picks better but does not end closer.** The committee ranks the true triple first
   in 34 cases, and its first member is exact in 34. Against the winner it ends closer in 12 cases and further in
   12, with the same median. Seed 287 is exact under the first member but further than the winner (0.084% against
   0.037%).
4. **The 0.01% line barely moves.** 263, 270, 279 and 292 cross under it; 255, 258 and 259 cross over, and 258
   and 269 cross over 0.1%. Of the 8 cases left over 0.01%, 4 have the true triple in the committee and edit a
   wrong weight (255, 259, 269, 287), 3 have a committee without it (258, 261, 296) and 1 is exact (260). Seed 296
   stays near 0.8%.

## Interpretation

- Screening on the final labels mostly fixes recall: the true triple was missing from 15 committees and is now
  missing from 4. The failures that remain are mostly a choice among members that fit every label equally well,
  and the current tie-break by fit cost makes that choice worse than the committee's coarse order (30 against 34
  exact). This is inferred from 50 cases; the disagreement numbers do not favour either tie-break.
- As in #39, picking the wrong triple often costs little: most of the 16 misses still end under 0.01%. The
  trade-off is between editing the true weights and matching the original network, and the two rules here land
  about level on the second.
- The 4 cases where the true triple fits fewer labels than the winner suggest the continuous fit stops short on
  that triple (inferred; the true values fit every label by construction). #67 saves every member's fit, compares
  tie-breaks directly and refits the near-best members from more starts.
