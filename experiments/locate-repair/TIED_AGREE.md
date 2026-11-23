# Find out why the tied members agree (roadmap #69)

**Date:** November 23, 2026 · **Status:** done · **Cost:** 18.4 core-hours

#68 ended with 41 of its 50 width-16 three-fault cases (seeds 250 to 299) still having two or more committee members
that fit every label, and in most rounds no input in a pool of 20,000 split them. This experiment keeps every one of
those tied members and asks what they are.

```bash
python experiments/locate-repair/tied_agree.py --output runs/agree --workers 4
python experiments/locate-repair/tied_agree.py --score runs/agree
```

For each of the 41 tied cases the script reruns #68 exactly (same committees, refits, pools and labels; it reproduces
#68's final pick in all 41) and keeps every member tied at the top at the end. It compares each with the winner
(#68's pick):

- **Mirror:** the member edits the winner's entries once each output-layer edit is moved to logit 0. Only the
  difference of the two logits matters, so these are the same repair (#22).
- **Extra entries:** the entries it edits that the winner does not, mirrors folded, with each entry's edit size in its
  layer's scale (#57's scaled edit) and the state of its hidden unit on the usual 1M-point probe in the member's own
  network: never active, always active, or mixed.
- **Agreement:** how often its network and the winner's disagree on the probe.

Scoring also puts each of the case's three faults back into the original network on its own and measures the probe
disagreement. A fault that changes no probe output is **silent**: the broken network computes the same function on
the probe whether or not it is fixed. Results are in `evidence/tied-agree/`.

## Results (measured)

| | Cases | Tied members | Mirrors | Other members | Largest tie |
|---|---|---|---|---|---|
| Only mirrors tie | 28 | 98 | 98 | 0 | 8 |
| One fault is silent | 4 (253, 254, 277, 285) | 465 | 9 | 456 | 200 |
| Neither | 9 (265, 267, 268, 270, 273, 281, 282, 283, 287) | 109 | 27 | 82 | 20 |
| All | 41 | 672 | 134 | 538 | 200 |

1. **No tied member is more than 0.01% from the winner.** Across all 672, the largest disagreement with the winner on
   the 1M-point probe is 0.009% (seed 287), and 242 match it on every probe input. So whichever tied member the repair
   picks, it ends up with the same network to within 0.01%.
2. **The mirror count is exact everywhere.** In all 41 cases the number of tied mirrors is exactly 2^k for a winner with
   k output-layer edits. In 28 cases those are the only tied members. Mirrors are fitted separately, so 52 of the 134
   still differ from the winner on a few probe inputs (all under 0.01%).
3. **A silent fault explains the big ties.** Only 4 of the 41 cases have a silent fault, and they hold 456 of the 538
   non-mirror members, including the two largest ties (254 with 200, 253 with 175) and 285 (77). In 254, 277 and 285,
   279 of the 283 non-mirror members edit both audible faults and spend the third slot somewhere else, and 245 of them
   move that third entry by less than 0.1 of its layer's scale. In 253 the pattern is looser: 89 of 173 edit both
   audible faults, and the winner itself finds only one of the three.
4. **The other 9 cases tie genuinely different repairs.** Their 82 non-mirror members mostly edit other weights by
   full-size steps (the extra edits' median per case runs from 0.03 to 3.4 of the layer's scale), and only 12 of the
   82 edit all three faults. Yet each still matches the winner on the probe to within 0.009%. These ties are small
   (3 to 20 members).
5. **The extra entries are not on idle units.** Of the 683 extra entries, 655 sit on hidden units that are active on
   part of the probe and inactive on the rest; only 2 sit on a unit that is never active, none on one that is always
   active, and 26 are output biases.

## Interpretation

- The tie-breaking work in #67 and #68 was aimed at a problem that does not cost anything on the probe: every member
  that fits every label is within 0.01% of the one the repair picks (measured on the probe; that this holds off the
  probe is inferred). Breaking these ties further would not move the 0.01% line.
- Ties come from three places. The output mirror symmetry is exact and harmless. A silent fault leaves a free slot,
  so any third weight the fit can leave almost unchanged also fits, which is why the biggest ties appear only where a
  fault is silent (4 cases, inferred from the overlap; the script does not test the cause directly). And in 9 cases
  the network has more than one way to undo the damage with three weights, all of which agree on the probe.
- "Exact" counts the right weights, but a silent fault cannot be told from any other unchanged weight by labels at
  all, so in those 4 cases no amount of labelling could make the repair edit it. That is a limit of the measure, not
  of the repair.
- What still stands between the repair and the 0.01% line is the 6 cases #68 left over it (258, 259, 260, 261, 269,
  296), none of which has a tie problem: three committees lack the true triple, one true triple fits fewer labels,
  and two exact repairs have values that are off. That is #70.
