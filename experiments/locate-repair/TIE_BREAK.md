# Ask labels where the duel's tied refits split (roadmap #91)

**Date:** November 28, 2026 · **Status:** done on #90's 50 fresh cases and #89's 200; not adopted, since it changes
which tied network wins but not how many are exact; the fixed repair at width 32 is #92 · **Cost:** 2.8 core-hours
recorded, against the roadmap's estimate of a few.

On #90's 50 fresh cases, 43 refit duels end with more than one candidate getting the most asked labels right, because
every refit is fitted on the labels it is then scored on. The true triple's refit is always among the tied, and all 14
results short of exact are such ties, lost on the duel's order. `tie_break.py` reruns #88's duel exactly and, where it
ends tied, breaks the tie:

- it draws a fresh pool of 200,000 inputs, and each round asks up to 64 where the tied networks split;
- it scores every tied network on the new labels before refitting it on them, so a refit is never scored on labels it
  was fitted to, then refits the tied triples on everything asked;
- it keeps the networks with the highest score so far, and stops when one is left, none split, or after 4 rounds; if
  more than one is left, the duel's rule decides among them;
- then, as in #90, #82's flat-line step runs on the winner only where it is the pick.

This was fixed in advance and ran on #90's 50 fresh cases, then on #89's 200 (scored against #89's networks under
#90's order, with the line step only where the duel kept the pick).

```bash
python experiments/locate-repair/tie_break.py --set fresh --output runs/b91f --workers 4
python experiments/locate-repair/tie_break.py --set shaped --output runs/b91s --workers 4
python experiments/locate-repair/tie_break.py --score runs/b91f runs/b91s
```

Evidence is in `evidence/tie-break/`: `fresh.json` and `shaped.json` (every case's duel, tie-break rounds, winner,
chosen network, line step and seconds) and `check.json` (the score of each set against the network before).

## Results (measured)

| | under 0.01% | exact | median disagreement | over the line |
|---|---|---|---|---|
| 50 fresh cases, #90's result | 50 | 36 | 0.0008% | none |
| 50 fresh cases, with the tie-break | 50 | **35** | 0.00065% | none |
| 200 shaped cases, #89 under #90's order | 200 | 159 | 0.0007% | none |
| 200 shaped cases, with the tie-break | **199** | 159 | 0.0006% | 331 (0.0620%) |

1. **The duels reproduce.** The rerun's duel picks the same winner as #90 on all 48 fresh duels and as #89 on all 197
   shaped ones.
2. **Most tied refits do not split even on 200,000 fresh inputs.** Of the 43 fresh ties, 20 have no input where the
   tied networks disagree, and the rest split on few (median 1 label asked, most 44 for 735; 127 in all). On the 200,
   77 of 170 ties have no split (449 labels in all).
3. **When they split, the labels do not favour the true triple.** In the 23 fresh ties with a round, the true
   triple's refit survives to the end in 15 and is knocked out in 8. The tie-break changes the winner in 11 fresh and
   52 shaped cases and settles on a single network in 9 and 44. No fresh case gains exact and one loses it: 737 goes
   from three broken weights at 0.0009% to two at 0.0011%. On the 200, 3 gain exact (317, 325, 535) and 3 lose it
   (311, 335, 444).
4. **Behaviour moves a little, mostly closer.** Fresh: 10 closer and 3 further, the largest moves 713 (0.0012% to
   0.0002%) and 743 (0.0009% to 0.0001%), the largest rise 737. Shaped: 39 closer and 19 further, the largest falls 428
   (0.0034% to 0.0007%) and 522 (0.0025% to 0.0002%), the largest rises 513 (0.0010% to 0.0049%) and 331.
5. **331 crosses the line because the line step no longer runs.** Its pick (exact) sits at 0.0621% before the line
   step, which takes it to 0.0067%. The tie-break asks 2 labels, both right for the true triple's refit and wrong for
   the pick, and takes the refit. That refit is at 0.0620% and differs from the pick on 7 of the million probe inputs,
   so it is nearly the same network, but under #90's order the line step runs only on a kept pick.
6. **Cost.** 0.67 core-hours for the 50 fresh cases and 2.09 for the 200 (inferred: most of it rerunning the duels'
   refits, since most ties never reach a tie-break refit).

## Interpretation

- The ties are not a shortage of labels. The tied refits are, almost everywhere, the same function: on most cases no
  input in 200,000 tells them apart, and where some do, the true triple's refit is as likely to lose as to win. Which
  of them is "exact" cannot be read from behaviour (inferred: the labels they split on sit at noise level, a few points
  in a million). This is #39's and #69's finding again, now on the duel's ties: the repair can match the original
  without editing the broken weights.
- So the tie-break is not adopted. It costs one case its place under the line (331) through the line step's gating,
  and leaves exact where it was (35 against 36 fresh, 159 against 159 shaped).
- 331 also shows that #90's gating of the line step is fragile: a refit that is nearly the pick misses the step the
  pick would get (inferred: running the line step on any flat winner, as #89 did, would keep 331 under the line).
- Exact repairs past this point need information that is not in the input-output labels, such as a prior over which
  weights break, and the repair already reaches the original's behaviour. The next test of the fixed repair is scale:
  the same order at width 32, #92.
