# Run the whole repair, fixed in advance, on fresh seeds (roadmap #90)

**Date:** November 27, 2026 · **Status:** done on 50 fresh seeds (700 to 749); breaking the duel's ties, where the
true triple ties at the top in every case that ends short of exact, is #91 · **Cost:** 30.9 core-hours recorded, 26.0
of them #65's and #71's label stages, as the roadmap expected.

Every step since #84 was built after looking at the same 200 cases, so #89's 200 of 200 under the 0.01% line is a fit
to those cases, not a rate. `fresh_end.py` runs the whole repair on 50 seeds nothing had been run on, with every step
and threshold fixed before the run:

- #83's first part as it ran there (#65's stage with its network saved, #71's stage and #81's gated duel), giving the
  pick and #65's 1,096 final labels;
- where the pick misses one of those labels, #84's keep-one search; elsewhere #89's keep-two search; both with #85's
  fitter, keeping the triples that fit every label;
- #88's refit duel on the pick and those triples;
- #82's flat-line step on the duel's winner only where the duel keeps the pick (#89's after-the-fact reading).

The result was fixed in advance as the network after the last step. Beside it the score reports the pick, the duel's
winner and the arm with the line step on every winner (#89's order).

```bash
python experiments/locate-repair/fresh_end.py --output runs/e90 --part 0 --parts 4    # first part, parts 0 to 3
python experiments/locate-repair/fresh_end.py --repair runs/e90 --workers 4           # search, duel and line step
python experiments/locate-repair/fresh_end.py --score runs/e90
```

Evidence is in `evidence/fresh-end/`: `fresh-end.json` (every case's search, candidates, duel, chosen network, line
step and seconds), `check.json` (the score), `first-part/` (each case's first part), `duel/` (each duel's rounds and
labels right) and `labels/` (#65's final labels for each case).

## Results (measured)

| 50 fresh cases | under 0.01% | exact | median disagreement | over the line |
|---|---|---|---|---|
| #65's own repair | 42 | 25 | 0.0023% | 701, 704, 716, 720, 731, 742, 744, 748 |
| The pick (#71, #81) | 49 | 34 | 0.0011% | 744 (0.0125%) |
| Search and refit duel | 50 | 36 | 0.0008% | none |
| Then the line step where the duel keeps the pick (the result) | **50** | **36** | **0.0008%** | none |
| The line step on every winner instead (#89's order) | 50 | 36 | 0.0008% | none |

| The same repair, shaped on these cases (#89) | under 0.01% | exact |
|---|---|---|
| The pick, 200 cases | 196 (98%) | 156 (78%) |
| The result, 200 cases | 200 (100%) | 159 (80%) |
| The pick, 50 fresh cases | 49 (98%) | 34 (68%) |
| The result, 50 fresh cases | 50 (100%) | 36 (72%) |

1. **Every fresh case ends under the line.** The one pick over it, 744 (one broken weight found, 0.0125%), misses a
   label, so the keep-one search runs; its 9,481 triples hold 3 perfect fits, and the duel takes the true triple at
   0.0001%, exact. The worst result is 735 at 0.0034%, against 331's 0.0067% on the 200.
2. **The repair moves most cases closer, a few slightly further.** Against the pick, 26 cases get closer, 9 further
   and 15 stay. The largest falls after 744 are 728 (0.0033% to 0.0005%), 748 (0.0034% to 0.0008%) and 716 (0.0031%
   to 0.0006%); the largest rises are 736 (0.0017% to 0.0025%), 739 (0.0008% to 0.0016%), 724 (0.0005% to 0.0010%)
   and 701 (0.0011% to 0.0016%), all well under the line.
3. **Exact moves 3 up and 1 down.** 706, 723 and 744 gain the missing broken weight; 736 loses one, its pick exact at
   0.0017% and the duel's winner two broken at 0.0025%. 13 results have two broken weights and 1 (727) has one.
4. **The search always holds the true triple when it finds anything.** 48 of 50 cases have at least one perfect fit
   (median 3, most 81), and the true triple is among them in all 48. The other two, 725 and 729, have exact picks and
   no other triple fits every label.
5. **The duel's misses are ties.** In 43 of the 48 duels more than one candidate gets the most asked labels right, and
   the true triple's refit is always among them. The winner is then the pick if it is tied, otherwise the first
   candidate in the search's order. Every one of the 14 results short of exact is such a case: the true triple's refit
   tied at the top and lost on order (for example 713, where 80 refits get all 172 asked labels right). Where the duel
   takes another triple (35 of 50), it is the true triple in 24.
6. **The line step barely matters here.** 10 winners have a flat direction; it runs on the 5 the duel kept and moves 2
   (717 from 0.0008% to 0.0006%, 722 from 0.0003% to 0.0002%). Running it on every winner instead would differ on one
   case, 719, where it would have gone from 0.0004% to 0.0001%. #89's reading that it hurts refits does not show on
   these 50.
7. **Labels.** #65's 1,096 a case, 413 by #71's and #81's stages, 2,602 by the duels (48 cases; median 15, most 256
   for 748, 2 ask none) and 160 by the line step (32 on each of 5).
8. **Cost.** 30.9 core-hours: 26.0 for the first part, 3.0 for the three keep-one searches (about an hour each),
   1.1 for 47 keep-two searches (median 76 seconds) and 0.9 for the duels and line steps.

## Interpretation

- The whole repair, fixed in advance, keeps its headline on cases it was not shaped on: 50 of 50 under the 0.01%
  line, including the one pick over it. Exact is lower than on the 200 (72% against 80%), but so is the pick's (68%
  against 78%), so the gap starts before the search (inferred: these seeds have more two-broken picks, 15 of 50
  against 34 of 200).
- Fixing an exact triple is no longer limited by the search: the true triple is a perfect fit wherever there is one.
  It is limited by the duel's tie rule. The duel's asked labels separate the pick from the refits, but not the refits
  from each other, because they agree on the rest of #71's pool. Asking labels where the tied refits split, from a
  fresh pool, is #91.
- On these 50 the line step's placement is close to neutral either way, so #89's choice to keep it only on kept picks
  costs one small move (719) and buys none here (inferred: the refits on the 200 had larger flat moves).
