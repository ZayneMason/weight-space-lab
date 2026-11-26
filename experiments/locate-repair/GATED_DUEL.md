# Duel only when the labels can tell the two apart (roadmap #81)

**Date:** November 26, 2026 · **Status:** done on #79's 100 cases and #80's 50, from saved duels; the two cases still
over the line are #82 · **Cost:** seconds; nothing was refitted and no new label was drawn, against the roadmap's
estimate of under one core-hour.

#80's two-round chooser duels #71's pick against #65's repair on up to 32 labels where the two disagree, and takes
#65's repair when it answers more of them right. It kept seed 319's pick, which #79's one-round chooser had lost, but
made 9 of 150 cases further from the original, all in duels with 1 to 7 split inputs. `gated_duel.py` scores, from the
duels #80 saved, a second round that runs only when the duel has at least `k` split inputs (counted on unlabeled
inputs, so a skipped duel asks no labels) and takes #65's repair only when it wins by at least `m` labels. The rule is
picked on #79's 100 cases (fewest cases made worse, then most made better, then fewest labels) and checked on #80's 50.

```bash
python experiments/locate-repair/gated_duel.py
```

Evidence is in `evidence/gated-duel/check.json`: the chosen rule, every chooser's counts on both sets, the whole grid
(k from 1 to 64, m from 1 to 8) on both, and each duel's split inputs.

## Results (measured)

| | #71 alone | #79 one round, ties last | two rounds (#80) | two rounds, k = 32 |
|---|---|---|---|---|
| #79's 100: under 0.01% / exact | 97 / 78 | 98 / 79 | 98 / 79 | **98 / 79** |
| #79's 100: closer / further than #71 alone | | 4 / 1 (319) | 6 / 6 | **3 / 0** |
| #79's 100: labels asked | 579 | 1,155 | 1,601 | 1,123 |
| #80's 50: under 0.01% / exact | 50 / 39 | 50 / 39 | 50 / 39 | 50 / 39 |
| #80's 50: closer / further | | 0 / 0 | 0 / 3 | 0 / 0 |
| #80's 50: labels asked | 296 | 449 | 704 | 392 |

1. **A minimum of 8 split inputs removes every loss.** Any `k` from 8 to 32, with `m` from 1 to 4, makes 3 cases
   closer on #79's set (400, 402, 414) and none further. The selection picks `k = 32`, `m = 1`, the cheapest of those.
   With `k` of 4 or less and `m` up to 4, 304 (5 split inputs, won 5 to 1) is lost; with `m = 8` 402 (won 18 to 14) is lost.
2. **It keeps everything that crosses the line.** 414 goes from 0.069% to 0.0035% (133 split inputs, won 32 to 0) and
   400 from 0.041% to 0.013% (75, won 28 to 4), as with #79. 319 keeps boundary (0.0016%): the duel has 84 split
   inputs and #65's repair loses it 0 to 32.
3. **It gains one #79 missed and drops two #79 took.** 402 improves from 0.0078% to 0.0049% (37 split inputs, won 18
   to 14), which #79 left alone. 426 (0.0063% to 0.0053%) and 430 (0.0051% to 0.0014%) are no longer taken; their
   duels had 5 and 4 split inputs. All three stay under the line either way.
4. **On #80's 50 it does nothing, which is right.** No duel there reaches 32 split inputs with #65's repair winning, so
   every pick is #71's, as it should be on a set where #65's repair improves no case.
5. **Fewer labels than #79.** 1,123 against 1,155 on #79's set and 392 against 449 on #80's, because round one never
   includes #65's repair and only 17 of 100 duels (3 of 50) reach 32 split inputs, each asking 32 labels.

## Interpretation

- The gated two-round chooser is the first rule in this line that makes no case worse across 150 cases while keeping
  414 and 400. It should replace #79's one-round chooser as the default, with `k = 32`, `m = 1`.
- The rule was picked on #79's cases, and #80's cases have nothing for it to gain, so the check shows only that it does
  no harm on fresh cases. Whether `k = 8` and `k = 32` differ anywhere is untested; they give the same picks here.
- 430 is the price: a real gain (to a quarter of the disagreement) on only 4 split inputs. Spending a few more labels
  when the split set is small, rather than skipping, might keep it (inferred: not run).
- The two cases over the line are unchanged: 331 (every candidate edits exactly the three broken weights and lands at
  0.062% to 0.077%) and 400 (0.013%). Both have the right weights with the wrong values. Asking more labels of the
  kind #65's stage asks for values, only in cases where every candidate edits the same weights, is #82.
