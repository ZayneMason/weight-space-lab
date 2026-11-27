# Run the whole repair with the gated duel and the flat-line step on fresh seeds (roadmap #83)

**Date:** November 27, 2026 · **Status:** done on 50 fresh cases (seeds 600 to 649); the one case a better search
could fix is #84 · **Cost:** 24.2 core-hours recorded (24.17 for the two stages, 0.015 for the line step), against the
roadmap's estimate of about 22.

#81's gated duel and #82's flat-line step were chosen on #79's and #80's 150 cases, where #82 put all 150 under the
0.01% line. `fresh_line.py` runs the whole cheap repair on 50 seeds nothing had been run on, exactly as #80 did
(#65's stage with its network saved, then #71's stage and its chooser on the same pool), then the gated duel on that
pick and the flat-line step on the duel's pick, from #65's final labels. The result was fixed in advance as the
network after both steps.

```bash
python experiments/locate-repair/fresh_line.py --output runs/l83 --part 0 --parts 4   # parts 0 to 3, one per core
python experiments/locate-repair/fresh_line.py --score runs/l83
python experiments/locate-repair/fresh_line.py --truth runs/l83
```

Evidence is in `evidence/fresh-line/`: `fresh-line.json` (every case's candidates with their network edits, the duel,
the line step's singular values and segments, and seconds), `check.json` (the score) and `truth.json` (the true broken
weights and the pick's weights refitted on #65's final labels in the two cases over the line).

## Results (measured)

| 50 fresh cases (600 to 649) | under 0.01% | under 0.1% | exact | median disagreement | over the line |
|---|---|---|---|---|---|
| #65's repair alone | 37 | 49 | 32 | 0.0029% | 13 cases |
| #71's chooser | 48 | 50 | 38 | 0.00105% | 611 (0.0118%), 641 (0.0366%) |
| + gated duel (#81) | 48 | 50 | 38 | 0.00105% | the same |
| + flat-line middle (#82) | 48 | 50 | 38 | 0.0011% | the same |
| + 32 labels along the line (the result) | **48** | **50** | **38** | 0.0011% | the same |

| Earlier fresh runs | seeds | under 0.01% | under 0.1% | exact |
|---|---|---|---|---|
| #72, original repair | 300 to 349 | 49 | 50 | 39 |
| #78, cheap repair | 400 to 449 | 48 | 50 | 39 |
| #80, cheap repair with #65's network saved | 500 to 549 | 50 | 50 | 39 |
| #83, with the gated duel and the line step | 600 to 649 | 48 | 50 | 38 |

1. **Neither new step changed an outcome.** The gated duel took #65's repair in no case: 9 of 50 duels reached 32
   split inputs and #65's repair lost all 9 (its best was 13 labels to 19, on 641). 5 of 50 picks have a flat
   direction (604, 610, 622, 633, 644); in 4 the disagreement is unchanged at every step and in 644 it goes from
   0.0010% to 0.0012%, under the line. No case moved under or over the line.
2. **#71's stage carries the result, as before.** Over #65's repair it makes 44 cases closer and 5 further (616, 618,
   623, 626, 639), moves 11 under the line and none over it.
3. **The rate matches the earlier fresh runs.** 48 of 50 under 0.01% and 38 exact, against 48 to 50 and 39 on #72's,
   #78's and #80's seeds. #82's 150 of 150 was on the cases its step was shaped on.
4. **611: even the true weights are over the line on these labels.** The pick edits two of the three broken weights
   and a wrong third. Refitting the true broken weights on #65's final 1,096 labels fits all of them and lands at
   0.0118%, the same as the chosen network; the pick's own weights on those labels land at 0.0216%. The direction is
   not flat (smallest singular value 0.24 of the largest).
5. **641: a perfect fit exists and no candidate has it.** The broken weights include a hidden bias. Refitted on #65's
   final labels the true weights get all 1,096 right and land at 0.0039%, under the line; the pick's weights get 8
   wrong (99.27%) and land at 0.0339%. All four of #71's candidates, the bigger committee among them, edit the same
   wrong triple with one broken weight. The gate that runs the bigger committee (#68's pick misses a label) fired
   here and in 7 other cases, where the pick was exact in 6.
6. **Cost.** 24.17 core-hours for the two stages (#80's 22.3 on other seeds), 0.015 core-hours (52 seconds) for the
   line step on all 50. With both steps the repair asks 734 labels against #71's 286: 288 for the 9 duels and 160
   for the 5 flat lines.

## Interpretation

- The two steps are safe and cheap but add nothing held out, so the repair's rate is still about 48 to 50 of 50 under
  the line. Keep them as defaults only if the labels they ask are free; otherwise #71's chooser alone gives the same
  outcomes here. Each has fixed cases only on the set it was shaped on.
- The two failures are different kinds. 611 is a labels problem: the true weights fit all 1,096 labels, but the labels do
  not pin their values, and the pick's direction is not flat enough for #82's test. 641 is a search problem:
  the labels already tell the true triple from the pick, and the search never tries it.
- 641 points to a cheap next step: when no candidate fits every label, try every triple that keeps the pick's
  best-supported weight and fits all labels (inferred: the pick's hidden_weight 8 is a broken weight, so such a
  search would include the true triple; whether it finds it, and what it costs, is #84).
