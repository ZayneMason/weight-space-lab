# Lab notebook

A running record of what the lab has, what each experiment showed, and what is
broken or unfinished. Newest entry first. Every entry separates **measured** results
from **interpretation**. Numbers come from committed evidence or from commands that
were actually run on the date shown.
The [roadmap](ROADMAP.md) turns the open questions into ranked experiments.

---

## 2026-11-24 — Roadmap #72: run the whole repair on 50 fresh cases

**What ran.** 50 fresh width-16 three-fault cases (seeds 300 to 349). First `early_values.py --width 16 --early 2
--two-step --middle --extra` (#65's repair), then `fresh_repair.py`, which runs #71 unchanged on #65's final labels:
#68's pick, restarts, boundary labels, the bigger committee where #68's pick misses a label, and the disagree chooser.
The chooser's pick was fixed in advance as the result. 37.6 core-hours recorded (19.1 and 18.4), against an estimate
of about 20. Write-up: [FRESH_REPAIR.md](../experiments/locate-repair/FRESH_REPAIR.md).

**Measured.**
- Whole repair: 49 of 50 under 0.01%, 50 under 0.1%, 39 exact. #65 alone: 41, 50, 37. #68's pick: 46, 50, 39.
- The chooser equals the probe's best choice on the line in every case; the one case over it, 331 (0.068%), is over
  for every candidate.
- Gained over #68: 308, 348, 349, all by boundary labels winning the vote outright. None lost; 341 moves from 0.0013%
  to 0.0014%.
- 244 labels asked in all, a median of 3; 10 cases have no input where the candidates differ.
- The bigger committee ran on 5 cases and rescued none.

**Interpretation.** #71's gate and chooser did not overfit: held out they reach 49 of 50 against 47 of 50 on the
cases they were tuned on (the fresh set is a little easier, #68 alone 46 against 44). Which fix rescues a case varies
between sets, so choosing from labels is the part that carries over. The repair now costs about 0.75 core-hours a
case, mostly building candidates.

---

## 2026-11-24 — Roadmap #71: choose among the fixes from labels alone

**What ran.** `pick_fixes.py` reran #68 on all 50 of #67's width-16 three-fault cases (reproducing #68's pick in all
50), built restarts and boundary-label candidates everywhere and a bigger-committee candidate where #68's pick misses
one of its own labels (5 cases), and chose among them with fresh labels: 64 uniform ones, or up to 32 where the
candidates disagree. 20.2 core-hours. Write-up: [PICK_FIXES.md](../experiments/locate-repair/PICK_FIXES.md).

**Measured.**
- Disagree chooser: 47 of 50 under 0.01% (258, 259, 269 gained, none lost), 49 under 0.1%, 39 exact, the same as
  choosing by the probe. It asks 341 labels in all, a median of 3 per case.
- Uniform chooser: every candidate gets all 64 labels right in all 50 cases, so it never switches.
- One case gets worse: 272, from 0.0007% to 0.0013%. Where the chooser and the probe differ (12 cases), every
  candidate is under 0.01% and the gap is under 0.001%.
- Boundary labels everywhere make 4 cases slightly worse (all still under 0.01%); restarts and the bigger committee
  make none worse.
- 260, 261 and 296 stay over the line, for #70's reasons.

**Interpretation.** A few labels where the candidates disagree are enough to choose among the fixes, so #70's upper
bound is reachable without the probe (on these 50). The gate and chooser were tuned on these cases; next, #72 runs
the whole repair on 50 fresh ones.

---

## 2026-11-24 — Roadmap #70: fix the six cases still over 0.01%

**What ran.** `fix_six.py` reran #68 on the 6 cases it left over 0.01% (reproducing #68's pick in all 6) and tried
three fixes on each: a committee twice as large with twice the pairs kept (on the 3 cases whose committee lacks the
true triple), 20 random restarts for the winner and the true triple, and 4 rounds of 16 labels next to the fitted
decision boundary. It also fitted the true triple to #68's labels as a yardstick. 5.9 core-hours. Write-up:
[FIX_SIX.md](../experiments/locate-repair/FIX_SIX.md).

**Measured.**
- 258: the bigger committee brings the true triple in and it wins, 0.139% to 0.0019%.
- 259: restarts let the true triple win, 0.024% to 0.0027%.
- 269: boundary labels move the exact repair's values from 0.014% to 0.0017%.
- 260: the true triple itself reaches only 0.065% on #68's labels; boundary labels take the repair to 0.044%.
- 261: the true triple fits only 97.1% of #68's labels (0.37%); the best repair stays at 0.021%.
- 296: the true triple would reach 0.037%, but its best pair ranks 168th and even a committee of 400 lacks it.
- No arm made any case worse; boundary labels improved all 6.

**Interpretation.** The six leftovers are four different problems (committee, fit, values, and a true triple that
cannot fit the labels). Each fix rescues one case; taking the best per case would put 47 of 50 under 0.01%, but that
choice uses the probe. Next: #71 picks among the fixes from labels alone.

---

## 2026-11-23 — Roadmap #69: find out why the tied members agree

**What ran.** `tied_agree.py` reran #68 on its 41 tied cases (reproducing #68's final pick in all 41), kept every
member tied at the top, and compared each with the winner: mirror or not, which other entries it edits and by how
much, the state of their hidden units, and its disagreement with the winner on the 1M-point probe. Scoring also put
each fault back into the original network alone to find silent faults. 18.4 core-hours. Write-up:
[TIED_AGREE.md](../experiments/locate-repair/TIED_AGREE.md).

**Measured.**
- 672 tied members; the largest disagreement with the winner on the probe is 0.009%, and 242 match it exactly.
- The tied mirror count is exactly 2^k in all 41 cases; 28 cases tie only mirrors (98 members).
- 4 cases have a silent fault (253, 254, 277, 285) and hold 456 of the 538 non-mirror members. In 254, 277 and 285,
  279 of 283 edit both audible faults and spend the third slot elsewhere, 245 of them by under 0.1 of the scale.
- The other 9 cases tie 82 non-mirror members (3 to 20 per case) that edit other weights by full steps; 12 edit all
  three faults.
- 655 of the 683 extra entries are on units active on part of the probe; 2 are on a never-active unit, 26 are output
  biases.

**Interpretation.** Every tie is harmless on the probe: any member the repair could pick is within 0.01% of the one
it does. The big ties come from silent faults leaving a free slot (inferred from the overlap), the rest from the
mirror symmetry and a few genuinely different three-weight repairs. Next: #70 works on the 6 cases still over 0.01%,
none of which is a tie problem.

---

## 2026-11-23 — Roadmap #68: ask labels where the tied members disagree

**What ran.** `tied_labels.py` repeated #67's rebuilt committees and refit on seeds 250 to 299 (reproducing #67's
refit-coarse pick in all 50), then, while two or more members fit the most labels, ran up to 3 rounds of 8 labels
placed where the tied members' networks split most evenly in a fresh pool of 20,000 inputs, refitting and picking
again after each. 22.8 core-hours. Write-up: [TIED_LABELS.md](../experiments/locate-repair/TIED_LABELS.md).

**Measured.**
- 44 of 50 cases start tied; 41 are still tied after 3 rounds. 87 of 127 rounds found no pool input where the tied
  members disagree, and in 19 cases no round did.
- Under 0.01%: 42 to 44 (255 and 287 cross under, none over). Exact: 35 to 37 (250 and 255). Under 0.1%: 48 both.
  Closer in 16, further in 8, geometric-mean change 0.76.
- The 6 cases with 10 or more splitting inputs in round 1 all end closer; 255 goes from 0.062% with one right weight
  to the exact triple at 0.0022%.
- 28 of the 41 cases still tied tie exactly 2^k members, the winner's mirror variants for k output-layer edits;
  13 tie more.

**Interpretation.** Labels break the rare ties between different networks, but almost every tie is between members
that compute the same function on random inputs, so the totals barely move (measured on the pool; identical
everywhere is inferred). Most remaining ties are the known mirror symmetry, which "exact" already counts. Next:
#69 looks at the 13 cases that tie more members than mirrors explain.

---

## 2026-11-23 — Roadmap #67: choose among the members that fit every label

**What ran.** `tie_break_final.py` rebuilt #66's committees on the 1,096 final labels (seeds 250 to 299), kept every
member's fit, and picked the winner six ways: smallest total edit (the repair's rule), the committee's coarse order,
the smallest largest single edit, and each of those after refitting the near-best members from 5 more starts.
21.1 core-hours. Write-up: [TIE_BREAK_FINAL.md](../experiments/locate-repair/TIE_BREAK_FINAL.md).

**Measured.**
- Exact repairs: 30 by total edit, 34 by coarse order or largest edit, 35 after the refit with either; 37 are exact
  under at least one rule.
- Under 0.01%: 42 to 43 under every rule. Median 0.0028% to 0.0029%. Typical disagreement 3% to 8% lower than the
  repair's rule.
- The true triple at the original values fits all 1,096 labels in all 50 cases. Its own fit ties the top in 42 of
  the 46 committees that hold it, 44 after the refit (255 and 269 fixed; 259 and 277 still short).
- Median 4 members tied at the top; 2 cases have all 200 tied. Of the 9 cases where the true triple ties but
  refit-coarse picks another member, 8 end under 0.01%; 255 ends at 0.062% with one right weight.

**Interpretation.** The coarse order is the better tie-break for finding the exact weights, and the refit helps a
little more, but behavior on the probe barely changes, because most tied members are near-identical networks
(inferred from 50 cases). Next: #68 asks labels where the tied members disagree.

---

## 2026-11-23 — Roadmap #66: rebuild the committee from the final labels

**What ran.** `rebuild_committee.py` reran the pair screen and the triple committee on each case's 1,096 final
labels from #65 (seeds 250 to 299), picked the winner as the repair does, took its middle and scored it against
#65's region arm on the same 1M-point probe. 19.2 core-hours. Write-up:
[REBUILD_COMMITTEE.md](../experiments/locate-repair/REBUILD_COMMITTEE.md).

**Measured.**
- The true triple is in 46 of 50 rebuilt committees, against 35 before: it entered 12 and left 1 (258).
- Exact repairs stay at 30: 251, 263, 265 and 292 become exact; 255, 258, 259 and 269 stop.
- Of the 16 committees that hold the true triple without picking it, 12 have it tied with the winner on all
  1,096 labels; in 4 its fit reaches fewer labels.
- The committee's own first member (ties by the coarse fit) is exact in 34. Against the winner it is closer in
  12 cases and further in 12.
- Under 0.01%: 41 before, 42 after (263, 270, 279, 292 in; 255, 258, 259 out). Under 0.1%: 48 before, 46 after.
  Median 0.0026% before, 0.0029% after.

**Interpretation.** Screening on the final labels fixes most of the recall gap; what is left is the choice among
members that fit every label, where the repair's tie-break by fit cost does worse than the committee's own order
on exact repairs (inferred from 50 cases). Neither rule ends closer to the original on the probe. Next: #67
compares tie-breaks on saved member fits and refits the near-best members from more starts.

---

## 2026-11-22 — Roadmap #65: ask the next labels where the region of values is still unsure

Write-up: [REGION_LABELS.md](../experiments/locate-repair/REGION_LABELS.md). Evidence: `evidence/region-labels/`.

**Measured.**
- Added `--extra` to `early_values.py`: after the middle step, one more round of 8 labels from a fresh pool of
  5,000, picked where the committee splits most evenly or where 200 points of the middle step's walk split most
  evenly. Each arm refits every member on 1,096 labels, picks the winner again and takes its middle. Ran it on 50
  fresh width-16 three-fault cases, seeds 250 to 299: 33.3 core-hours. The final labels are on disk.
- Under 0.01%: repair 36, middle alone 37, committee-split labels 37 (41 with the middle), region-split labels 41
  (41 with the middle). Region-split labels then middle: closer than the repair in 42 of 50, none moved over
  either line, geometric-mean change 0.56 (committee arm 0.65, middle alone 0.83). Head to head, the region arm is
  closer in 32 cases and further in 9.
- Exact repairs stay at 30 of 50 in every arm. In the median case only 1 pool input splits the committee; the
  walk's networks split on 3.
- 9 cases stay over 0.01% after the region arm. 2 are exact; none of the other 7 committees holds the true triple.

**Interpretation.**
- The last 8 labels are better spent where the fitting values disagree than where the committee does. What is left
  is mostly a committee without the true triple, which labels on that committee cannot fix.

---

## 2026-11-21 — Roadmap #64: take the middle only when the labels leave room, on fresh cases

Write-up: [MIDDLE_STEP.md](../experiments/locate-repair/MIDDLE_STEP.md). Evidence: `evidence/middle-step/`.

**Measured.**
- Added `--middle` to `early_values.py`: after each stage's last round it runs #63's 1,600-step walk on the
  winner's values and records the middle in the row. Ran the two-step repair with it on 50 fresh width-16
  three-fault cases, seeds 200 to 249: 32.4 core-hours.
- The repair: 50 / 36 under 0.1% / 0.01%, 31 exact. The rule fixed beforehand (mean chord at least 0.005 spreads)
  uses the middle in 8 cases, 6 closer and 2 further: 50 / 37, with 224 crossing under 0.01%.
- The middle whenever it fits as many labels: closer in 32, further in 15, 50 / 39 (208, 224 and 235 cross under,
  none over). Median 0.0053% to 0.0045%, worst 0.096% to 0.088%.
- By mean chord, the geometric-mean change is 0.82, 0.99 and 0.81 from narrowest to widest (#63: 0.96, 0.92,
  0.73).

**Interpretation.**
- The middle is a small gain almost everywhere, so the width threshold does not help. The unconditional middle,
  #63's main arm, held on fresh cases.

---

## 2026-11-21 — Roadmap #63: refine the values once the repair has the right weights

Write-up: [REFINE_VALUES.md](../experiments/locate-repair/REFINE_VALUES.md). Evidence: `evidence/refine-values/`.

**Measured.**
- Added a `keep` hook to `switch_triples.run` and replayed the kept stage of all 58 cases in seeds 100 to 199
  whose repair edits exactly the three broken weights, to recover their 1,088 final labels. All 58 replays end at
  the saved disagreement.
- Compared the repair's values with the middle of the region of values that fit those labels (mean of a
  hit-and-run walk, three walks per case). Under 0.01%: 41 for the repair, 41 to 43 for the middle. The middle is
  closer in 30 to 37 cases and further in 20 to 26. One or two cases under the line move over it.
- Split by how far the region reaches: in the 14 cases with the widest regions the middle is closer in every walk
  in 11, with a geometric-mean change of 0.73; in the 29 narrowest it is 0.96.

**Interpretation.**
- The fit sits at an edge only where the labels leave room. There the middle helps; elsewhere the values are
  already pinned and the rest of the gap needs other labels (inferred, thresholds picked after seeing the data).

---

## 2026-11-20 — Roadmap #62: the third-stage rule on fresh cases

Write-up: [THIRD_RULE.md](../experiments/locate-repair/THIRD_RULE.md). Evidence: `evidence/third-stage-rule/`.

**Measured.**
- Added `--final-choice last-if-perfect` to `early_values.py` (default `last`) and ran the two-step repair on 50
  fresh width-16 three-fault cases, seeds 150 to 199: 29.6 core-hours. 36 cases stop after one stage, 7 after
  two, 7 reach the third.
- Keeping the last stage: 49 / 32 / 31 (under 0.1%, under 0.01%, exact), worst 0.136%. The new rule: 48 / 32 / 31.
  Of the five imperfect third stages it drops, one ends closer (157) and four further (160, 170, 186, 197); 197
  goes from 0.039% to 0.102%.
- The two perfect third-stage fits (159, 198) are exact and cross under 0.01%. All five exact third-stage wins in
  seeds 100 to 199 have a perfect fit.
- Pooled over seeds 100 to 199: last 99 / 67 / 58, the rule 98 / 67 / 58, the oracle 99 / 68 / 58.
- 17 of the 33 pooled cases at or over 0.01% edit exactly the three broken weights.

**Interpretation.**
- #61's rule was fitted to three cases. An imperfect third stage ends closer in 4 of 8 cases overall, so dropping
  it is no better than keeping it. `last` stays the default at width 16.
- Half the remaining 0.01% misses already have the right weights, so the next lever is the values (#63).

---

## 2026-11-20 — Roadmap #61: which stage's repair to keep

Write-up: [FINAL_STAGE.md](../experiments/locate-repair/FINAL_STAGE.md). Evidence: `evidence/final-stage-choice/`.

**Measured.**
- Rescored the saved per-stage results of three runs (#54 width 16, #60 width 32, #58 largest-edit width 16)
  under six final-choice rules that use only practice accuracy, plus a probe-based oracle. No new runs.
- Keeping the third stage only when it fits every practice label: width 16 stays 50 / 35 / 27 (under 0.1%, under
  0.01%, exact) and its worst case drops from 0.069% to 0.047%. Width 32 goes from 8 to 10 under 0.01%, the same
  lines as the oracle. On the largest-edit run, lines are unchanged; four cases end closer and one a hair further.
- It misses seed 210 (width 32) and seed 104 (largest-edit run): perfect third-stage practice fits on none of the
  true weights.
- Skipping the third stage saves 45% of the CPU at width 32 at no loss, but at width 16 it loses the three exact
  wins (102, 109, 129).

**Interpretation.**
- All six exact third-stage wins come with a perfect practice fit, so that is the signal to keep it. Two of eight
  perfect fits are still lookalikes.
- The rule was picked after seeing these runs, so this is in-sample. Fresh seeds come next (#62).

---

## 2026-11-20 — Roadmap #60: the two-step repair on a network twice as wide

Write-up: [WIDTH32.md](../experiments/locate-repair/WIDTH32.md). Evidence: `evidence/width-32/`.

**Measured.**
- Ran the two-step rule unchanged (`--width 32 --early 2 --two-step`) on 12 fresh three-fault cases at hidden
  width 32, seeds 200 to 211. The screen ranks a median 29,468 triples against 13,001 at width 16.
- All 12 end under 0.1% disagreement and 8 under 0.01% (67%, against 70% at width 16). 5 end on the exact three
  weights (42% against 54%), and the true triple is in the final committee in 6 (50% against 80%).
- 7 cases end at the first stage, 1 at the second and 4 at the third. The third stage makes all four worse than
  the second left them (seed 211: 0.0081% to 0.030%). At width 16 it ends three of its six exact.
- 2.06 CPU hours per case, 5.2 times width 16. The third stage is 45% of the CPU. About 25 core-hours in all.
- Stopping after the second stage at width 32 would give 12, 10 and 5 (under 0.1%, under 0.01%, exact) for about
  1.1 hours per case. At width 16 it would give 49, 33 and 24.

**Interpretation.**
- The rule carries over to a wider network: repairs behave like the original about as often, but land on the
  exact broken weights less often.
- At width 32 the third stage's flag fires on cases that are already close, and the wider pair screen then finds
  a better practice fit that behaves worse (inferred from four cases).
- Next: keep the second stage's repair unless the third stage fits every practice label (#61), scored from the
  saved per-stage results.

---

## 2026-11-20 — Roadmap #59: the largest-edit tie-break only in the widest rerun

Write-up: [LATE_EDIT.md](../experiments/locate-repair/LATE_EDIT.md). Evidence: `evidence/late-edit/`.

**Measured.**
- Ran the two-step rule with the coarse order for the first two stages and the largest-edit order only in the
  1,000-pair rerun (`--tie-break largest-edit --tie-break-from 2`). It covered #51's nine lost cases and the six
  unseen cases that reach the third stage, reusing their saved first two stages.
- #51's nine cases all finish at the second stage (1,000 triples), so they are identical to the coarse order by
  construction. #57's five (55, 63, 76, 93, 96) already end on the exact three weights there.
- Of the six that reach the last stage, four end closer and two further. Seed 102 crosses under 0.01%. None
  crosses 0.1%. Seed 124 loses its one true weight (0.012% to 0.068%).
- Over all 15 cases: 15 under 0.1% in both arms, 8 under 0.01% against 7, 10 exact in both, the same 11 with
  the true triple in the final committee. The last stage costs 4% more CPU. About 9 core-hours in all.

**Interpretation.**
- The coarse order stays the default and the tie-break line is closed. Early it hurts (#58); late it has
  almost nothing to work on.
- #57's wins came from a setting (1,000 pairs, 200 triples) the two-step rule does not use for those cases
  (inferred from the stage each case stops at).
- Every repair here is under 0.1%. The next question is whether the rule holds on a wider network (#60).

---

## 2026-11-19 — Roadmap #58: the largest-edit tie-break in the full repair

Full write-up: [experiments/locate-repair/LARGEST_EDIT.md](../experiments/locate-repair/LARGEST_EDIT.md).
Branch `claude/triple-largest-edit`, stacked on `claude/triple-tie-breaks`.

### Setup

- **Question.** Does #57's best tie-break (smallest largest single edit) help when it runs inside the whole
  two-step repair, not just on a finished committee?
- **Design.** `early_values.py --tie-break largest-edit` on the 50 unseen cases, seeds 100 to 149, compared
  case by case with #54's coarse-order run (`largest_edit_check.py`). 3 new tests.

### Results (measured)

1. 49 of 50 under 0.1% (coarse: 50), 34 under 0.01% (35), 28 exact (27). Seed 104 crosses the 0.1% line, at
   0.32% on no true weights.
2. The true triple is in the final committee in 35 cases against 40: lost in 6, gained in 1.
3. The loss starts at the first stage: 41 cases under 0.1% after it, against 46.
4. 24.3 CPU hours against 19.8; 9 cases need the widest rerun against 6.
5. Seeds 124, 126 and 149 stay lost. #51's nine lost cases were not rerun.

### Interpretation

- The coarse order stays the default; the new order is kept as an option.
- #57's offline ranking did not carry over, probably because the order also shapes the first committee and so
  every label chosen after it (inferred).

---

## 2026-11-19 — Roadmap #57: a better order among perfect practice fits

Full write-up: [experiments/locate-repair/TIE_BREAKS.md](../experiments/locate-repair/TIE_BREAKS.md).
Branch `claude/triple-tie-breaks`, stacked on `claude/triple-both-wide`.

### Setup

- **Question.** In the 12 cases where the rules lose the true triple among perfect practice fits, would a
  different tie-break than the coarse-fit rank keep it inside the 200-triple committee?
- **Design.** Rebuild the 12 committees at 1,000 pairs and 8 control committees at 200 pairs, record each fit's
  features (`tie_breaks.py`), and rank the true triple under seven tie-breaks (`tie_rank.py`). 2 new tests.

### Results (measured)

1. Smallest largest single edit keeps 5 of 12 (current order: 0). Largest smallest margin keeps 3; the other
   four keep 1 each.
2. No single tie-break keeps all 12; the best per case would keep 8. Seeds 52, 62, 68 and 124 are lost under all
   seven.
3. Sharing the 200 places between orders keeps fewer (2 to 4).
4. All 8 controls stay kept under every tie-break, so the controls cannot rank them.

---

## 2026-11-19 — Roadmap #56: a third rerun with 1,000 pairs and 1,000 triples

Full write-up: [experiments/locate-repair/BOTH_WIDE.md](../experiments/locate-repair/BOTH_WIDE.md).
Branch `claude/triple-both-wide`, stacked on `claude/triple-unseen-misses`.

### Setup

- **Question.** Does a rerun with both `--pairs 1000` and `--committee 1000` fix the three unseen cases the
  two-step rule leaves flagged, as #55 predicted for seed 149 only?
- **Code.** `--three-step` in `early_values.py`; `both_wide_check.py`; 2 new tests.

### Results (measured)

1. Seed 149 is fixed on the exact weights (0.045% to 0.0055%); 124 and 126 still lack the true triple.
2. Over all 50 unseen cases: 36 under 0.01% and 28 exact (two-step: 35 and 27), worst 0.050%, 22.2 CPU hours
   (19.8).
3. Seed 124 now fits every label but is worse (0.012% to 0.029%), so the end flag stops seeing it.
4. About 0.8 CPU hours per flagged case; fires on 3 of 50.

---

## 2026-11-19 — Roadmap #55: where the two-step rule still loses the true triple

Full write-up: [experiments/locate-repair/UNSEEN_MISSES.md](../experiments/locate-repair/UNSEEN_MISSES.md).
Branch `claude/triple-unseen-misses`, stacked on `claude/triple-two-step-unseen`.

### Setup

- **Question.** On seeds 124, 126 and 149, the three unseen cases whose repair still misses a label after both
  reruns, where does the true triple drop out?
- **Design.** Rebuild each committee from the practice labels with 200 and with 1,000 pairs (`screen_ranks.py`,
  new pair-count argument).
- **Code.** `unseen_misses.py`; 1 new test.

### Results (measured)

1. With 1,000 pairs all three true triples are extended (best true pair ranks 11, 281, 217).
2. All three are cut among perfect practice fits: ranks 1,441, 9,277 and 298 at 1,000 pairs.
3. Only 1,000 pairs with 1,000 triples together would keep one of them (149); the rule never combines the two.
4. Seed 124's true triple ranks lower with 1,000 pairs (1,441) than with 200 (1,177).

---

## 2026-11-19 — Roadmap #54: the two-step rule end to end on unseen cases

Full write-up: [experiments/locate-repair/UNSEEN.md](../experiments/locate-repair/UNSEEN.md).
Branch `claude/triple-two-step-unseen`, stacked on `claude/triple-two-step`.

### Setup

- **Question.** Does the two-step rule from #53, built into `early_values.py --two-step`, keep every case under
  0.1% on 50 three-fault cases it was not designed on (seeds 100 to 149)?
- **Design.** Four runs side by side; each case saves every step, so the plain run is scored from step 1.
- **Code.** `unseen_check.py`; 2 new tests.

### Results (measured)

1. All 50 under 0.1% (worst 0.069%), 35 under 0.01%, 27 exact, 40 true triple in the committee, 19.8 CPU
   hours (plain run: 46, 31, 20, 29 in 10.9).
2. The first flag fires on 15 cases and the second on 6; the 1,000-pair screen takes 7.4 CPU hours.
3. All four plain failures end between 0.0049% and 0.015%. Seed 133 is the one case made worse past 0.01%.
4. Three cases (124, 126, 149) still miss a label after both reruns; their true triple never reaches the
   committee. Seven of the 15 cases above 0.01% were never flagged.

---

## 2026-11-19 — Roadmap #53: the two-step rule on all 100 three-fault cases

Full write-up: [experiments/locate-repair/TWO_STEP.md](../experiments/locate-repair/TWO_STEP.md).
Branch `claude/triple-two-step`, stacked on `claude/triple-big-committee`.

### Setup

- **Question.** Does "200 triples; if the repair misses a label, 1,000 triples; if it still does, 1,000 pairs"
  put all 100 three-fault cases under 0.1%?
- **Design.** `--committee 1000` on the 11 end-flag cases of seeds 0 to 49; step 3 reuses #48's and #50's
  1,000-pair runs.
- **Code.** `two_step_check.py`; 1 new test.

### Results (measured)

1. All 100 under 0.1% (worst 0.088%), 75 under 0.01%, 69 exact, 79 every visible fault, 90 true triple in
   the committee, 42.9 CPU hours (plain run: 93, 63, 55, 64, 73 in 26.2).
2. On seeds 0 to 49 the larger committee alone fixes neither failure (14, 49: never extended) and breaks seed
   17 (0.086% to 0.12%); the second flag fires on all three plus 32 and 47.
3. The 1,000-pair screen runs on 10 of 100 cases.
4. The worst case (seed 70, 0.088%) is never flagged; 16 of the 25 cases above 0.01% are never flagged.

---

## 2026-11-19 — Roadmap #52: a larger committee instead of a wider screen

Full write-up: [experiments/locate-repair/BIG_COMMITTEE.md](../experiments/locate-repair/BIG_COMMITTEE.md).
Branch `claude/triple-big-committee`, stacked on `claude/triple-rank-misses`.

### Setup

- **Question.** #51 found the remaining true triples cut on a tie among perfect practice fits. Does keeping
  1,000 triples instead of 200 (still 200 pairs) fix them?
- **Design.** `early_values.py --early 2 --committee 1000 --only` the 16 fresh end-flag seeds.
- **Code.** `--committee` in `early_values.py`, `big_committee_check.py`; 2 new tests.

### Results (measured)

1. Fixes 4 of 5 fresh failures (63, 76, 96 exact; 83 on other weights); 49 of 50 under 0.1%, 33 exact, in
   18.8 CPU hours (wider screen: 47, 28, 26.5).
2. The true triple reaches the committee in exactly the 8 cases #51 predicted.
3. A rerun costs a median 1,103 s, against 2,808 s for the wider screen and 983 s for the plain run.
4. Seed 61, never extended from 200 pairs, gets worse (0.21%); the wider screen fixes it.
5. The end flag stays up in only 5 of 16 after the larger committee; rerunning those 5 with 1,000 pairs puts
   all 50 fresh cases under 0.1% (worst 0.088%), 35 exact, in 22.5 CPU hours.

---

## 2026-11-19 — Roadmap #51: where the three remaining failures lose the true triple

Full write-up: [experiments/locate-repair/RANK_MISSES.md](../experiments/locate-repair/RANK_MISSES.md).
Branch `claude/triple-rank-misses`, stacked on `claude/triple-end-flag`.

### Setup

- **Question.** Seeds 63, 76 and 96 stay above 0.1% at 1,000 pairs with no true triple in the committee. Is
  the true triple lost in the pair screen or in the cut to 200?
- **Design.** #47's `screen_ranks.py` on the nine fresh cases whose 1,000-pair committee lacks it (52, 55,
  62, 63, 66, 68, 76, 93, 96).
- **Code.** `rank_misses.py`; 1 new test.

### Results (measured)

1. All nine are lost in the cut: a true pair ranks in the best 70 of 3,321, the true triple is extended at
   200 pairs and fits every practice label, and ranks 204th to 4,839th among at least 200 perfect fits.
2. So more pairs cannot help them; a committee of 1,000 would hold the true triple in 8 of 9, including 63
   (253), 76 (840) and 96 (494).
3. With #47, 18 of 21 checked misses are lost in the cut and 3 never extended.

---

## 2026-11-19 — Roadmap #50: widen when the finished repair still misses a label

Full write-up: [experiments/locate-repair/END_FLAG.md](../experiments/locate-repair/END_FLAG.md).
Branch `claude/triple-end-flag`, stacked on `claude/triple-adaptive-widen`.

### Setup

- **Question.** #49's end flag (the finished 200-pair repair misses a label) caught all 7 failures in seeds 0
  to 99. Does rerunning the flagged cases with 1,000 pairs fix them?
- **Design.** `early_values.py --early 2 --pairs 1000 --only` the 16 flagged fresh seeds; seeds 0 to 49 reuse
  #48's 1,000-pair runs.
- **Code.** `end_flag_check.py`; 1 new test.

### Results (measured)

1. Two of the five fresh failures are fixed (61, 83, both exact once the true triple enters the committee).
   63 (0.23%), 76 (0.10%) and 96 (0.44%) stay above 0.1% with the true triple still missing at 1,000 pairs.
2. No flagged case crosses 0.1% upward; seed 6 goes from 0.003% to 0.011%.
3. All 100 cases: 97 under 0.1%, 66 under 0.01%, 59 exact, 68 every visible fault, 49.3 CPU hours (26.2 for
   200 pairs everywhere).
4. The flag stays up after the rerun in 13 of 16 fresh cases, including 4 that end under 0.011% with the true
   triple in the committee.
5. Nine of the 16 fresh 1,000-pair committees lack the true triple; six of those end under 0.1% anyway.

---

## 2026-11-18 — Roadmap #49: the adaptive widening flag on 50 fresh cases

Full write-up: [experiments/locate-repair/ADAPTIVE_WIDEN.md](../experiments/locate-repair/ADAPTIVE_WIDEN.md).
Branch `claude/triple-adaptive-widen`, stacked on `claude/triple-wide-screen`.

### Setup

- **Question.** #48's flag (no 200-pair committee member fits every practice label before targeted labels)
  picked out exactly the two failures on seeds 0 to 49. Does it hold on fresh cases?
- **Design.** `early_values.py --early 2 --widen-to 1000` on seeds 50 to 99.
- **Code.** `--widen-to` in `early_values.py`, `adaptive_check.py`; 2 new tests.

### Results (measured)

1. The flag never fired: in all 50 fresh cases some committee member fits every practice label.
2. 45 of 50 under 0.1%, 28 under 0.01%, 26 exact, 30 every visible fault, 13.9 CPU hours.
3. The five failures (61, 63, 76, 83, 96; worst 0.44%) all lack the true triple in the committee; the flag
   caught none of them.
4. The true triple is missing from 15 of 50 fresh committees (12 of 50 on seeds 0 to 49).
5. An end flag, "the finished 200-pair repair still misses a label", fires on 16 fresh cases and 11 of seeds 0
   to 49 and catches all 7 failures. Scored on seeds 0 to 49 from #48's runs: 50 of 50 under 0.1% in 22.8 CPU
   hours (200 pairs 12.3, 1,000 pairs 51.5).

### Interpretation

- *Inference:* fitting the practice labels says little about whether the true triple is in the committee;
  missing labels after the targeted rounds does. Roadmap #50 reruns the 16 flagged fresh cases with 1,000
  pairs.

---

## 2026-11-18 — Roadmap #48: the wider screen on all 50 three-fault cases

Full write-up: [experiments/locate-repair/WIDE_SCREEN.md](../experiments/locate-repair/WIDE_SCREEN.md).
Branch `claude/triple-wide-screen`, stacked on `claude/triple-screen-misses`.

### Setup

- **Question.** #47's wider screen (extend the best 1,000 pairs, not 200) fixed both clear failures on 8
  cases. Does it reach 50 of 50 under 0.1% on all 50 three-fault cases without new losses?
- **Design.** `early_values.py --early 2 --pairs 1000` on seeds 0 to 49, against the 200-pair runs of #45
  and #46. Also score an adaptive rule: widen only when no committee member fits every practice label before
  targeted labels.
- **Code.** `wide_compare.py`; 1 new test that checks the evidence.

### Results (measured)

1. 200 / 1,000 pairs: under 0.1% 48 / **50**, under 0.01% 35 / 37, exact 29 / 30, every visible fault
   34 / 35, true triple in committee 38 / 41. Worst case 0.61% / 0.079%.
2. Nothing lost at 0.1%. At 0.01%, 6 gained and 4 lost (6, 20, 22, 33; all still under 0.06%); 2 of the 4 end
   on the same weights as with 200 pairs.
3. Cases ending on the same weights with both screens still move by a factor of 0.09 to 3.7.
4. The committee gains the true triple in 14, 47 and 49 and loses it nowhere; the 9 still lacking it are the
   9 #47 found lost in the tie at the cut.
5. Cost: median 3,497 against 894 seconds per case; 51.5 against 12.3 CPU hours.
6. The adaptive flag fires on exactly seeds 14 and 49. Scored from the two runs, adaptive gets 50 of 50 under
   0.1%, 37 under 0.01%, 31 exact, 36 every visible fault, in 14.0 CPU hours.

### Interpretation

- *Inference:* the real effect is the two clear failures fixed; the shuffle at 0.01% is mostly different
  targeted labels, not the screen. The adaptive rule becomes the three-fault default, and roadmap #49 checks
  the flag on fresh cases.

---

## 2026-11-18 — Roadmap #47: where the committee screen loses the true triple

Full write-up: [experiments/locate-repair/SCREEN_MISSES.md](../experiments/locate-repair/SCREEN_MISSES.md).
Branch `claude/triple-screen-misses`, stacked on `claude/triple-fresh-switch`.

### Setup

- **Question.** The committee of 200 triples misses the true triple in 12 of 50 cases, including both clear
  failures (seeds 14 and 49). At which step of the screen does it drop out?
- **Design.** Rank diagnosis on the 12 cases; then extend the best 1,000 pairs instead of 200 on the 3 cases
  never extended (14, 47, 49) and 5 controls (0, 25, 26, 27, 28), with #45's 2-early-round rule.
- **Code.** `screen_ranks.py`; `--pairs` and `--only` on `early_values.py`; `pairs_kept` on the committee;
  1 new test.

### Results (measured)

1. In 9 of 12 the true triple is extended and fits every practice label, but ties with the 200th member and
   ranks 206th to 1,175th. In 3 (14, 47, 49) none of its pairs is in the best 200, so it is never extended.
2. In seeds 14 and 49 the true triple gets every practice label right and no extended triple does (best
   members 99.6% and 99.9%). Seed 14's best true pair ranks 204th; seed 49's 930th.
3. With 1,000 pairs, seed 14 goes from 0.61% to 0.0092% and seed 49 from 0.34% to 0.0069%, both on the exact
   broken weights. Seed 47 reaches 0.0037% on a lookalike (from 0.041%).
4. Controls keep their weight scores; seed 28 (lost in the tie) still lacks the true triple.
5. The wider screen costs about 3.5 times as long per case.

### Interpretation

- *Inference:* the joint pair ordering is the weak step for the clear failures (seed 49's best true pair is
  7th on the plain pair score). Losing the true triple in a tie mostly does no harm, because lookalikes fill
  the committee. Roadmap #48 runs the wider screen on all 50 cases.

---

## 2026-11-18 — Roadmap #46: two early rounds on fresh cases, and one early round

Full write-up: [experiments/locate-repair/FRESH_SWITCH.md](../experiments/locate-repair/FRESH_SWITCH.md).
Branch `claude/triple-fresh-switch`, stacked on `claude/triple-switch-point`.

### Setup

- **Question.** #45's 2-early-round lead over 4 was two or three cases on the 25 cases every three-fault rule
  was tuned on. Does it hold on fresh cases, and is 1 early round enough?
- **Design.** 0, 2 and 4 early rounds on seeds 25 to 49 (0 is #37's rule); 1 early round on seeds 0 to 24.
- **Code.** No new code; 1 new test that checks the evidence files.

### Results (measured)

1. Fresh cases, 0 / 2 / 4 early rounds: under 0.1% 23 / 24 / 24, under 0.01% 15 / 15 / 16, every visible
   fault 15 / 15 / 16, exact 13 in all three.
2. All 50 cases: under 0.1% 45 / 48 / 48 and under 0.01% 31 / 35 / 34 for 0 / 2 / 4 early rounds.
3. Two against four on fresh cases: 12 closer, 10 further, 3 level, median ratio 1.00.
4. One early round on seeds 0 to 24: 23 under 0.1% (loses seed 17 at 0.108%), 18 under 0.01%, weights as #37.
5. The true triple is missing from 8 of the 25 fresh committees (original cases: 4 of 25). Seven of those 8
   still end under 0.1% with 2 or 4 early rounds. Seed 49, the eighth, fails in every fresh arm (0.34% to 0.40%); the true triple
   would score better than every member on the practice labels before any targeted label.

### Interpretation

- *Inference:* #45's 2-against-4 lead was noise. What replicates is that a few early rounds of value labels
  move about 3 of 50 cases under 0.1% at no cost to the weights. Two rounds stays the default because it
  needs half the value queries.
- *Inference:* both clear failures left (seeds 14 and 49) are screening failures. Roadmap #47 widens the screen.

---

## 2026-11-17 — Roadmap #45: how many early rounds of value labels, for three broken weights

Full write-up: [experiments/locate-repair/SWITCH_POINT.md](../experiments/locate-repair/SWITCH_POINT.md).
Branch `claude/triple-switch-point`, stacked on `claude/triple-early`.

### Setup

- **Question.** #44 used 4 early rounds of #40's half mix, then committee splits only. Is 4 the right
  switch point?
- **Design.** #44's 25 cases and rule, with 2 early rounds (200 value queries) and 6 (600).
- **Code.** `early_values.py --early`, with 1 new test.

### Results (measured)

1. Two early rounds: 24 of 25 under 0.1%, 20 under 0.01% (best so far; #44: 18, #40: 19, #37: 16), median
   0.0041%, with 200 value queries.
2. Its weights are #37's exactly: 19 every visible fault, 16 exact, reached after 48 labels (#44: 64).
3. Six early rounds: 23 under 0.1% (loses seed 17), 16 under 0.01%, 18 every visible fault, 15 exact.
4. Every visible fault after 40 labels: 18, 16, 14 for 2, 4, 6 early rounds.
5. The true triple refitted on the 2-round labels: 25 under 0.1%, 22 under 0.01%.
6. Seed 14 still fails in every arm; its committee never holds the true triple.

### Interpretation

- *Inference:* a little value labelling early is enough; more delays the committee splits that find the last
  broken weight. The 2-against-4 lead is two or three cases, within one run's noise, so roadmap #46 checks it
  on fresh cases and tries 1 early round.

---

## 2026-11-17 — Roadmap #44: value labels early, committee splits late, for three broken weights

Full write-up: [experiments/locate-repair/EARLY_VALUES.md](../experiments/locate-repair/EARLY_VALUES.md).
Branch `claude/triple-early`, stacked on `claude/triple-half-line`.

### Setup

- **Question.** #40's half-and-half mix every round has the best behaviour but finds fewer visible faults;
  #43 showed late value queries change nothing. Does #40's mix in the first four rounds, then committee
  splits only, keep both gains?
- **Design.** #40's 25 three-fault cases; rounds 1 to 4 give 4 labels to value queries and 4 to the
  committee's most even splits, rounds 5 to 8 give all 8 to splits.
- **Code.** `experiments/locate-repair/early_values.py` with 2 tests.

### Results (measured)

1. It keeps both gains: 24 of 25 under 0.1% (#40: 24, #37: 22) and 19 every visible fault, 16 exact
   (#37: 19 and 16, #40: 17 and 15). The visible-fault cases are #37's exactly.
2. Under 0.01%: 18 (#37: 16, #40: 19). Median 0.0061%.
3. 400 of 1,600 labels are value queries, half of #40's 797.
4. Seeds 20 and 17, which #37 left over 0.1%, end at 0.0078% and 0.098%.
5. The true triple refitted on these labels has 25 under 0.1% and 19 under 0.01%.
6. Seed 14 still fails (0.56%): its committee never contained the true triple.

### Interpretation

- *Inference:* value labels pay off early, while the committee still disagrees widely, and committee
  splits find the last broken weight. The remaining failure is in the screen, not the labels.
  Roadmap #45 tries 2 and 6 early rounds to find the switch point.

---

## 2026-11-17 — Roadmap #43: half value labels from the behaviour line on, for three broken weights

Full write-up: [experiments/locate-repair/HALF_LINE.md](../experiments/locate-repair/HALF_LINE.md).
Branch `claude/triple-half-line`, stacked on `claude/triple-line`.

### Setup

- **Question.** #42 dropped every committee label at the 0.1% line and lost precision. Does #40's
  half-and-half mix, started at the same line, keep #37's weight-finding and add #40's behaviour gain?
- **Design.** #40's 25 three-fault cases; all 8 labels go to committee splits until fewer than 20 of 20,000
  pool inputs split it, then 4 value queries and 4 splits.
- **Code.** `experiments/locate-repair/half_line.py` with 2 tests.

### Results (measured)

1. Negative. 239 of 1,600 labels are value queries, and the result is #37's at both lines: 22 under 0.1% and
   16 under 0.01%, the same cases (#40: 24 and 19; #42: 22 and 13). Median 0.0074%.
2. Every visible fault 18 and exact 15 (seed 11 loses its true triple). Against #37, 10 closer, 8 further,
   7 unchanged.
3. The true triple refitted on these labels has 23 and 18, the same as on #37's labels.
4. Seed 20 never reaches the line and stays at 0.37%; seed 17 mixes only in the last round and ends at 0.116%.

### Interpretation

- *Inference:* #42's loss came from dropping all committee labels, but value queries asked that late do not
  help. #40's gains are in cases that reach the line late or never, so they come from early value labels.
  Roadmap #44 tries the reverse order: the mix early, committee splits late.

---

## 2026-11-17 — Roadmap #42: switching at the behaviour line for three broken weights

Full write-up: [experiments/locate-repair/SWITCH_LINE.md](../experiments/locate-repair/SWITCH_LINE.md).
Branch `claude/triple-line`, stacked on `claude/triple-switch`.

### Setup

- **Question.** #41's trigger (no pool input splits the committee) almost never fires for three faults. Does
  a trigger at the 0.1% line (fewer than 20 of 20,000 pool inputs split it) keep #40's behaviour gain
  without its visible-fault loss?
- **Design.** #40's 25 three-fault cases; all 8 labels go to committee splits until the line, then all 8 to
  value queries. The splitting count is recorded every round.
- **Code.** `experiments/locate-repair/switch_line.py` with 2 tests.

### Results (measured)

1. Negative. The trigger fires in 23 of 25 cases and 472 of 1,600 labels are value queries, but under 0.1%
   stays at 22 and under 0.01% falls from 16 to 13 (#37: 22 and 16; #40: 24 and 19). Median 0.0095%.
2. Every visible fault 19 and exact 16, as in #37. Against #37, 9 cases closer, 11 further, 5 unchanged.
3. The true triple refitted on this arm's labels has 13 under 0.01%, against 18 on #37's and 20 on #40's.
4. Seed 20 never reaches the line; seed 17 switches only for the last round and ends at 0.142%.

### Interpretation

- *Inference:* the last few splitting inputs still carry value information, so dropping every committee
  label at the line costs more than value queries give back. #40's fixed half-and-half mix is still the best
  behaviour rule; roadmap #43 asks whether starting it at the line keeps the visible faults.

---

## 2026-11-17 — Roadmap #41: the switch rule for three broken weights

Full write-up: [experiments/locate-repair/SWITCH_TRIPLES.md](../experiments/locate-repair/SWITCH_TRIPLES.md).
Branch `claude/triple-switch`, stacked on `claude/triple-values`.

### Setup

- **Question.** Does #39's switch rule (committee splits while any exist, value queries after) keep #40's
  behaviour gain for three faults without its visible-fault loss?
- **Design.** #40's 25 three-fault cases, all 8 labels per round chosen by the switch rule, plus the true
  triple fitted two ways as a reference.
- **Code.** `experiments/locate-repair/switch_triples.py` with 2 tests.

### Results (measured)

1. Negative: the committee keeps splitting. 16 of 25 cases never switch; only 101 of 1,600 labels are value queries.
2. Final counts are #37's exactly: 22 under 0.1%, 16 under 0.01%, 19 every visible fault, 16 exact.
3. Against #37, 4 cases end closer, 1 further, 20 unchanged. Seeds 17 and 20 never switch and stay above 0.1%.
4. Both told fits end at 23 and 18; seed 2's told fit did not stick this time.

### Interpretation

- *Inference:* three-fault committees keep lookalikes that still differ on a few pool inputs, so "no split
  left" is too strict a trigger. A trigger at the 0.1% behaviour line is roadmap #42.

---

## 2026-11-17 — Roadmap #40: value queries for three broken weights

Full write-up: [experiments/locate-repair/TRIPLE_VALUES.md](../experiments/locate-repair/TRIPLE_VALUES.md).
Branch `claude/triple-values`, stacked on `claude/switch-queries`.

### Setup

- **Question.** Do #38's value queries help three-fault repairs, which already use a continuous refit?
- **Design.** #37's 25 three-fault cases. Each round, 4 of 8 labels go to inputs where the winning triple and a
  copy with one value nudged by 0.01 spreads disagree; the other 4 to the committee's most even splits.
- **Code.** `experiments/locate-repair/triple_values.py` with 1 test.

### Results (measured)

1. 24 of 25 repairs under 0.1% and 19 under 0.01% (#37: 22 and 16); median 0.0044% (#37: 0.0056%).
2. Every visible fault drops from 19 to 17, and exact entries from 16 to 15.
3. Closer to the original in 16 cases, further in 9. Seeds 17 and 20 move under 0.1%, with their true-triple fit.
4. Seed 14 still fails at 0.87%. In seed 2 the true-triple refit stays stuck at about 2% while the winner ends at 0.0015%.

### Interpretation

- *Inference:* the two-fault trade repeats: committee splits find the weights, value queries set them.

---

## 2026-11-17 — Roadmap #39: switching to value queries once the committee agrees

Full write-up: [experiments/locate-repair/SWITCH_QUERIES.md](../experiments/locate-repair/SWITCH_QUERIES.md).
Branch `claude/switch-queries`, stacked on `claude/value-queries`.

### Setup

- **Question.** Can a rule that spends targeted labels while the committee disagrees, and value queries
  after, keep #38's behaviour gains without its one-case loss in weight-finding?
- **Design.** #38's 25 cases and refined fit. `switch`: all 8 labels targeted while any pool input splits
  the committee, else all 8 value queries. `fill`: targeted only on splitting inputs, value queries for the rest.
- **Code.** `experiments/locate-repair/switch_queries.py` with 2 tests.

### Results (measured)

1. `switch`: 23 of 25 every visible fault (as with no value queries), 24 under 0.1% and 19 under 0.01%
   (#38's best arm: 22, 24, 20), with 498 value queries against 745.
2. It reaches the strict line later: 8 of 25 under 0.01% after 24 labels, against 17 for #38's arm.
3. Case by case it is not closer than #38's arm (lower in 7, higher in 13). `fill` matches #38's arm.
4. The committee often re-splits after a switch (7 of 25 cases); seed 23 still fails.

### Interpretation

- *Inference:* #38's one-case weight-finding loss is noise-sized; a switch recovers it at the cost of a
  slower approach to 0.01%. A fixed half suits small label budgets.

---

## 2026-11-17 — Roadmap #38: queries that pin down the winning repair's values

Full write-up: [experiments/locate-repair/VALUE_QUERIES.md](../experiments/locate-repair/VALUE_QUERIES.md).
Branch `claude/value-queries`, stacked on `claude/behaviour-scores`.

### Setup

- **Question.** Can labels aimed at the winner's values close #37's 0.13% to 0.17% right-weight misses?
- **Design.** #37's 25 random-size two-fault cases, four arms: value queries (half of each round's labels
  go to inputs where the winner and a copy with one value nudged 0.01 spreads disagree), middle-of-tie
  grid fit, a refined fit (61 x 61 grid within 1.5 grid steps of the grid winner), and refined fit plus
  value queries. Every round's winner is probed on a million inputs.
- **Code.** `experiments/locate-repair/value_queries.py` with 4 tests.

### Results (measured)

1. Refined fit plus value queries: 24 of 25 under 0.1% and 20 under 0.01% after 64 labels, against #37's
   20 and 9. Median disagreement 0.0016% against 0.015%. Lower than #37 in 24 cases, higher in none.
2. Alone, neither change does much: value queries 21 and 10; refined fit 22 and 13.
3. Middle-of-tie on the coarse grid is worse: 4 of 25 under 0.1% before any queries (11 at the edge), 20 at the end.
4. #37's four right-weight misses (seeds 5, 8, 12, 20) end at 0.029% or less. Seed 23 still picks a wrong
   weight in every arm (0.10% to 0.16%).

### Interpretation

- *Inference:* #37's leftover two-fault error was the fit's 0.025-spread resolution, not the labels. Value
  queries only help once the fit can move by less than a grid step.
- *Inference:* splitting labels costs a little weight-finding (22 against 23 every visible fault); switching
  to value queries only once the committee agrees might keep both. Not tested.

---

## 2026-11-17 — Roadmap #37: rescoring two- and three-fault repairs by behaviour

Full write-up: [experiments/locate-repair/BEHAVIOUR_SCORES.md](../experiments/locate-repair/BEHAVIOUR_SCORES.md).
Branch `claude/behaviour-scores`, stacked on `claude/lookalikes`.

### Setup

- **Question.** Does the behaviour score (#36) change the two-fault results too, and how many targeted
  labels does a behavioural repair need?
- **Design.** #35's two-fault runs (random sizes, plus a 1.6-spread control) and #34's three-fault
  runs, with the winner probed on a million inputs after every round. A side check fits only the
  true pair to see where the leftover disagreement comes from.
- **Code.** `experiments/locate-repair/behaviour_scores.py` with 2 tests, `value_check.py`, and a
  `probe` argument on the pair and triple query loops.

### Results (measured)

1. Three faults: 16 of 25 repairs behave like the original (under 0.1%) after 16 labels, against 9
   with every visible fault; 22 against 19 after 64.
2. Two faults: the behaviour score is stricter. 20 of 25 behave like the original after 64 labels,
   against 23 with every visible fault. Four winners (seeds 5, 8, 12, 20) edit both exact weights and
   disagree on 0.13% to 0.17%; one wrong-weight winner (seed 11) is at 0.057%.
3. A fit on only the true pair lands within 0.03 spreads of the original values in seeds 5, 8 and 12
   and still disagrees on 0.12% to 0.17%; in seeds 13 and 15 it lands within 0.008 and disagrees on
   0.002% or less.
4. The 1.6 control: 23 and 23, with two misses of each kind.

### Interpretation

- *Inference:* the entry score misses lookalikes, and the behaviour score catches right weights set
  slightly wrong. 1,024 practice labels resolve about 0.1%, which is where the two-fault misses sit,
  and targeted labels split the committee over weights, not values. Report both scores.

---

## 2026-11-17 — Roadmap #36: are the last lookalikes wrong at all?

Full write-up: [experiments/locate-repair/LOOKALIKES.md](../experiments/locate-repair/LOOKALIKES.md).
Branch `claude/lookalikes`, stacked on `claude/boundary-queries`.

### Setup

- **Question.** Do the repairs that edit the wrong entries also behave differently from the
  original network?
- **Design.** #34's committee once per case; 64 targeted labels with uniform, boundary and mixed
  pools; each tied winner probed on a million fresh inputs against the teacher.
- **Code.** `experiments/locate-repair/lookalikes.py` with 2 tests.

### Results (measured)

1. With every pool, the winner disagrees with the teacher on under 0.1% of inputs in 22 of 25 cases,
   while it edits all three broken entries in only 16 to 17.
2. 6 of the 9 wrong-entry winners (uniform pool) disagree on 0.009% or less, as close as a fit on
   the true entries. The other three: seed 14 at 0.87% (no true triple in the committee), seeds 17
   and 20 at 0.12% and 0.37%, matched by the fit on the told entries.
3. The mixed pool gives 16, 19 and 22, the same as uniform.

### Interpretation

- *Inference:* scoring a repair by its entries undercounts. Different settings of a small ReLU
  network compute nearly the same function; three-fault repair restores behaviour in 22 of 25.

---

## 2026-11-17 — Roadmap #35: querying near the suspects' decision boundaries

Full write-up: [experiments/locate-repair/BOUNDARY_QUERIES.md](../experiments/locate-repair/BOUNDARY_QUERIES.md).
Branch `claude/boundary-queries`, stacked on `claude/triple-committee`.

### Setup

- **Question.** Does drawing #34's query pool near the tied members' decision boundaries break the
  last ties?
- **Design.** #34 unchanged except the pool: of 200,000 uniform inputs, the 20,000 closest to a
  boundary of any tied member.
- **Code.** `triple_committee.py --pool boundary`, with a third test.

### Results (measured)

1. After 64 examples: all three found 17 of 25 (uniform 16), every visible fault 19 (19), pass 24
   (24). The boundary pool trails at every earlier round.
2. Late-round query splits stay near even (median 0.31 against 0.10), but the true triple's fit gets
   every label right in 16 of 21 cases against 18.
3. Fixes seeds 4 and 20, loses 2 and 21; seeds 10 and 15 keep the same ties.

### Interpretation

- *Inference:* where to ask is not the limit. The surviving ties may be lookalikes that agree with
  the teacher almost everywhere (roadmap #36).

---

## 2026-11-16 — Roadmap #34: targeted examples on a 200-triple committee

Full write-up: [experiments/locate-repair/TRIPLE_COMMITTEE.md](../experiments/locate-repair/TRIPLE_COMMITTEE.md).
Branch `claude/triple-committee`, stacked on `claude/continuous-triples`.

### Setup

- **Question.** Can targeted examples pick the true triple out of the ~160 that fit practice
  perfectly after #33's continuous fit?
- **Design.** #31's random-size cases at H = 16. Committee: #33's top 200. Eight rounds of 8
  targeted examples; each member re-fitted every round from its last and current values.
- **Code.** `experiments/locate-repair/triple_committee.py` with 2 tests.

### Results (measured)

1. After 64 examples: all three found 16 of 25, every visible fault 19, pass 24 (practice only: 3,
   3, 15). The median number of members tied at the best accuracy falls from 159 to 3.
2. The committee held the true triple in 21 cases; 16 of those ended with it, 3 ended on a tied
   lookalike, 1 on a fit that missed one label.

### Interpretation

- *Inference:* with a working fit, three faults behave like two. 19 of 25 (three faults) and 23 of
  25 (two faults, #32) are the honest numbers on bugs of any size.

---

## 2026-11-16 — Roadmap #33: a continuous fit for every three-fault suspect

Full write-up: [experiments/locate-repair/CONTINUOUS_TRIPLES.md](../experiments/locate-repair/CONTINUOUS_TRIPLES.md).
Branch `claude/continuous-triples`, stacked on `claude/offset-screen`.

### Setup

- **Question.** With the coarse grid gone as a filter, does the true triple reach the suspect list?
- **Design.** #31's random-size cases at H = 16. Every extended triple gets #29's margin fit from its
  current values and from its coarse-7 optimum; triples are ranked by practice accuracy.
- **Code.** `experiments/locate-repair/continuous_triples.py` with 2 tests.

### Results (measured)

1. The true triple fits practice perfectly in 23 of the 24 cases that contain it.
2. A median of 160 triples per case fit perfectly. Top-40 recall is 9 (ties by smallest change)
   or 13 (ties by coarse-7 order); top-200 recall is 16 or 21.

### Interpretation

- *Inference:* the fit is solved; the practice set now underdetermines the triple, as #21 found for
  pairs. Roadmap #34 puts targeted examples on a 200-triple committee.

---

## 2026-11-16 — Roadmap #32: denser screens for bugs of any size

Full write-up: [experiments/locate-repair/DENSE_SCREEN.md](../experiments/locate-repair/DENSE_SCREEN.md).
Branch `claude/offset-screen`, stacked on `claude/random-sizes`.

### Setup

- **Question.** Does a denser grid make #26's pair screen and #27's triple ranking independent of
  the bug size?
- **Design.** #31's random-size cases. Pairs: a 25 × 25 joint screen. Triples: 25-point pairs, then
  coarse 7³, coarse 11³, and the coarse-11 top 1,000 refined.
- **Code.** `experiments/locate-repair/dense_screen.py` (and `random_sizes.py --points`), 2 tests.

### Results (measured)

1. Pairs after 64 targeted examples, every visible fault found: 23 of 25 (13 points: 20), at about
   45% more time per case.
2. Triples in the top 40: coarse 11 8 (coarse 7: 5), refined 14 (#31: 11). In 8 cases the true
   triple ranks below 1,000 on the 11-point grid.

### Interpretation

- *Inference:* 23 of 25 is the honest two-fault number. For three faults the coarse grid is the
  bottleneck; roadmap #33 replaces it with a continuous fit of every extended triple.

---

## 2026-11-16 — Roadmap #31: the screens were tuned to the bug size

Full write-up: [experiments/locate-repair/RANDOM_SIZES.md](../experiments/locate-repair/RANDOM_SIZES.md).
Branch `claude/random-sizes`, stacked on `claude/triple-recall`.

### Setup

- **Question.** Do #26's pair results and #28–#30a's triple rankings hold when the faults are not
  exactly 1.6 spreads, the size that the coarse grids happen to contain?
- **Design.** The same cases with each fault's size drawn between 1.0 and 2.2 spreads: #26's pair
  pipeline at H = 32 and #30a's triple rankings at H = 16. Control for pairs: size 1.6 with a
  14-point joint screen, whose points miss 1.6.
- **Code.** `experiments/locate-repair/random_sizes.py` with 3 tests.

### Results (measured)

1. Pairs after 64 targeted examples, every visible fault found: 25 (#26), 21 (control, grid moved
   only), 20 (random sizes). Pass: 25, 25, 23.
2. Triples, true triple in the top 40: coarse 7 falls from 17 to 5; coarse 9, which never matched,
   goes from 8 to 7; refined from 19 to 11.

### Interpretation

- *Inference:* #26's 25 of 25 and the triple suspect lists behind #28 and #29 were flattered by a
  grid point next to the corruption. Targeted examples and the pass-without-finding results
  still hold. Roadmap #32 makes the screen independent of grid placement.

---

## 2026-11-16 — Roadmap #30a: ranking triples, and a grid that matches the bug

Full write-up: [experiments/locate-repair/TRIPLE_RECALL.md](../experiments/locate-repair/TRIPLE_RECALL.md).
Branch `claude/triple-recall`, stacked on `claude/warm-triples`.

### Setup

- **Question.** How should #27's extended triples be ranked so the true triple reaches the
  suspect list more often than #28's 17 of 25?
- **Design.** #27's 25 cases at H = 16. The true triple's position under a 7³ grid (#28's), a 9³
  grid, 7³ grids over ±2.0 and ±3.0 spreads, and the coarse-7 top 1,000 refined with #29's
  margin fit.
- **Code.** `experiments/locate-repair/triple_recall.py` with 3 tests.

### Results (measured)

1. Top-40 recall: coarse 7 17, coarse 9 8, span ±2.0 4, span ±3.0 3, refined 19 (with coarse-7
   order breaking ties; 14 with smallest change).
2. The 7-point grid over ±2.5 has a point at ±1.67 spreads; the corruption is exactly ±1.6. Grids
   whose nearest point is 0.27–0.4 spreads away lose most of the recall.

### Interpretation

- *Inference:* the coarse screen in #27–#29 is partly tuned to the corruption size by accident,
  and #26's 13-point pair screen shares the same grid point. Roadmap #31 tests the arc with
  other corruption sizes before building more on it.

---

## 2026-11-16 — Roadmap #29: a fitter that keeps the true triple perfect

Full write-up: [experiments/locate-repair/WARM_TRIPLES.md](../experiments/locate-repair/WARM_TRIPLES.md).
Branch `claude/warm-triples`, stacked on `claude/triple-queries`.

### Setup

- **Question.** If each triple's fit starts from its previous round's values and is refined
  continuously, does #28's true triple keep a perfect fit, and do more cases get found?
- **Design.** #28 unchanged except the fit: best of #27's fine grid fit, a warm-started local fit,
  and Nelder–Mead on a hinge loss of the output margin. The told triple is fitted every round too.
- **Code.** `experiments/locate-repair/warm_triples.py` with 3 tests.

### Results (measured)

1. After 64 examples the true triple fits every label in 16 of the 17 cases where it is a
   candidate (#28: in 15 of 17 no candidate did). The told fit fits every label in 22 of 25.
2. All three found: 12 of 25 (#28: 10). Visible faults found: 15 (#28: 13). Pass: 23 (#28: 22).
3. Remaining misses: 8 cases without the true triple among the candidates, and 3 real ties
   (seeds 4, 11, 20) where both triples fit all 1,088 labels.

### Interpretation

- *Inference:* the fit is no longer the main limit. Candidate recall is, as it was for pairs
  before #26's joint screen. Next: roadmap #30.

---

## 2026-11-16 — Roadmap #28: targeted examples for three broken weights

Full write-up: [experiments/locate-repair/TRIPLE_QUERIES.md](../experiments/locate-repair/TRIPLE_QUERIES.md).
Branch `claude/triple-queries`, stacked on `claude/three-faults`.

### Setup

- **Question.** Do #22's targeted examples settle #27's three-fault ties the way they settled
  two-fault ties?
- **Design.** #27's 25 cases at H = 16; #27's coarse top 40 triples as candidates; eight rounds
  of 8 targeted examples, every triple re-fitted finely each round. Seeds 11, 15 and 20 rerun
  with the true triple's fit recorded.
- **Code.** `experiments/locate-repair/triple_queries.py` with 2 tests.

### Results (measured)

1. All three faults found: 5 of 25 with no examples, 10 after 64. Visible faults: 8 to 13.
   Pass: 18 to 22.
2. The true triple is among the 40 candidates in 17 cases.
3. After 64 examples the winner fits every example in 2 of 25 cases (19 before). In 15 of the
   17 cases with the true triple available, no candidate fits all the teacher's labels, though
   the teacher's values would.
4. In the reruns the true triple's own fit ends at 97.2–99.4%; in seeds 11 and 15 it won a
   round and later lost.

### Interpretation

- *Inference:* the grid fit, not the data, now limits three-fault localization. The examples
  sit where candidates disagree, where a coarse fit of three values is least precise. Next:
  warm-started local fits (roadmap #29).

---

## 2026-11-16 — Roadmap #27: three broken weights

Full write-up: [experiments/locate-repair/THREE_FAULTS.md](../experiments/locate-repair/THREE_FAULTS.md).
Branch `claude/three-faults`, stacked on `claude/joint-screen`.

### Setup

- **Question.** With three faults, does #26's joint pair screen still rank a two-fault pair
  near the top, and does extending its best pairs by a third entry find all three?
- **Design.** H = 16, 25 cases, three faults in three different units. Joint pair screen; the
  best 200 pairs extended by every other entry; coarse 7³ fits, the best 20 re-fitted finely;
  practice data only, no queries.
- **Code.** `experiments/locate-repair/three_faults.py` with 3 tests.

### Results (measured)

1. The first two-fault pair is a median 21st of 3,321; the true triple is among the extended
   triples in 24 of 25 cases, and in the coarse top 20 in 11.
2. The winner holds all three faults in 5 of 25 cases, yet the repair passes 17 (told: 20)
   and undoes a median 98.5% of the damage.

### Interpretation

- *Inference:* three faults bring back #21's look-alikes at the triple level. Targeted queries
  on the top triples are the next step.

---

## 2026-11-16 — Roadmap #26: a joint screen finds every visible fault

Full write-up: [experiments/locate-repair/JOINT_SCREEN.md](../experiments/locate-repair/JOINT_SCREEN.md).
Branch `claude/joint-screen`, stacked on `claude/candidate-recall`.

### Setup

- **Question.** #25's four misses were pairs the coverage screen ranked past 3,000. Does a
  small joint fit of every pair rank them higher?
- **Design.** H = 32, 25 cases. Each of the 13,041 pairs gets a 13 × 13 joint grid on the
  practice set, ranked by accuracy with coverage breaking ties. The best 40 then get #22's
  targeted queries.
- **Code.** `experiments/locate-repair/joint_screen.py` with 3 tests.

### Results (measured)

1. The broken pair's median position falls from 327th (coverage) to 5th; it is in the joint
   top 40 in all 21 cases without a silent fault, including all four of #25's misses.
2. After 48 targeted queries every visible fault is found in 25 of 25 cases, the broken pair
   wins in all 21 non-silent cases, and the benchmark passes 25 of 25.
3. The screen takes about 54 seconds per case for all pairs, cheaper than #25's pipeline.

### Interpretation

- *Inference:* coverage misses interacting faults. A crude joint fit, ordered by accuracy and
  not by smallest change, sees them. The two-fault localization problem is solved on these
  cases at H = 32.

---

## 2026-11-16 — Roadmap #25: getting the broken pair onto the suspect list

Full write-up: [experiments/locate-repair/CANDIDATE_RECALL.md](../experiments/locate-repair/CANDIDATE_RECALL.md).
Branch `claude/candidate-recall`, stacked on `claude/active-screen`.

### Setup

- **Question.** #24's targeted queries find the broken pair whenever it is a candidate. Which
  candidate rule keeps it most often, and how far does that take localization at H = 32?
- **Design.** Coarse fits on the coverage screen's top 3,000 pairs; candidate rules compared
  offline; then #22's targeted queries on 100 candidates per case (fit 3,000, smallest change).
  25 cases. Faults that change no practice label alone are marked silent.
- **Code.** `experiments/locate-repair/candidate_recall.py` with 5 tests.

### Results (measured)

1. Keeping 100 candidates from the top 3,000 raises recall from 13 to 18 of 25; keeping 200
   adds nothing.
2. After 64 targeted queries the broken pair wins in all 18 of those cases (#24: 12 of 25
   overall), and the benchmark passes 24 of 25.
3. Four cases have a silent fault (no practice label changes); the winner finds the visible
   fault in all four. Every visible fault is found in 21 of 25 cases. The 4 misses ranked past
   3,000 in the screen.

### Interpretation

- *Inference:* with targeted labels, locating two visible faults at H = 32 now fails only where
  the screen ranks them too low. A localization score should ignore silent faults.

---

## 2026-11-16 — Roadmap #24: targeted examples on a wider network

Full write-up: [experiments/locate-repair/ACTIVE_SCREEN.md](../experiments/locate-repair/ACTIVE_SCREEN.md).
Branch `claude/active-screen`, stacked on `claude/pair-screen`.

### Setup

- **Question.** Do #22's targeted queries still single out the broken pair at H = 32, where
  candidates come from #23's coverage screen instead of an all-pairs search?
- **Design.** H = 32, 25 cases; coverage screen, coarse fits on the top 1,000, best 40 as
  candidates; eight rounds of 8 targeted or 8 random queries from a 20,000-input pool.
- **Code.** `experiments/locate-repair/active_screen.py` with 2 tests.

### Results (measured)

1. After 64 targeted queries the broken pair wins in 12 of 25 cases, up from 5; 64 random
   queries reach 6. Targeted passes 22 of 25, random 19.
2. The broken pair was among the 40 candidates in 13 cases, and won in 12 of them.
   Of the 12 cases without it, 6 lost it at the screen and 6 at the coarse fit.
3. In all 12 found cases the remaining ties are exactly 2 to the power of the number of
   output-layer faults.

### Interpretation

- *Inference:* #22's result holds at H = 32. Locating two faults is now limited by candidate
  recall, not by choosing among candidates.

---

## 2026-11-16 — Roadmap #23: a cheap screen for pairs of broken weights

Full write-up: [experiments/locate-repair/PAIR_SCREEN.md](../experiments/locate-repair/PAIR_SCREEN.md).
Branch `claude/pair-screen`, stacked on `claude/active-queries`.

### Setup

- **Question.** Can a cheap screen replace #21's all-pairs search, so that pair search
  reaches H = 32 (13,041 pairs)?
- **Design.** Two screens score every pair: a Newton step on a logistic margin loss, and
  coverage (the share of wrong practice examples the two entries' single edits correct
  together). The coverage top 200 or 1,000 pairs then go through #21's joint fits.
  H = 8 and 32, 50 cases each.
- **Code.** `experiments/locate-repair/pair_screen.py` with 3 tests.

### Results (measured)

1. Broken pair's median rank at H = 32: 338 under coverage against 2,952 under Newton.
   The coverage top 1,000 contains it in 72% of cases.
2. At H = 8, screen-then-fit finds the pair in 28 of 50 cases, matching #21's exhaustive rate.
3. At H = 32, screen-then-fit passes 72% and finds the pair in 10% (#20's greedy repair: 56%
   and 4%). Of 36 screened-in pairs, 16 were lost at the coarse fit; of the 20 left, 5 won.

### Interpretation

- *Inference:* at H = 32 the screen is no longer the bottleneck. Many pairs fit practice
  perfectly, as in #21, so #22's targeted queries are the next step.

---

## 2026-11-16 — Roadmap #22: asking where the suspects disagree

Full write-up: [experiments/locate-repair/ACTIVE_QUERIES.md](../experiments/locate-repair/ACTIVE_QUERIES.md).
Branch `claude/active-queries`, stacked on `claude/pair-search`.

### Setup

- **Question.** Can a few teacher-labelled examples, chosen where #21's candidate pairs
  disagree, single out the two broken entries?
- **Design.** H = 8, 25 cases; #21's 20 candidate pairs per case; eight rounds of 8 targeted
  (most-disagreed) or 8 random queries, re-fitting every candidate each round.
- **Code.** `experiments/locate-repair/active_queries.py` with 2 tests.

### Results (measured)

1. After 64 targeted queries the broken pair wins in 20 of 25 cases, up from 14; 64 random
   queries leave it at 14. Every case passes the benchmark.
2. In 4 of the 5 misses the broken pair was not among the 20 candidates.
3. The ties left after 64 queries are exactly the mirror symmetry: 2 to the power of the
   number of output-layer faults.

### Interpretation

- *Inference:* #21's ambiguity comes from sampling, not from the network, apart from the
  mirror symmetry. A benchmark that can query the original model could score localization
  as well as repair.

---

## 2026-11-16 — Roadmap #21: searching pairs of weights for two faults

Full write-up: [experiments/locate-repair/PAIR_SEARCH.md](../experiments/locate-repair/PAIR_SEARCH.md).
Branch `claude/pair-search`, stacked on `claude/two-faults`.

### Setup

- **Question.** Can ranking pairs of entries, instead of one at a time, find #20's two
  broken entries from practice data?
- **Design.** #20's cases. Joint pair fits over a shortlist (top 8 or 16 entries by
  single-edit score) at H = 8 and 32, 50 cases each, and over all 861 pairs at H = 8 on
  25 cases; the 20 best pairs re-fitted finely.
- **Code.** `experiments/locate-repair/pair_search.py` with 2 tests.

### Results (measured)

1. Single-edit scores barely rank the faults: median rank 39th of 162 at H = 32, both in
   the top 16 in 1 of 50 cases. Shortlist pair search found both faults in 0 of 50 there.
2. All pairs at H = 8 pass 23 of 25 cases (told: 96% in #20) but pick the broken pair in 14.
3. In 9 of the 11 misses, the broken pair fit practice no better than the winner, which
   needed a smaller change; in 2, a wrong pair fit strictly better.

### Interpretation

- *Inference:* 1,024 examples leave a two-fault repair underdetermined, and the
  smallest-change rule prefers look-alike pairs. Finding the pair needs targeted data or a
  different prior; passing the rule needs neither.

---

## 2026-11-16 — Roadmap #20: two broken weights

Full write-up: [experiments/locate-repair/TWO_FAULTS.md](../experiments/locate-repair/TWO_FAULTS.md).
Branch `claude/two-faults`, stacked on `claude/combined`.

### Setup

- **Question.** With two corrupted entries in different hidden units, does the benchmark
  still pass repairs that miss the faults, and is the gap finding the entries or fitting them?
- **Design.** #7's cases plus a second corrupted entry; H = 8 and 32, 50 seeds each; 1,024
  practice examples. Repairs: one min-change edit, two greedy edits, a joint 200 × 200 re-fit
  of the chosen pair, and the same with the true entries given.
- **Code.** `experiments/locate-repair/two_faults.py` with 2 tests.

### Results (measured)

1. One edit passes 10% of cases at both widths (one fault: 85–92% in #10).
2. Two greedy edits pass 30% (H = 8) and 56% (H = 32). At H = 32, 21 of the 28 passing
   repairs touch neither broken entry, and still undo a median 90% of the damage.
3. Told the two entries and searching both values together: 96% at both widths. A joint
   re-fit of the untold repair's own pair: 30% and 46%.
4. Re-fitting the true entries one at a time: 80% (H = 8) and 64% (H = 32).

### Interpretation

- *Inference:* finding the entries is the gap. Passing the benchmark at H = 32 is weak
  evidence that a repair found the bug, because many other pairs reshape the small damaged
  region (median 4.6% of the input space).

---

## 2026-11-16 — Roadmap #19: mixing the author's guesses with the under-column table

Full write-up: [experiments/breakout-anatomy/COMBINED.md](../experiments/breakout-anatomy/COMBINED.md).
Branch `claude/combined`, stacked on `claude/under-fit`.

### Setup

- **Question.** Does a mix of the author's guesses and #18's under-column table beat both?
- **Design.** 24 variants (every single-row swap both ways, groups of rows), 200 episodes each
  at sticky 0.1 with paired differences; the best swap rerun on 1,000 fresh episodes.
- **Code.** `experiments/breakout-anatomy/combine.py` with 2 tests; `returns.py` now keeps
  per-episode scores and takes a seed offset.

### Results (measured)

1. No mix beats the author's table. Row 8 from the under table gave +8.5 ± 3.7 on 200
   episodes and +0.9 ± 1.8 on 1,000 fresh ones.
2. The author's rows 3–5 lift the under table by 88.2 ± 22.3 (to 445), against +33 for the
   three single swaps added up. Row 5 alone lowers it by 12.9.
3. Losing row 5 costs the author's table 58.3 ± 20.6; no other single row costs more than 14.

### Interpretation

- *Inference:* rows 3–5 are one unit, consistent with all three assuming the same full wall.
  The under fit's gain does not stack with them.

---

## 2026-11-16 — Roadmap #18: fitting the guesses to the under column

Full write-up: [experiments/breakout-anatomy/UNDERFIT.md](../experiments/breakout-anatomy/UNDERFIT.md).
Branch `claude/under-fit`, stacked on `claude/returns`.

### Setup

- **Question.** #8 fitted the accurate table to the nearest catch column, so a corner catch
  counted as a hit. Does fitting to the under column fix the brick rate?
- **Design.** Refit every rising-ball cell on #8's 50 fitting episodes to the nearest catch
  column, the under column, or the corner column; score each at sticky 0 and 0.1 (200 episodes).
- **Code.** `experiments/breakout-anatomy/underfit.py` with 2 tests; `ball_future` now
  returns the under and corner columns.

### Results (measured)

1. The nearest-column refit matches #8's table on all 180 cells (296.4 ± 14.7 at sticky 0.1).
2. The under-column table scores 834 without sticky actions and 356.7 ± 16.3 at sticky 0.1,
   against 481.6 ± 17.3 for the author's table.
3. The corner-column table scores 602 and 305.8 ± 14.7, with 12.4% corner catches.
4. For every table, 95% to 97% of corner catches had the paddle still, one column ahead of
   the ball, on the step before.
5. The under table still dies 5.13 times per 1,000 returns turning at row 4, against 2.36.

### Interpretation

- *Inference:* #8's table was accurate about the wrong target. The right target closes a third
  of the gap; the rest is the author's early-positioning bet on short flights.

---

## 2026-11-16 — Roadmap #17: why the accurate table's returns miss bricks

Full write-up: [experiments/breakout-anatomy/RETURNS.md](../experiments/breakout-anatomy/RETURNS.md).
Branch `claude/returns`, stacked on `claude/sticky-sweep`.

### Setup

- **Question.** Without sticky actions the accurate table scores 569 against 834 because
  fewer of its returns break a brick (#16). What decides whether a return breaks one?
- **Design.** Record every catch's type (under or corner), column, bricks before the next
  catch and turn row, for both tables at sticky 0 and 0.1 (200 episodes). Counterfactual:
  the accurate table with every corner catch returned as an under catch.
- **Code.** `experiments/breakout-anatomy/returns.py` with 4 tests.

### Results (measured)

1. Without sticky actions the ball's path is identical for both tables until the accurate
   table's first corner catch (step 112). It makes 19 in 10,000 steps; the author's makes none.
2. The author's cycle repeats every 30 returns, covers all ten columns evenly and breaks a
   brick on all 833 returns. The accurate table's cycle is 78 returns long and 182 of 733
   returns break nothing.
3. Returning corner catches as under catches gives exactly 834 bricks in 10,000 steps without
   sticky actions, and 1.00 bricks per return at sticky 0.1 (author 0.90, accurate 0.80).
4. But no-brick returns reach the top wall, the safest flights, so the counterfactual dies
   more (2.18 per 10,000 steps against 1.59) and scores 338, against 296 and the author's 482.

### Interpretation

- *Inference:* the author's table does two separate things. It makes few corner catches (2.0%
  against 6.7%), keeping the ball in a cycle that breaks a brick every return, and it parks
  the paddle safely on short flights. Corner catches explain all of the brick-rate gap and
  none of the safety gap.

---

## 2026-11-16 — Roadmap #16: is the author's bet only a sticky-action hedge?

Full write-up: [experiments/breakout-anatomy/STICKY.md](../experiments/breakout-anatomy/STICKY.md).
Branch `claude/sticky-sweep`, stacked on `claude/paddle-motion`.

### Setup

- **Question.** If the author's guesses are a hedge against sticky actions, the accurate table
  should match them without sticky actions, and fall further behind as they become common.
- **Design.** Published, accurate and perfect-prediction variants at sticky 0 to 0.3, 200
  episodes each; scores split into deaths per 10,000 steps and bricks per 10,000 steps.
- **Code.** `experiments/breakout-anatomy/sticky_sweep.py` with 1 test.

### Results (measured)

1. Without sticky actions nothing dies, but the accurate table scores 569 against 834: its
   paddle returns the ball 734 times in 10,000 steps and 569 returns break a brick, against
   834 and 834 for the author's.
2. The accurate table's death rate is about twice the author's at sticky 0.02 to 0.1 (2.0,
   1.95, 2.0) and 1.5 times at 0.3: a constant factor, not a growing one.
3. At sticky 0.1 slower scoring is about 37% of the score gap on a log scale.
4. Perfect prediction tracks the author's table within about 20% in hazard at every setting.

### Interpretation

- The sticky hedge is only part of the advantage; the author's guesses also aim returns at
  bricks. *Inference:* the catch column sets the return angle, so a table a column off changes
  the rally's path.

---

## 2026-11-16 — Roadmap #15: paddle motion at the turn

Full write-up: [experiments/breakout-anatomy/MOTION.md](../experiments/breakout-anatomy/MOTION.md).
Branch `claude/paddle-motion`, stacked on `claude/substitutes`.

### Setup

- **Question.** Why does the accurate table die 3.4 times as often on row-3 flights while
  starting closer to the catch (#13)?
- **Design.** #13's 300 episodes per table, followed step by step: paddle motion at the turn
  (toward, still, away), settled on the catch before arrival, late sticky overrides, out of
  reach.
- **Code.** `experiments/breakout-anatomy/motion.py` with 3 tests.

### Results (measured)

1. Row 3: 17% of the accurate table's flights start with the paddle moving away (2% for the
   author's); they die 7.4 per 1,000 and are 36 of 64 row-3 deaths. Reweighting to the
   author's motion mix gives 0.89 per 1,000 instead of 2.24.
2. Row 4: 43% of the author's flights start with the paddle parked on the catch (1 death in
   21,757); the accurate table's still paddles are 1.10 columns off and die 6.5 per 1,000.
3. 92 of the author's 155 deaths are out of reach; 189 of the accurate table's 240 deaths had
   a sticky override in the last two steps, and only 35 were out of reach.

### Interpretation

- *Inference:* the author's guesses trade a few unwinnable flights for many with time to spare;
  the accurate table turns them into close calls that one sticky action can lose.

---

## 2026-11-16 — Roadmap #14: which weights stand in for the broken one

Full write-up: [experiments/locate-repair/SUBSTITUTES.md](../experiments/locate-repair/SUBSTITUTES.md).
Branch `claude/substitutes`, stacked on `claude/breakout-deaths`.

### Setup

- **Question.** #10's passing edits mostly change an entry other than the corrupted one.
  Which entries, and why do they work?
- **Design.** H = 32 and 128, 50 seeds each, 1,024 practice examples. A substitute is any
  other entry whose smallest practice-perfect edit passes and undoes ≥95% of the damage.
  Classed as the output-logit mirror, same unit, same output or elsewhere; units compared by
  input-direction cosine. The min-change editor's pick is recorded.
- **Code.** `experiments/locate-repair/substitutes.py` with 5 tests.

### Results (measured)

1. In all 37 output-weight cases, the same unit's weight into the other logit repairs exactly
   when the corrupted entry does (an exact symmetry of a 2-logit classifier).
2. The editor picks that mirror in 6 of 41 cases at H = 32; counting it as found raises
   localization from 46% to 61%. At H = 128 it is 2 of 46 (35% to 39%).
3. Substitutes in other units point the same way: median input-direction cosine 0.94 and 0.89,
   with 55% and 48% above 0.9 against 15–16% for all other units.
4. The corrupted entry itself is a valid near-complete repair in 83% and 96% of cases.

### Interpretation

- Part of the H = 32 mislocalization is a scoring artifact: the mirror is the same repair.
  *Inference:* at H = 128 near-parallel units, not the mirror, account for most stand-ins,
  and the editor often settles for a partial repair (only 8 of its 26 elsewhere picks are
  near-complete).

---

## 2026-11-16 — Roadmap #13: what kills the accurate Breakout table

Full write-up: [experiments/breakout-anatomy/DEATHS.md](../experiments/breakout-anatomy/DEATHS.md).
Branch `claude/breakout-deaths`, stacked on `claude/breakout-rowswap`.

### Setup

- **Question.** Do the deaths #12 attributes to rows 3–5 really come from the shortest flights?
- **Design.** Trace every falling flight in 300 episodes under the published and the accurate
  table: turn-down row, steps left, paddle distance to the catch, death.
- **Code.** `experiments/breakout-anatomy/deaths.py` with 1 test.

### Results (measured)

1. About 70% of deaths under both tables follow a bounce off the bottom row of the wall (row 4
   turn, about 5 steps left); ceiling bounces almost never kill (3 in 75,519 flights).
2. On those flights the accurate table starts 1.52 columns away against 0.99, and dies 5.44
   against 2.14 times per 1,000.
3. At row 3 the accurate table starts closer (0.76 against 0.94) yet dies 3.4 times as often.

### Interpretation

- #12's mechanism is confirmed for the shortest flight. *Inference:* paddle motion at the turn
  matters too, which position alone does not capture.

---

## 2026-11-16 — Roadmap #12: where the author's Breakout guesses win

Full write-up: [experiments/breakout-anatomy/ROWSWAP.md](../experiments/breakout-anatomy/ROWSWAP.md).
Branch `claude/breakout-rowswap`, stacked on `claude/breakout-closedloop`.

### Setup

- **Question.** Which ascending cells carry the ~200-point gap between the author's table and
  #8's accurate one?
- **Design.** Swap one ball row (then groups of rows) between the two tables in both
  directions; 300 episodes each.
- **Code.** `experiments/breakout-anatomy/rowswap.py` with 2 tests.

### Results (measured)

1. Accurate guesses in rows 3–5 only (54 of 180 cells): 284.3, no better than all-accurate (297.0).
2. Author's guesses in rows 3–5 only: 448.3 against published 497.2.
3. Rows 0–2 and 6–8 move the score by at most 22 points.

### Interpretation

- *Inference:* rows 3–5 are where an ascending ball may bounce off the wall at once, the
  case with the least time left. The author's table assumes that bounce; the accurate one
  averages it away. This is #9's deadline idea, localized; #9's model missed it because it
  assumed the paddle had already reached the guess.

### Post angle

The payoff for the Breakout series: "The 'wrong' guesses only matter in three rows, and
there they are a bet on the worst case."

---

## 2026-11-16 — Roadmap #11: refitting the Breakout guesses on their own play

Full write-up: [experiments/breakout-anatomy/CLOSEDLOOP.md](../experiments/breakout-anatomy/CLOSEDLOOP.md).
Branch `claude/breakout-closedloop`, stacked on `claude/practice-size`.

### Setup

- **Question.** Is #9's gap between the author's table and accurate tables a distribution-shift
  effect (fitted on the published network's play, then installed)?
- **Design.** Three starting tables (published, physics-only full wall, #8's best mean
  distance); each round plays episodes 5000–5049 with the current table, refits visited cells
  for mean distance, scores 300 episodes. Three refits; final tables at the full protocol.
- **Code.** `experiments/breakout-anatomy/closedloop.py` with 1 test.

### Results (measured)

1. The accurate table is 1.11 columns off on published play and 1.42 on its own: shift is real.
2. No refit scores above 340 (300 episodes); finals 238.6, 278.4 and 319.3 against 502.9.
3. One refit of the published table on its own play drops it from 497 to 297.

### Interpretation

- Fitting the guesses for landing accuracy is what costs the score, on any play distribution
  tried. *Inference:* the author's table works through the paddle's path, not the landing.

### Post angle

Part three of the Breakout series: "Optimising the part you understand broke the part you didn't."

---

## 2026-11-16 — Roadmap #10: how many examples it takes to find the broken weight

Full write-up: [experiments/locate-repair/PRACTICE.md](../experiments/locate-repair/PRACTICE.md).
Branch `claude/practice-size`, stacked on `claude/breakout-hedge`.

### Setup

- **Question.** #7's not-told editor rarely finds the corrupted entry. Is that a shortage of
  practice data?
- **Design.** #7's cases at H = 8, 32, 128 (50 seeds each, 141 kept), practice sets of 32 to
  1,024 examples with the held-out test fixed, min-change editor on a 1e-2 value grid.
- **Code.** `experiments/locate-repair/practice_size.py` with 2 tests.

### Results (measured)

1. Passing rises from 47–50% to 85–92% as practice grows from 32 to 1,024 examples.
2. Editing the corrupted entry at 642 parameters rises only from 2% to 34%; 25 of 40 passes
   there still edit a different entry.
3. Median damage undone rises from 73–88% to 97–100%; entries with a practice-perfect edit
   fall from 130 to 5 at H = 128.

### Interpretation

- With enough data, wrong-entry passes are near-complete repairs. *Inference:* several
  entries can stand in for the broken one almost everywhere, so localization has no
  behavioural answer in many cases.

### Post angle

"More data fixes the repair, not the diagnosis." Follow-up to #7's post.

---

## 2026-11-16 — Roadmap #9: why the author's rough guesses win

Full write-up: [experiments/breakout-anatomy/HEDGE.md](../experiments/breakout-anatomy/HEDGE.md).
Branch `claude/breakout-hedge`, stacked on `claude/locate-repair`.

### Setup

- **Question.** #8 found the released ascending guesses (1.76 columns off) score 502.9 and a
  more accurate brick-blind table (1.12 off) scores 294.1. What does the author's table get right?
- **Design.** Rebuild a table from physics alone (landing if rows 1–3 were a full wall, the
  earliest possible bounce); fit a deadline-aware table and a most-likely-landing table on
  184,502 ascending states from held-out episodes 5000–5049; install each in units 2 and 3
  and score at the published protocol (1,000 episodes, sticky 0.1).
- **Code.** `experiments/breakout-anatomy/hedge.py` with 5 tests.

### Results (measured)

1. All 180 published ascending targets lie on a full-wall catch column. The guesses are the
   earliest-bounce landings.
2. The physics-only full-wall table scores 493.0 ± 7.9 (published 502.9 ± 7.7).
3. The deadline-aware fit is identical to #8's best-mean-distance table on all 180 cells
   (294.1): lateness, as modelled, never binds for the accurate table. The published table
   is the most often late by that model.
4. Fitting for exact hits (73% vs 43%) scores 398.2 ± 7.7: better than mean distance, still
   about 100 points below the published and full-wall tables (42% exact hits).

### Interpretation

- No accuracy measure computed on the published network's play ranks the tables correctly.
- *Inference, untested:* fitted tables are fitted to states the published network visits and
  then change which states occur; the physics table needs no fit. A closed-loop refit would test it.

### Post angle

"The hand-written network's 'wrong' guesses are physics: it assumes the earliest bounce."
Pairs with #8's "a better guess made it worse".

---

## 2026-11-16 — Roadmap #7 (offline half): passing the test by fixing the wrong weight

Full write-up: [experiments/locate-repair](../experiments/locate-repair/README.md).
Branch `claude/locate-repair`, stacked on `claude/breakout-targets`.

### Setup

- **Question.** If the editor is not told which entry is broken, does one-entry repair
  still work, and does it find the fault?
- **Design.** 2-input ReLU networks with 2–128 hidden units (12–642 parameters), one
  corrupted entry anywhere, the benchmark's own datasets and success rule. Model-free
  strategies with the practice labels: told the entry; not told and pick the smallest
  relative practice-perfect edit; whether any entry works; the scripted random control.
  100 cases per width. "Damage undone" is measured on 20,000 uniform inputs.
- **Code.** `experiments/locate-repair/analyze.py` with 4 tests. It imports the benchmark's
  dataset builder but changes nothing in `weight_space_lab/`.
- **Not run.** Model editors: they need paid API or CLI allowance and approval.

### Results (measured)

1. Told the entry: 76–86% success at every size. Not told: 70% at 12 parameters, 47–52%
   from 42 to 642.
2. The not-told editor edits the corrupted entry 48% of the time at 12 parameters and 3%
   at 642. At 642 parameters, 44 of its 47 successes edit a different entry.
3. Some single edit passes the rule in 86–95% of cases, more as networks grow.
4. Successes are partial repairs: the not-told editor undoes a median 75–79% of the
   input-space damage at H ≥ 32, against 95–96% when told.
5. Random control: about 2% not told, 14–26% told.

### Interpretation

- With many parameters, plenty of single edits pass the benchmark rule without touching
  the fault. A locate-and-repair benchmark has to score localization or dense behaviour,
  not only the pass rule.

### Post angle

"Your model-editing benchmark can be passed by fixing the wrong weight." Pairs with the
task-geometry post on the benchmark's rule.

---

## 2026-11-16 — Roadmap #8: would a better guess keep the Breakout network alive longer?

Full write-up: [experiments/breakout-anatomy/TARGETS.md](../experiments/breakout-anatomy/TARGETS.md).
Branch `claude/breakout-targets`, stacked on `claude/breakout-anatomy`.

### Setup

- **Question.** #2 found the network's ascending guesses are often wrong because it cannot
  see bricks. Do better guesses help?
- **Design.** Replace only the ascending target code with five alternatives (follow the
  ball, center, no-brick physics, a best-fit brick-blind table, and the true landing
  column), plus a pure oracle policy. Full protocol for each, plus a catch profile on 20
  further episodes: how long the paddle had been still when each catch happened.
- **Code.** `experiments/breakout-anatomy/targets.py`, 4 tests in
  `tests/test_breakout_targets.py`.

### Results (measured)

1. **Perfect ascending prediction does not help:** 480.6 versus 502.9 for the published
   guesses (−22.3 ± 9.2 on the same seeds).
2. **A more accurate brick-blind guess is much worse:** mean error 1.12 versus 1.78
   columns, but 294.1 points and more than twice the death hazard.
3. **97 of 100 deaths in the catch profiles are last-moment arrivals:** the policy chose
   a move on the same step the ball reached the paddle row. All 12 of the published
   network's deaths there are of this kind; such catches are 0.7% of its catches.
4. **The network trades survival for score.** A pure oracle policy dies six times less
   often per step (89.5% of episodes reach the ceiling) but breaks bricks at a third of
   the rate, scoring 231.7.

### Interpretation

- What limits the network is how often it arrives just in time, not how accurate its
  average guess is.
- Why the author's guesses beat a more accurate table is open. Untested hypothesis: they
  assume the earliest possible brick bounce, which hedges the worst case.

### Post angle

"Better predictions made the hand-written network worse." Pairs with #2 as a two-part
post: the network does what its author said; and the author's imprecise guesses are
better than accurate ones.

---

## 2026-11-16 — Roadmap #2: does the hand-written Breakout network do what its author said?

Full write-up: [experiments/breakout-anatomy](../experiments/breakout-anatomy/README.md).
Branch `claude/breakout-anatomy`, stacked on `claude/p50c-budget-curve`.

### Setup

- **Question.** The released Experiment 5 network came with a plain-language description
  from the model that wrote it. Is that description true, and which parts carry the score?
- **Ground truth.** An exact replica of MinAtar's ball rules gives, for every state, the
  two columns where the paddle survives. A test checks it against the environment.
- **Runs.** Semantics on episodes 0–19 (158,482 states). Thirty variants at the full
  protocol (1,000 episodes each): eight second-layer silencings, twelve weight-noise
  draws, three coarser target codes, greedy actions, and five sticky-action settings.
- **Code.** `experiments/breakout-anatomy/anatomy.py`, 7 tests in
  `tests/test_breakout_anatomy.py`. Nothing under `skills/` or `weight_space_lab/` changed.
- **Check.** The intact run reproduces the committed full replay exactly (502.880 ± 7.747).
- **Environment.** CPU only. Codex and Hugging Face are still unreachable from this
  container (proxy refuses both), so no second model was used.

### Results (measured)

1. **The description is accurate where it matters.** The second layer selects the right
   direction context on 99.4% of states. Descending landing targets are within half a
   column of a true catch column on 99.2% of states (100% from row 3 down).
2. **The network is blind to bricks.** The brick channel has no weights. Its ascending
   targets are within half a column only 42% of the time (mean error 1.76 columns),
   improving as the ball climbs.
3. **Every death is caused by sticky actions.** With sticky actions off, 1,000 of 1,000
   episodes reach the 10,000-step ceiling, each scoring exactly 834. Scores fall smoothly
   as stickiness rises: 772.8 at 0.025, 696.3 at 0.05, 502.9 at 0.1, 149.3 at 0.2.
   Greedy and sampled actions give identical results.
4. **Half of the second layer is nearly redundant.** Silencing any of the four "move with
   the ball" comparisons drops the score below 9. Silencing a "move against the ball"
   comparison costs 3–4% (ascending) or 45% (descending).
5. **It is a precision program.** 1% multiplicative weight noise drops the mean to 18.9
   in one of three draws; 3% noise drops all three below 10. Rounding the target code
   to 2-column steps drops the mean to 4.5.

### Interpretation

- The model wrote a correct program for a noise-free game. Its only failure mode is the
  environment overriding its chosen move, which its design does not anticipate.
- The ascending predictions are about two columns off half the time, yet the comparisons
  that use them are essential. Approximate pre-positioning is load-bearing.
- The noise fragility is probably the second layer's 15-unit gate margin (inference;
  not isolated by tensor).

### Post angle

"An LLM wrote a neural network by hand. We checked whether it does what the LLM said it
does." It mostly does; it is blind to bricks; it never dies without injected noise.

### Still open

- The fresh network (1,076 nonzero weights) is still missing, so released versus fresh is
  not done.
- Would a better ascending guess buy survival? Added as roadmap #8 and run next.

---

## 2026-11-14 — Roadmap #1: is p50c a math edit or a stop-thinking edit?

Full write-up: [experiments/p50c-termination](../experiments/p50c-termination/README.md).
Branch `claude/p50c-budget-curve`, built on the November 10 audit branch.

### Setup

- **Plan versus what ran.** The roadmap planned token-budget curves from this lab's full
  GPU run. Neither the GPU nor that run's transcripts are available here: no GPU in this
  container, and the full run has not been committed (finding 3 below).
- **Data used instead.** The upstream repository (revision `4d09ba6`) turned out to
  release its own **full** MATH-500 evaluation records for the unedited model and all
  three agent-written edits (p10c, p25c, p50c). That is per-sample correctness for 500
  problems × 4 samples, shard aggregates, and the full text of sample 0 for problems
  0–199.
- **Design.** A paired closure analysis on those records (CPU, about 10 s): split the
  problems by whether each model closed its `</think>` block, and attribute the accuracy
  difference cell by cell.
- **Code.** `qwen/termination.py` with 11 new tests in `qwen/test_termination.py`; the Qwen
  suite now has 19/19 passing. It is a separate module on purpose: `reproduce_qwen.py`,
  `lab.py` and `weight_space_lab/` are unchanged, because their hashes pin in-progress
  jobs and benchmark campaigns.
- **Environment.**
  - Codex is still unreachable: no CLI or sign-in, and `api.openai.com` is blocked.
    No second model was used.
  - To fetch the pinned Qwen tokenizer, I created a `weight-space-lab` cloud environment
    that allows Hugging Face hosts (with approval). The `tokenizer.json` download was
    still refused by this session's permission policy, so token-budget curves are
    implemented and tested but not run.
- **Checks.**
  - The per-problem records reproduce every published aggregate exactly.
  - This lab's grader matches the recorded grade on 800/800 transcripts.
  - The analysis output is byte-identical across two runs.

### Results (measured)

| Arm | Accuracy, 2,000 samples | Capped | Gain over unedited |
|---|---:|---:|---:|
| unedited | 7.90% [6.10, 9.90] | 93.0% | — |
| p10c | 12.35% [10.20, 14.65] | 87.3% | +4.45 [2.80, 6.15] |
| p25c | 14.10% [11.80, 16.50] | 86.4% | +6.20 [4.55, 7.90] |
| p50c | 48.10% [44.35, 51.75] | 33.1% | +40.20 [36.85, 43.55] |

Paired by problem (sample 0, problems 0–199), p50c gains +38.5 points [31.5, 45.5]:

- **103% of the gain** comes from the 140 problems where the original never closed its
  reasoning and p50c did. There p50c is correct on 81 and the original on 2.
- **On the 12 problems where both closed, the edit adds nothing:** the original gets 12
  right and p50c 11.
- **185 of the original's 187 unfinished samples end in a repetition loop** (98.4–98.9%
  for thresholds 0.05–0.2). The loop begins at a median of 22% of the way through the
  text.
- **In 78 of the 81 rescued problems, the original never wrote the right answer in any
  `\boxed{}`** within its 32,768 tokens.
- **p10c and p25c show the same pattern.** Their gains also come entirely from rescued
  problems, with no change where both finished; each stops finishing 3–4 problems the
  original had solved.

*Interpretation:* p50c is an **anti-loop edit**, consistent with upstream's description
of it as removing a "loop direction". The roadmap hypothesis ("gain mostly from
finishing; little difference among finished samples") holds. But the original was not
"done thinking and unable to stop": it got stuck before reaching an answer. Better
reasoning on the rescued problems cannot be fully excluded, because the original's
behaviour past its loop is unobservable, and the both-finished comparison rests on 12
easy problems.

### Surprises

1. **The data already existed.** Upstream released full-run records for all three edits,
   so roadmap #1 did not need a GPU to get a first answer.
2. **My first "answer found anywhere" scan was wrong, and the tests caught it.** It
   handed text slices to `last_boxed`, which returns the *last* box in a slice and drops
   everything before `</think>`. Boxed answers inside the reasoning were missed or
   mis-ordered. A dedicated `boxed_at` parser fixed it before any numbers were recorded.
3. **p50c's loops sometimes move to the answer section.** 42 of its 200 samples close
   their reasoning without a boxed answer, and 26 of those 42 end in a repetition loop
   *after* `</think>` (for example, "Wait, I will provide the answer…"). A lenient reading
   recovers only 4, so missing boxes cost about 2 points rather than hiding a large gain.
4. **Answers graded while unfinished show up at full scale too.** The original's
   accuracy (7.9%) exceeds its finish rate (7.0%), so at least 0.9 points come from capped
   samples. In the transcripts, 2 of its 15 correct answers come from unfinished reasoning.

### Still open

- Token-budget curves. They need the pinned `tokenizer.json`, which a person can allow
  in this session's permissions, or the lab's own run with stored token IDs.
- The lab's own full GPU run, and the prefix-equivalence check on 10 problems.
- Whether the edit changes reasoning on hard problems. Roadmap #6's dose–response on
  problems the original does not loop on would address it.

---

## 2026-11-10 — Full audit, two new analyses, roadmap

**Environment for today's runs:** Linux cloud container, 4 CPUs, no GPU, Python 3.13.18,
NumPy 2.5.4, SafeTensors 0.8.0, MinAtar 1.0.15, PyTorch 2.14.0 (CPU use only). No
hosted-model, Fireworks or paid API calls were made. Codex was unavailable: no CLI, no
sign-in, and `api.openai.com` is blocked by this environment's network policy. Every
review and design check below was done in this session without a second model.

### What exists

| Component | Status | Main entry point |
|---|---|---|
| Checkpoint toolkit: inspect, tensor slices, bounded set/add/scale edits, MLP evaluator, paired compare | Working; NPZ/SafeTensors, F16/F32/F64 | `lab.py inspect / tensor / edit / evaluate / compare` |
| Offline sign-classifier demo (prewritten patch) | Working | `lab.py demo` |
| Research controller: metered proposals/feedback, freeze, one final evaluation, hash-chained events | Working; validated with manual controls only | `lab.py experiment …` |
| JSONL editor-session bridge for external editors | Working; validated with a scripted control only | `lab.py editor serve / audit` |
| Weight-editing benchmark: 4 synthetic families, matched cases, scripted controls, Claude/Fireworks API and Claude/Codex CLI adapters, scorecards | Working; **no native API trials run yet** | `lab.py benchmark …` |
| MinAtar Breakout reproduction of Takehi et al. Experiment 5 | Working end to end on CPU | `lab.py breakout` |
| Qwen3.5-0.8B p50c reconstruction and MATH-500 runner | Reconstruction verified; 2-problem pilot done; **full run not in the repo** | `qwen/` |
| Configurable GPU layouts (open PR [ZayneMason/weight-space-lab#4](https://github.com/ZayneMason/weight-space-lab/pull/4)) | Tests pass; no GPU inference run | `lab.py gpu …` (PR branch) |
| Codex plugin manifest and DSH bundle | Node tests pass; active-profile install unverified | `index.js`, `.codex-plugin/` |

### What each experiment has shown so far

**Breakout: writing a policy network directly (attributed reproduction).** Recorded
November 4 in [RESULTS.md](../RESULTS.md):

| Network | Mean score | SE | Episodes |
|---|---:|---:|---:|
| Published (Takehi et al.) | 517.411 | 7.417 | 1,000 |
| Released weights, local replay | 502.880 | 7.747 | 1,000 |
| Fresh GPT-6 Astra response, local | 511.530 | 5.547 | 1,000 |
| Published PPO, best 100M / 500M steps (reference only) | 15.779 / 25.344 | — | — |

The local replay sits 1.35 combined SEs below the published mean. The fresh network
scores 8.65 points above the replay, with a paired-difference SE of 9.00, so it matches
the released network but is not better. The fresh response used no tools and no
feedback, took 623 s, and wrote 76 edit statements: **1,076 nonzero weights out of
20,486** (5.3%). The published answer has 92 statements and reconstructs all 20,486
parameters exactly, 815 of them nonzero (see analysis 3 below).

**Qwen p50c (attributed reproduction).** The reconstruction is exact: 130,188 BF16
entries changed (65,260 in the layer 15 MLP down-projection, 64,928 in layer 19), both
edited-tensor hashes match, and all 318 other exported tensors are bitwise unchanged.
The pilot used 2 MATH-500 problems × 4 samples per arm with a 32,768-token ceiling:

| Measure | Original | p50c |
|---|---:|---:|
| Correct samples | 1/8 | 6/8 |
| Samples that hit the token ceiling | 8/8 | 1/8 |
| Total generated tokens | 262,144 | 134,717 |

Per sample (from `evidence/qwen/pilot-samples.json`): on problem 0 the original's only
correct sample **was one cut off at the ceiling**, and p50c solved it 4/4 in 2.3k–9.2k
tokens. On problem 1 the original hit the ceiling 4/4; p50c got 2/4 right, finishing in
23k–27k tokens, with one wrong finished answer and one sample at the ceiling.

**Controller and editor bridge (infrastructure validation).** With prewritten controls:
the regressing candidate fixed 8 and broke 8 practice examples; the feasible candidate
fixed 8 and broke none. Final accuracy went from 75% to 100% on 256 held-out examples
with zero regressions. The stdio bridge reproduces the same final outcome through a
real subprocess. These are plumbing checks, not editor results.

**Editing benchmark.** The recorded 12-case calibration gave: feasible control 12/12,
no-op 0/12, random 1/12. The small CLI pilots (structured interface v2, threshold task
seed 47) gave:

- Codex CLI (`gpt-6.1-sol`, low effort): froze a candidate scoring 255/256 with 100%
  repair and **1 regression**, so it failed the zero-regression rule. It used 1 feedback
  call, 10 tool requests and 95,532 input / 420 output tokens.
- Claude CLI (`sonnet`, low effort): 4 valid inspection/status actions, then
  `error_max_structured_output_retries` before selecting anything, so the outcome is
  unmeasured.

Earlier v1 attempts by Codex scored 253/256 with 3 regressions (seed 11) and 250/256
with 6 regressions (seed 23). Every Codex attempt so far repaired the target region and
then broke a few protected examples.

### Runs today

| Check | Result |
|---|---|
| Main Python suite (`main`) | 87 tests: 86 passed, 1 skipped (optional legacy reference directory) |
| Qwen unit suite (CPU torch) | 8/8 passed |
| Node plugin suite | 2/2 passed |
| PR #4 branch: main suite / Qwen suite | 105 tests: 104 passed, 1 skipped / 9/9 passed |
| PR #4 `gpu inspect`, `gpu plan` with no GPU | Reported "unavailable" and planned zero workers, without errors |
| `lab.py demo` + `verify` | 0/6 → 6/6, 2 of 4 entries changed; verify passed |
| `scripts/check_controller.py` + audit | Matches the recorded controls; audit passed (13 events) |
| `scripts/check_editor_session.py` + audit | Frozen, then finalized separately; audit passed |
| `scripts/check_benchmark.py` | Feasible 12/12, no-op 0/12, **random 2/12** (fresh private seed; recorded run had 1/12) |
| Breakout smoke (8 episodes, 1,000 steps) | 77.875, SE 3.0905, matching the recorded value; all assets and the 92-statement reconstruction verified |
| **Breakout full replay** (1,000 episodes, 10,000 steps, 3 workers, 62 s) | **502.880, SE 7.747: an exact match** to the recorded local replay |

### New analyses today

**1. Breakout scores mostly measure survival time** ([experiment](../experiments/breakout-ceiling/README.md)).
From the 1,000 retained episodes of today's replay:

- **517 of 1,000 episodes (51.7%) reach the 10,000-step ceiling.** Capped episodes
  average 697.5 points; episodes that end average 294.6.
- Score correlates with episode length at **r = 0.992** (R² = 0.983), at a nearly
  constant 0.069 points per step.
- *Interpretation:* the headline mean is mostly "how long the policy keeps the ball in
  play", capped by the protocol. Survival curves are the natural way to compare the
  released and fresh networks. That needs the fresh checkpoint, which exists only on
  the original workstation (`runs/breakout-reproduction/fresh-astra/fresh.npz`).

**2. Benchmark geometry: precision, not discovery, decides success** ([experiment](../experiments/task-geometry/README.md)).
I swept the single editable entry for 100 seeds × 4 families (400 cases):

- **Random baseline ≈ 11.4%** (1.36 successes expected per 12 cases). The recorded 1/12
  and today's 2/12 are both ordinary draws. By family: threshold 5.7%, slope 24.2%,
  routing 9.3%, hidden 6.2%.
- **A perfect practice score passes the held-out rule only 27–42% of the time.**
- **The zero-regression rule punishes overshoot 2–7× more than undershoot** (median
  tolerances). The practice-perfect window lies mostly past the correct value, so its
  midpoint succeeds only 20–38% of the time, while **the smallest practice-perfect edit
  succeeds 73–83%**.
- *Interpretation:* every recorded Codex attempt shows the overshoot signature, with
  full repair and a handful of regressions. With one no-op feedback call, an editor
  has all the information it needs to compute the minimal edit exactly. The tasks
  currently test careful arithmetic plus tool discipline. Localization is given away
  by the `describe` method, which names the editable entry.

**3. The released Breakout network is an 815-weight program** ([pilot](../experiments/breakout-circuit/README.md)).

- Only **815 of 20,486 parameters are nonzero** (4.0%). The network uses 6 of 32
  first-layer units, 8 of 32 second-layer units, and drives only the left and right
  logits. Its constants look hand-set (±1000, ±200, 30; 61 distinct values). The
  "20,486-parameter network" in earlier docs is the architecture, not the program.
- Single-unit zero-ablations over 200 episodes: the intact copy scores 481.65 (equal,
  episode for episode, to the first 200 of today's full replay). **Removing any one
  first-layer unit drops the mean to 0.5–8.4**, with episodes lasting 11–97 steps.
- *Interpretation:* the network is a tightly coupled circuit with no redundant units
  at layer 1. The fresh network (1,076 nonzero weights) is the same kind of object,
  which makes a released-versus-fresh circuit comparison feasible (roadmap #2).

### Audit findings: broken, fragile or unfinished

1. **The Claude CLI result is probably caused by the harness, not the model**
   (`weight_space_lab/benchmark_cli.py`). The Claude adapter sets
   `MAX_STRUCTURED_OUTPUT_RETRIES=0` and `--max-turns 2` and passes a 9-branch `anyOf`
   JSON schema. The recorded failure is `error_max_structured_output_retries`. The Codex
   adapter has no equivalent zero-retry limit. So the Claude-versus-Codex contrast is
   confounded by transport settings. *Not yet verified*: the retained
   `stdout.txt`/`stderr.txt` in `runs/cli-structured-pilot/` on the original
   workstation should be checked, and the case rerun with retries allowed (roadmap R3).
2. **Truncated generations can be graded correct** (`qwen/reproduce_qwen.py`, `grade` /
   `last_boxed`). This follows the released extractor. When `</think>` never appears,
   the search covers the whole unfinished text, so a `\boxed{}` written mid-reasoning
   counts. The original model's only pilot success was a capped sample. Full-run
   reports should also give accuracy with capped samples scored as wrong, and accuracy
   conditional on finishing.
3. **The full MATH-500 comparison is not in the repo.** REPORT.md (November 4) says it
   was launched. No full results, partial progress or status are committed. Note that
   `job.json` pins `script_sha256`, so a run started with the `main` runner cannot be
   resumed under the PR #4 runner, which changes `reproduce_qwen.py`. Finish it with the
   code that started it, or start a fresh declared run.
4. **No native API benchmark trials have run.** All 16 API slots in
   `pilot.example.json` are pending. The only model outcomes are 2 v2 CLI pilots and 3
   older v1 attempts.
5. **No external-agent controller trial has run.** The threshold-repair protocol is
   still manual-control-only.
6. **NumPy pin drift:** CI installs `numpy==2.3.5`, while `requirements.txt`, the `uv`
   script headers and all recorded results use 2.5.4. The tests pass on both, but CI
   is not testing the pinned version. (`qwen/requirements.in` pins `<2.5` for the GPU
   environment; that is a separate environment, but worth stating.)
7. **Random-control rates in docs read as point estimates.** "1/12" in VERIFICATION.md
   and BENCHMARK.md comes from one private draw. The geometry analysis gives the
   expected rate. Scorecards could report the analytic expectation next to the draw.
8. **Fresh Breakout checkpoint and raw transcripts are local-only.** Without
   `fresh.npz`, no new analysis of the agent-written network (survival, sparsity,
   ablation) can run from the repository.
9. **The benchmark gives the location away.** `describe` reveals the exact editable
   tensor and index, and practice example IDs reveal their stratum. This is documented
   as deliberate, but it means current tasks cannot measure localization.
10. **README.zh.md** was not updated with today's findings. It still describes the
    November 5 state.

### Next

See [ROADMAP.md](ROADMAP.md). The two highest-ranked experiments reuse runs that are
already planned (the full MATH-500 pass) or nearly free (post-hoc truncation analysis
of its transcripts).
