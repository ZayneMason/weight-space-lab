# Locate-and-repair: passing the test by fixing the wrong weight (roadmap #7, offline half)

**Date:** November 16, 2026 · **Status:** reference strategies done; model editors not run ·
**Cost:** CPU, about 5 minutes · **No model or API calls**

The editing benchmark names the entry to change, so it measures arithmetic and tool
discipline but not *finding* a fault. This analysis removes that hint and asks what
model-free strategies can do, as the network grows.

```bash
python experiments/locate-repair/analyze.py --widths 2 8 32 128 --seeds 100 --output runs/locate-repair
```

## Task

- A 2-input ReLU network with H ∈ {2, 8, 32, 128} hidden units and 2 outputs: 5H + 2
  parameters (12 to 642).
- A random teacher (redrawn if it predicts one class almost everywhere). One entry,
  chosen uniformly among all parameters, is shifted by ±1.6 times its tensor's standard
  deviation. Entries whose corruption is (nearly) invisible are redrawn: the task needs
  16 practice and 128 held-out inputs the corruption gets wrong.
- Practice and held-out sets, and the success rule, are the benchmark's own
  (`weight_space_lab.benchmark_tasks._dataset`): 16 + 16 practice examples; on 256
  held-out examples, at least 90% repair and zero regressions.
- The editor may change any one entry. It sees all weights and the practice inputs, and
  one no-change feedback call gives it all 32 practice labels.

## Strategies

| Strategy | What it does |
|---|---|
| given | Told the entry; takes its smallest edit that is perfect on practice (the best strategy in [task geometry](../task-geometry/README.md)) |
| min-change | Not told; for every entry finds its smallest practice-perfect edit, then picks the entry whose edit is smallest relative to its tensor's spread |
| any | Whether *some* entry's smallest practice-perfect edit succeeds (upper bound for rules that pick among them) |
| random / random-given | The benchmark's scripted random control: three uniform values in [−3, 3], chosen by practice accuracy, then fewer practice regressions; random entry, or the true one |

Values are searched on a 1e-3 grid within ±4 of the current value. "Damage undone"
compares the network with the teacher on 20,000 uniform inputs: 1 means it agrees
everywhere again, 0 means no better than before the edit.

## Results (measured, 100 cases per width)

| H | Parameters | given | min-change | any | random-given | random | min-change edited the corrupted entry | Entries with a practice-perfect edit (median) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 12 | 81% | 70% | 86% | 19.0% | 4.9% | 48% | 2 |
| 8 | 42 | 76% | 49% | 84% | 14.4% | 2.1% | 27% | 6 |
| 32 | 162 | 86% | 52% | 93% | 24.7% | 2.3% | 7% | 25 |
| 128 | 642 | 80% | 47% | 95% | 25.5% | 2.1% | **3%** | 104 |

![Success and localization against network size](figures/locate-repair.png)

1. **Hiding the location costs about 30 points, and the cost stops growing.** Against
   the location-given strategy, min-change loses 11 points at 12 parameters and 27–34
   points from 42 to 642 parameters, holding near 50%. The hypothesized collapse with
   size does not happen.
2. **Localization does collapse.** The min-change editor edits the corrupted entry in 48% of
   cases at H = 2 and 3% at H = 128. At H = 128, 44 of its 47 successes edit a *different*
   entry.
3. **Some single edit almost always passes.** The share of cases where at least one entry's
   smallest practice-perfect edit meets the rule grows from 86% to 95% with size; at
   H = 128 a median of 16 different entries pass.
4. **Passing the rule is not the same as undoing the damage.** Told the location, the
   smallest edit undoes a median 95–96% of the damage at every size. Not told, min-change
   undoes 95% at H = 2 but 75–79% at H ≥ 32. Among min-change *successes* at H = 32 and
   128, 15–19% undo less than half the damage.
5. **The random control stays near 2%** without the location, against 14–26% with it.

## Interpretation

- In bigger networks there are many single edits that restore behaviour on the inputs the
  benchmark samples, and practice data rarely singles out the real fault. An editor
  that only needs to pass the rule does not need to find the bug; an editor judged on
  finding it would fail almost always.
- A located benchmark should therefore score localization separately from passing, or
  test on dense inputs near the damaged region, or both.
- The corruption's footprint shrinks with width (median damage 14.9% of input space at
  H = 2, 1.5% at H = 128), which is part of why other entries can compensate. This is a
  property of the task generator, not only of the networks.

## Limits

- Model editors (the roadmap's main arms) were not run: they need paid API calls or CLI
  allowance and explicit approval.
- One corruption size (1.6 standard deviations) and one architecture family. The
  generator redraws invisible corruptions, so every case has a visible fault.
- Min-change is one reasonable selection rule among many; "any" bounds what a better rule
  could reach with practice data alone.

Evidence: [`evidence/locate-repair/locate-repair.json`](../../evidence/locate-repair/locate-repair.json)
(per-case rows and the summary). Code: `analyze.py`; 4 tests in `tests/test_locate_repair.py`.

Follow-up: [PRACTICE.md](PRACTICE.md) (roadmap #10) grows the practice set from 32 to
1,024 examples.
[SUBSTITUTES.md](SUBSTITUTES.md) (roadmap #14) asks which entries stand in for the
broken one: an exact output-logit mirror, or units that point the same way.
[TWO_FAULTS.md](TWO_FAULTS.md) (roadmap #20) breaks two entries at once: passing repairs at
H = 32 mostly touch neither, and only a told joint search reliably fixes both.
[PAIR_SEARCH.md](PAIR_SEARCH.md) (roadmap #21) searches pairs of entries: practice data
often fits several pairs perfectly, so even an exhaustive search finds the broken pair only
about half the time.
[ACTIVE_QUERIES.md](ACTIVE_QUERIES.md) (roadmap #22) asks the teacher about inputs where the
candidate pairs disagree: 64 such examples settle most cases that 1,024 random ones could not.
[PAIR_SCREEN.md](PAIR_SCREEN.md) (roadmap #23) screens all pairs cheaply so the search reaches
H = 32: a coverage screen ranks the broken pair about nine times better than a Newton screen.
[ACTIVE_SCREEN.md](ACTIVE_SCREEN.md) (roadmap #24) runs the targeted queries at H = 32 on
screened candidates: they find the broken pair in 12 of the 13 cases whose candidates include it.
[CANDIDATE_RECALL.md](CANDIDATE_RECALL.md) (roadmap #25) keeps more candidates: the broken pair is found in
18 of 25 width-32 cases, and every visible fault in 21.
[JOINT_SCREEN.md](JOINT_SCREEN.md) (roadmap #26) screens every pair with a 13 × 13 joint fit: with 48 targeted
queries it finds every visible fault in 25 of 25 width-32 cases.
[THREE_FAULTS.md](THREE_FAULTS.md) (roadmap #27) breaks three entries at H = 16: the pair screen still finds
two of them, but practice data alone picks all three in only 5 of 25 cases.
[TRIPLE_QUERIES.md](TRIPLE_QUERIES.md) (roadmap #28) adds targeted examples for three faults: all three
found rises from 5 to 10 of 25, and the grid fit, not the data, limits the rest.
[WARM_TRIPLES.md](WARM_TRIPLES.md) (roadmap #29) fits each triple from its last round's values: the true
triple keeps a perfect fit, and all three faults are found in 12 of 25 cases.
[TRIPLE_RECALL.md](TRIPLE_RECALL.md) (roadmap #30a) ranks three-fault suspects: refinement helps, and the
coarse grid works partly because one of its points sits next to the corruption size.
[RANDOM_SIZES.md](RANDOM_SIZES.md) (roadmap #31) re-tests the pair and triple screens with bug sizes their
grids cannot match: recall drops, so the earlier headline numbers were partly a grid coincidence.
[DENSE_SCREEN.md](DENSE_SCREEN.md) (roadmap #32) makes the screens denser: the pair pipeline finds every visible
fault in 23 of 25 random-size cases, and the triple ranking improves only to 14.
[CONTINUOUS_TRIPLES.md](CONTINUOUS_TRIPLES.md) (roadmap #33) fits every three-fault suspect continuously: the true
triple fits practice perfectly, but so do a median 160 others.
[TRIPLE_COMMITTEE.md](TRIPLE_COMMITTEE.md) (roadmap #34) asks the teacher where 200 suspects disagree: all three
faults found in 16 of 25 random-size cases, every visible fault in 19.
[BOUNDARY_QUERIES.md](BOUNDARY_QUERIES.md) (roadmap #35) draws those queries near the suspects' decision boundaries:
17 of 25 all three, 19 every visible fault, about level with uniform queries.
[LOOKALIKES.md](LOOKALIKES.md) (roadmap #36) checks the repaired networks' behaviour on a million inputs: 22 of 25
match the original network on all but 0.1%, though only 16 to 17 edit the exact broken entries.
[BEHAVIOUR_SCORES.md](BEHAVIOUR_SCORES.md) (roadmap #37) probes every round's winner: three-fault repairs behave
like the original much sooner than their entries say (16 against 9 of 25 after 16 labels), while four two-fault
repairs edit the exact weights and still miss by 0.13% to 0.17%, so two-fault behaviour ends at 20 of 25, not 23.
[VALUE_QUERIES.md](VALUE_QUERIES.md) (roadmap #38) fixes those misses: a finer local fit plus labels aimed at
the winner's values brings 24 of 25 two-fault repairs under 0.1% and 20 under 0.01%, against 20 and 9.
[SWITCH_QUERIES.md](SWITCH_QUERIES.md) (roadmap #39) switches to those value queries once the committee agrees:
level on behaviour (24 and 19) with every visible fault back at 23, but it reaches the 0.01% line later.
[TRIPLE_VALUES.md](TRIPLE_VALUES.md) (roadmap #40) gives three-fault repairs the same value queries: 24 of 25 under 0.1%
and 19 under 0.01% (#37: 22 and 16), but every visible fault drops from 19 to 17, the same trade as for two faults.
[SWITCH_TRIPLES.md](SWITCH_TRIPLES.md) (roadmap #41) tries #39's switch rule on three faults: the committee almost
never runs out of splits, so only 101 of 1,600 labels become value queries and the result is #37's exactly.
[SWITCH_LINE.md](SWITCH_LINE.md) (roadmap #42) moves the trigger to the 0.1% line (fewer than 20 pool inputs split
the committee): it fires in 23 cases, but under 0.01% falls from 16 to 13 and under 0.1% stays at 22, so dropping
every committee label at the line hurts.
[HALF_LINE.md](HALF_LINE.md) (roadmap #43) keeps half the labels on committee splits past the same line: #42's loss
is undone but nothing is gained (22 and 16, #37's cases exactly), because the cases #40 fixes reach the line late or never.
[EARLY_VALUES.md](EARLY_VALUES.md) (roadmap #44) reverses that: #40's mix in the first four rounds, then committee splits only. It keeps both gains: 24 of 25 under 0.1% as in #40, and 19 every visible fault as in #37, with half #40's value queries.
[SWITCH_POINT.md](SWITCH_POINT.md) (roadmap #45) moves the switch: two early rounds (200 value queries) is the best three-fault arm so far, 24 of 25 under 0.1% and 20 under 0.01%, with #37's weights exactly; six early rounds is worse on every count.

[FRESH_SWITCH.md](FRESH_SWITCH.md) (roadmap #46) checks #45 on 25 fresh cases: 2 early rounds no longer beat 4 (15 and 16 under 0.01%), but over all 50 cases early value labels put 48 under 0.1% against 45 without them, with the same weights; one early round loses seed 17, and the committee misses the true triple in 8 fresh cases, including seed 49, the one clear failure.

[SCREEN_MISSES.md](SCREEN_MISSES.md) (roadmap #47) finds where the committee loses the true triple: 9 of 12 times in a tie at the cut to 200, 3 times because none of its pairs is among the best 200; extending the best 1,000 pairs fixes both clear failures (seeds 14 and 49 end under 0.01% on the exact weights).

[WIDE_SCREEN.md](WIDE_SCREEN.md) (roadmap #48) runs the 1,000-pair screen on all 50 three-fault cases: 50 of 50 under 0.1% with nothing lost at that line, at about 4 times the cost; widening only when no committee member fits every practice label fires on exactly the two failures and gets the same 50 of 50 at 1.1 times the cost.

[ADAPTIVE_WIDEN.md](ADAPTIVE_WIDEN.md) (roadmap #49) runs #48's widening flag on 50 fresh three-fault cases: it never fires, and the five cases that miss 0.1% all lack the true triple; flagging a case when its finished repair still misses a label catches all seven failures across seeds 0 to 99.

[END_FLAG.md](END_FLAG.md) (roadmap #50) reruns the 16 fresh cases the end flag picks with 1,000 pairs: it fixes two of the five fresh failures (61 and 83, both exact) and breaks nothing at 0.1%, for 97 of 100 three-fault cases under 0.1% at 1.9 times the cost of 200 pairs; seeds 63, 76 and 96 still lack the true triple.

[RANK_MISSES.md](RANK_MISSES.md) (roadmap #51) finds where the nine fresh 1,000-pair committees lose the true triple: all nine in the cut to 200 among hundreds of perfect practice fits, none in the pair screen, so a larger committee, not a wider screen, is the fix to try.

[BIG_COMMITTEE.md](BIG_COMMITTEE.md) (roadmap #52) keeps 1,000 triples instead of 200 on the 16 fresh end-flag cases: it fixes 4 of the 5 fresh failures (3 on the exact weights) at 40% of the wider screen's cost per case, and trying it first, then the wider screen only where the end flag is still up, puts all 50 fresh cases under 0.1%.

[TWO_STEP.md](TWO_STEP.md) (roadmap #53) scores the two-step rule (larger committee when the repair misses a label, then the wider screen if it still does) on all 100 three-fault cases: 100 of 100 under 0.1%, 75 under 0.01% and 69 exact at 1.6 times the cost of the plain run.

[UNSEEN.md](UNSEEN.md) (roadmap #54) runs the two-step rule end to end with `early_values.py --two-step` on 50 unseen three-fault cases: all 50 under 0.1% (46 for the plain run), 35 under 0.01% and 27 exact at 1.8 times the cost; three cases still miss a label because their true triple never reaches the committee.

[UNSEEN_MISSES.md](UNSEEN_MISSES.md) (roadmap #55) finds where the two-step rule still loses the true triple on the three unseen cases it cannot fully repair: never in the pair screen once 1,000 pairs are extended, always in the cut among hundreds to thousands of perfect practice fits.

[BOTH_WIDE.md](BOTH_WIDE.md) (roadmap #56) reruns the three still-flagged unseen cases with 1,000 pairs and 1,000 triples together: it fixes seed 149 on the exact weights, as #55 predicted, and not the other two, for 12% more CPU; available as `early_values.py --three-step`.

[TIE_BREAKS.md](TIE_BREAKS.md) (roadmap #57) tries seven tie-breaks among perfect practice fits in the 12 cases where the true triple is lost: preferring the smallest largest single edit keeps 5 of them in the committee, the current order none, and no tie-break keeps all 12.

[LARGEST_EDIT.md](LARGEST_EDIT.md) (roadmap #58) runs #57's best tie-break inside the whole two-step repair on seeds 100 to 149: it ends 49 of 50 under 0.1% against 50 for the coarse order, loses the true triple more often than it saves it and costs 23% more CPU, so the coarse order stays the default.

[LATE_EDIT.md](LATE_EDIT.md) (roadmap #59) uses the largest-edit order only in the 1,000-pair rerun: #51's nine cases never reach that stage, and on the six unseen cases that do it ends four closer and two further with none crossing 0.1%, so the tie-break line is closed.

[WIDTH32.md](WIDTH32.md) (roadmap #60) runs the two-step repair unchanged at hidden width 32 on 12 fresh three-fault cases: all 12 end under 0.1% and 8 under 0.01%, about the same shares as width 16, but only 5 end exact, each case costs 5.2 times as much CPU, and the 1,000-pair third stage makes all four cases it reaches worse.

[FINAL_STAGE.md](FINAL_STAGE.md) (roadmap #61) rescores the saved per-stage results: keeping the third stage only when it fits every practice label never loses a line in three runs, moves width 32 from 8 to 10 of 12 under 0.01% and cuts the width-16 worst case from 0.069% to 0.047%, at no extra cost; it was picked after seeing the data, so fresh seeds come next.

[THIRD_RULE.md](THIRD_RULE.md) (roadmap #62) runs #61's rule on 50 fresh width-16 cases: it drops five imperfect third stages and four of them end further, one crossing 0.1%, so `last` stays the default; a perfect third-stage fit is still the sign of an exact repair, and 17 of the 33 cases over 0.01% across 100 cases already edit the right three weights.

[REFINE_VALUES.md](REFINE_VALUES.md) (roadmap #63) replays the 58 cases whose repair edits exactly the right three weights and moves their values to the middle of the region that fits the same labels: it is a little closer on average but moves the 0.01% line by 0 to 2 cases, all the gain coming where the labels leave room (14 cases, 11 closer in every walk).

[MIDDLE_STEP.md](MIDDLE_STEP.md) (roadmap #64) runs the middle-of-the-region step inside the repair on 50 fresh width-16 cases: the width threshold fixed beforehand changes 8 cases and takes under 0.01% from 36 to 37, while the middle whenever it fits as many labels takes it to 39 with none moving over, so #63's width bands did not hold but its unconditional arm did.

[REGION_LABELS.md](REGION_LABELS.md) (roadmap #65) asks 8 more labels after the repair on 50 fresh width-16 cases, either where the committee splits or where the middle step's fitting values split: both take under 0.01% from 36 to 41 once the middle is taken, but the region-split labels end closer than the repair in 42 of 50 with none moving over, and neither changes the 30 exact repairs, since none of the 7 inexact cases left over 0.01% has the true triple in its committee.

[REBUILD_COMMITTEE.md](REBUILD_COMMITTEE.md) (roadmap #66) reruns the pair screen and the committee on each case's 1,096 final labels for #65's 50 cases: the true triple is now in 46 committees instead of 35, but the repair still edits exactly the right three weights in 30, since in 12 of the 16 misses the true triple ties the winner on every label and loses the tie-break by fit cost; the committee's own first member is exact in 34, and the 0.01% line moves only from 41 to 42.

[TIE_BREAK_FINAL.md](TIE_BREAK_FINAL.md) (roadmap #67) picks the winner six ways from #66's rebuilt committees: breaking ties among the members that fit every label by the coarse fit instead of the total edit finds the exact three broken weights in 34 cases instead of 30, and 35 after refitting the near-best members, but the 0.01% line stays at 42 to 43, since most tied members are near-identical networks; the true values fit every label in all 50 cases, so the misses come from the fit and the tie-break, not the labels.

[TIED_LABELS.md](TIED_LABELS.md) (roadmap #68) asks up to 3 rounds of 8 labels where the members tied on every label disagree, on #67's 50 cases: 41 of the 44 tied cases are still tied afterwards, since in 87 of 127 rounds no input in a pool of 20,000 splits the tied members; where they do split, labels help (seed 255 goes from 0.062% to the exact triple), so the 0.01% line moves from 42 to 44 and exact repairs from 35 to 37; 28 of the 41 remaining ties are the size the output-layer mirror symmetry predicts.

[TIED_AGREE.md](TIED_AGREE.md) (roadmap #69) keeps every member that #68 left tied on every label, in 41 cases, and compares each with the winner: all 672 are within 0.009% of it on the 1M-point probe, so no tie costs anything; 28 cases tie only the output-layer mirrors, the three largest ties come from cases where one fault is silent and leaves a free third slot, and the other 9 tie a few genuinely different three-weight repairs.

[FIX_SIX.md](FIX_SIX.md) (roadmap #70) tries three fixes on the 6 cases #68 left over 0.01%: a bigger committee rescues 258 (the true triple enters and wins), 20 restarts rescue 259, and 64 boundary labels rescue 269, so each fix brings one case under the line and none makes any case worse; 260 is limited by its labels (even the true triple reaches only 0.065%), 261's true triple cannot fit the labels, and 296's true triple would work but ranks outside even a doubled committee.

[PICK_FIXES.md](PICK_FIXES.md) (roadmap #71) runs #70's three fixes on all 50 of #67's cases and chooses among #68's pick and the fixes from labels alone: 64 uniform labels never separate the candidates, but up to 32 labels where the candidates disagree (a median of 3) choose as well as the probe, putting 47 of 50 under 0.01% (258, 259 and 269 gained, none lost) and 39 exact; the one case it makes worse, 272, stays at 0.0013%.

[FRESH_REPAIR.md](FRESH_REPAIR.md) (roadmap #72) runs the whole repair, unchanged from #71, on 50 fresh width-16 three-fault cases (seeds 300 to 349): the disagree chooser's pick, fixed in advance as the result, puts 49 of 50 under 0.01% and all 50 under 0.1% (against 41 for #65's repair alone and 46 for #68's pick), matching the probe's own best choice in every case, with a median of 3 extra labels; the three rescues over #68 all come from boundary labels, and the run cost 37.6 recorded core-hours.

[TIME_REPAIR.md](TIME_REPAIR.md) (roadmap #73) times every part and step of #71's stage on a 35-case sample from the same labels: one step, the margin refine (Nelder-Mead on the label margins), takes 87% of it in every case, while the fixes and the chooser cost 0.01 core-hours together; the rerun reproduces the chooser's pick in 35 of 35.

[CUT_REFINE.md](CUT_REFINE.md) (roadmap #74) swaps cheaper margin refines into #71's stage on #73's 35 cases: a loss that recomputes only the hidden units an edit touches cuts the stage 1.40x with the same pick, network and labels in all 35; adding looser stopping cuts it 1.79x and keeps 34 under 0.01% and 28 exact, though 3 picks change.

[TIME_65.md](TIME_65.md) (roadmap #75) reruns #65's stage on 8 of #73's cases with the original, columns and loose refines: the original reproduces #65's labels and repairs bit for bit and spends 86% of its time in the refine; columns cuts the stage 1.65x and loose 2.03x with 7 of 8 under 0.01% throughout, but the cheaper refines change the labels the stage ends with (columns in 4 of 8), so #71's picks on top still need checking (#76).

[END_TO_END.md](END_TO_END.md) (roadmap #76) runs #71's stage on the labels #75's cheaper refines saved and scores the whole repair on 8 cases: columns cuts it 1.54x with the original's pick in 8 of 8, loose 1.96x with all 8 under 0.01% but 3 picks changed; a batched Nelder-Mead that follows SciPy's steps exactly cuts #71's stage a further 1.39x over columns with the same picks.

[FRESH_CHEAP.md](FRESH_CHEAP.md) (roadmap #77) runs the whole repair on 12 fresh seeds as #72 ran it and with columns plus the batched refine in both stages: the cheap repair takes 6.24 core-hours against 13.67 (2.19x), keeps all 12 under 0.01% and makes the original's pick in 11 of 12, losing one exact repair.

[FRESH_50.md](FRESH_50.md) (roadmap #78) runs the cheap repair on 50 more fresh seeds: 20.88 core-hours against #72's 37.56 (1.80x), 48 of 50 under 0.01% (49 in #72) and 39 exact in both; the original repair, rerun on the two cases over the line, returns the same networks.

[KEEP_65.md](KEEP_65.md) (roadmap #79) adds #65's own repair as a fifth candidate in #71's chooser, on 100 fresh cases: with ties going to #71's candidates, 98 under 0.01% (97 without) and 79 exact (78), seed 414 fixed (0.069% to 0.0035%) and none taken over the line; ties going to #65's repair is worse.

[SAVED_65.md](SAVED_65.md) (roadmap #80) saves #65's own network edits and reruns the cheap repair on 50 fresh seeds: the saved repair matches #65's record in all 50 (the rebuild in 44, and the rebuild makes 508 worse), #71's chooser alone already has all 50 under 0.01%, and a two-round chooser keeps #79's 319 pick but loses small amounts in 9 of 150 cases.

[GATED_DUEL.md](GATED_DUEL.md) (roadmap #81) runs the second-round duel only when it has at least 32 split inputs: on #79's 100 cases 3 closer and none further (#79's one round: 4 and 1), 98 under 0.01% and 79 exact, fewer labels than #79; on #80's 50 no pick changes.

[VALUE_LABELS.md](VALUE_LABELS.md) (roadmap #82) finds why 331 and 400 stayed over the line: their fitting region is a flat line the hit-and-run walk cannot travel. Taking the middle of the segment along the flat direction found from the labels' margin gradients puts all 150 of #79's and #80's cases under 0.01%; #65-style region labels do not.

[FRESH_LINE.md](FRESH_LINE.md) (roadmap #83) runs the whole repair with #81's gated duel and #82's flat-line step on 50 fresh seeds: 48 of 50 under 0.01% and 38 exact, the same as without them. Of the two cases over the line, 641 has a triple that fits every label and no candidate tries it.

[KEEP_ONE.md](KEEP_ONE.md) (roadmap #84) searches every triple that keeps one of the pick's weights in the 27 of 200 cases whose pick misses a label: 641 gets its true broken weights and goes under the line, 5 cases get closer and none worse. In 10 cases the fitter misses a perfect fit that exists.

[PERFECT_FIT.md](PERFECT_FIT.md) (roadmap #85) adds a linear-program step after the fitter: the true broken weights become a perfect fit in all 27 of #84's searched cases (from 17), 3 more cases get closer and none worse, and no outcome changes.

[FLAT_STACK.md](FLAT_STACK.md) (roadmap #86) runs #82's flat-line step on every case after #85: 199 of 200 under 0.01% and 158 exact. 611, the last over the line, has a wrong third weight; its true weights sit on a flat line of perfect fits.
