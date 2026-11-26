# Roadmap

**Written:** November 10, 2026, after the audit in the [lab notebook](LAB_NOTEBOOK.md).

The experiments are ranked by **how likely each is to produce a finding worth a blog
post**: a clear result whichever way it comes out, on a question people outside this
repo care about, at a cost the lab can actually pay. Each entry states a falsifiable
hypothesis, a predeclared method, the figure it would produce, its cost, and its
prerequisites.

The lab's usual rules apply to all of them: declare trials before running, report
every slot (failures included), keep held-out data out of selection, and label
scripted controls separately from model results.

| Rank | Experiment | Compute | New code | Post likelihood |
|---:|---|---|---|---|
| 1 | Is p50c a math edit or a stop-thinking edit? | GPU, the full run already planned | Small (post-hoc analysis) | Very high |
| 2 | Anatomy of a hand-written Breakout network | CPU, minutes | Small (pilot exists) | High |
| 3 | The harness is the benchmark | CLI allowance + about $5 API | Small (one transport revision) | High |
| 4 | How reliable is writing a policy by hand? | Account allowance, CPU eval | None | High |
| 5 | Why editors overshoot | Shares #3's runs + one prompt arm | Small | Medium–high |
| 6 | Dose–response of a published LLM edit | GPU, ~10 GPU-hours | Moderate (scaled BF16 deltas) | Medium–high |
| 7 | Locate-and-repair: a difficulty scaling curve | CPU + API | Moderate (new task generator) | Medium |
| 8 | Would a better guess keep the Breakout network alive longer? | CPU, about 15 minutes | Small | Medium–high |

---

## 1. Is p50c a math edit or a stop-thinking edit?

> **Status, November 14: first result in.** This uses upstream's released full-run
> records, not a new GPU run ([write-up](../experiments/p50c-termination/README.md)).
> All of p50c's +40-point gain comes from problems where the original model was stuck in
> a repetition loop. On the 12 problems both models finished, the edit adds nothing. In
> 78 of 81 rescued problems, the original never wrote the right answer before looping.
> The hypothesis holds, with one correction: the original was not failing to stop
> after finding the answer; the loop started before it got there. Still to do: token-budget
> curves (tokenizer download blocked in that session) and the lab's own paired run.

**Hypothesis.** Most of the published p50c edit's MATH-500 gain comes from the edited
model *finishing* within the 32,768-token budget, not from better reasoning. Among
samples that finish, the accuracy difference between arms is small (at most ⅓ of the
overall difference).

**Why it could be a post.** The pilot already points this way, though it is anecdotal:
the original model hit the ceiling on 8/8 samples, p50c on 1/8, and the original's only
correct sample was a capped one. "A weight edit that makes a small model stop thinking"
is a sharper, more interesting claim than "+X% on MATH". Either outcome is reportable.

**Method.**
1. Finish the predeclared full paired run: 500 problems × 4 samples × 2 arms at 32,768
   tokens, using the code that started it (see notebook finding 3).
2. Build **budget curves after the fact**. Each sample's first *B* tokens are what a
   run with ceiling *B* would have produced, because seeds are per request and
   decoding is autoregressive. So for B ∈ {1k, 2k, 4k, 8k, 16k, 32k}, truncate the
   retained token IDs, decode, and regrade with the same extractor. Check this
   equivalence on 10 problems by actually rerunning them at *B* = 4k and comparing
   token IDs; if it fails, report that and run real reduced-ceiling evaluations.
3. Decompose accuracy = P(finish) · P(correct | finish) + P(correct and capped), per
   arm, with the problem-clustered bootstrap the runner already uses.
4. Report three scorings side by side: as published (capped samples graded); capped
   samples counted wrong; finished samples only.

**Plot.** Accuracy (solid) and finish rate (dashed) against token budget on a log₂
axis, one colour per arm, with 95% bootstrap bands. Beside it, a stacked bar per arm:
correct-finished / wrong-finished / correct-capped / wrong-capped.

**Cost / prerequisites.** The full run is already planned (4,000 generations, up to
131M tokens). The post-hoc analysis is CPU only. One small runner addition: a
`budget-curve` command that regrades truncated prefixes.

---

## 2. Anatomy of a hand-written Breakout network

> **Status, November 16: done for the released network**
> ([write-up](../experiments/breakout-anatomy/README.md)). The author's description is
> accurate: correct direction context on 99.4% of states, exact descending landing
> targets. The network cannot see bricks, so ascending targets are right only 42% of the
> time. With sticky actions off it never dies (1,000/1,000 episodes at the ceiling,
> 834 each), so every death is injected noise. Half of the second layer is nearly
> redundant; 1–3% weight noise breaks it. Still blocked: the fresh-network comparison
> (needs `fresh.npz`). Follow-up: #8.

**Hypothesis.** The published Experiment 5 answer is a small, readable program and not
a distributed representation. It has no redundancy: removing any of its active units
destroys play. The fresh agent-written network implements a *different* program with
similar survival.

**Why it could be a post.** Today's audit found that the released 600-32-32-6 network
has only **815 nonzero parameters out of 20,486**. It uses 6 first-layer units, 8
second-layer units and 2 actions, with gate-like constants (±1000, ±200, 30) and only
61 distinct values. It still scores about 20× the published PPO baselines. Reverse-
engineering an LLM-written neural program is novel, cheap and visual.

**Method.**
1. Structure: list active units, receptive fields over the 10×10×6 input channels,
   and the logic implemented by layer 2 (the ±1000 pairs look like sign detectors).
   Render the receptive fields as images.
2. Ablations, all at the published protocol: zero each first- and second-layer unit;
   quantize all weights to the nearest of the 61 values; drop the k smallest-magnitude
   first-layer weights. A [pilot](../experiments/breakout-circuit/README.md) of the
   single-unit first-layer ablations ran today. Zeroing any one of the 6 units drops
   the mean over 200 episodes from 481.7 to between 0.5 and 8.4, so the hypothesis of
   no redundancy already holds at layer 1.
3. Compare with the fresh network (1,076 nonzero weights). This needs `fresh.npz`
   from the original workstation, committed or shared as evidence.
4. Use survival curves from [breakout-ceiling](../experiments/breakout-ceiling/README.md)
   as the main metric, since score tracks survival (r = 0.992).

**Plot.** (a) Receptive-field images for the 6 first-layer units. (b) Survival curves
for intact versus each ablation. (c) Released and fresh networks drawn as circuit
diagrams with the same layout.

**Cost / prerequisites.** CPU only. A full 1,000-episode evaluation takes about 1 minute
on 3 cores. `experiments/breakout-circuit/ablate.py` exists. The fresh-network
comparison is blocked until its checkpoint is available.

---

## 3. The harness is the benchmark

**Hypothesis.** The recorded Claude CLI failure (`error_max_structured_output_retries`)
comes from the transport settings (`MAX_STRUCTURED_OUTPUT_RETRIES=0`, `--max-turns 2`,
a 9-branch strict schema), not from the model. Allowing normal retries, or using the
native Anthropic tool API, gives Claude a completion rate comparable to Codex's. More
generally, for a single model the outcome spread across transports is as large as the
spread across models within a transport.

**Why it could be a post.** "Your agent benchmark is measuring your harness" has broad
appeal. The repository already records the confound, and the fix is cheap to test.

**Method.** Predeclare 20 matched cases (4 families × 5 seeds) and these editor ×
transport arms:

- Claude via CLI v2 (unchanged, `retries=0`);
- Claude via a new CLI v3 with retries at the client default, all else equal;
- Claude via native API tools;
- Codex via CLI v2.

Optionally add one Fireworks model via native tools. Inspect the retained
stdout/stderr of the existing failed pilot first. Keep every slot. Classify each
outcome as execution failure, no selection, frozen but failed, or success.

**Plot.** One stacked horizontal bar per editor × transport, split into the four
outcome categories, ordered by success rate, with each arm's total cost annotated.

**Cost / prerequisites.** Roughly 60 CLI decision rounds of account allowance, plus
an estimated $3–5 of Anthropic API under the existing $10 guard. Needs a
`cli-structured-actions-v3` transport revision. v2 stays unchanged for comparability.
Needs explicit authorization for paid and CLI calls (`--allow-paid-api`, `--allow-cli`).

---

## 4. How reliable is writing a policy by hand?

**Hypothesis.** The one fresh Breakout success (511.5) is a single draw from a wide
distribution. Across k = 8 predeclared fresh attempts, at least one scores below 100,
and the median is below the released network's 502.9.

**Why it could be a post.** The upstream claim and the local reproduction each rest on
one response. A predeclared distribution, failures included, is the obvious missing
measurement. The result is interesting either way: "8/8 worked" is a strong result too.

**Method.** k = 8 fresh tool-free conversations using the same prompt, model and effort
as the recorded attempt, with no retries, all reported. Freeze each network before
evaluation. Evaluate each at the full protocol (1,000 episodes, 10,000 steps), then run
the survival and anatomy analyses from #2. Record the token usage and latency of every
attempt.

**Plot.** A strip plot of per-attempt mean ± SE, with horizontal lines for the
published value, the local replay and PPO-500M. Inset: survival curves for all
attempts.

**Cost / prerequisites.** About 8 × 17k output tokens at xhigh effort (about 10 minutes
each) on Codex account allowance; CPU evaluation takes about 1 minute per attempt.
Needs explicit approval for fresh generation (`docs/BREAKOUT.md`). A fresh conversation
cannot rule out training-data exposure, so state that caveat in the post.

---

## 5. Why editors overshoot

**Hypothesis.** Model editors place their edit at the computed correct value or past
it, so they systematically overshoot. This fails the zero-regression rule even though
repair is perfect: every recorded Codex attempt had 100% repair plus 1–6 regressions.
Adding one neutral sentence about the asymmetry raises success substantially.

**Why it could be a post.** It connects a clean geometric fact with an observable model
behaviour. From [task geometry](../experiments/task-geometry/README.md): the smallest
practice-perfect edit succeeds 73–83% of the time, the practice-window midpoint only
20–38%, and overshoot tolerance is 2–7× tighter than undershoot.

**Method.** Reuse #3's matched cases. For every frozen candidate, record its value as a
signed position relative to the case's success window. Add one predeclared prompt arm
per editor: "Regressions are not allowed; prefer the smallest change that fixes the
practice errors." Compare success with and without it, paired by case. Keep the
geometry strategies as reference lines.

**Plot.** A strip plot of each frozen value's normalized position, with 0 at the
correct value and the success window shaded, one row per editor × prompt arm. A
paired success-count table sits beside it.

**Cost / prerequisites.** One extra arm on #3's cases (about half of #3's cost again).
Needs the per-case window, computable with `experiments/task-geometry/analyze.py`
logic from the campaign's private held-out draw. This step must run after
finalization, as an operator-side analysis.

---

## 6. Dose–response of a published LLM edit

**Hypothesis.** The p50c effect is monotone in the edit scale α and lives mostly in one
of its two tensors (layer 15 or layer 19). Finish rate responds before accuracy does.

**Why it could be a post.** A dose–response curve turns a binary "edited versus
original" into a mechanism-shaped result, and it is rarely shown for published weight
edits.

**Method.** Candidates are base + α·(p50c − base) with α ∈ {0, 0.5, 1, 1.5}, applied to
layer 15 only, layer 19 only, or both. That gives 10 distinct checkpoints, with BF16
rounding recorded and verified like the existing reconstruction. Evaluate each on a
fixed, predeclared 100-problem subset × 4 samples at an 8,192-token ceiling. The
subset is chosen before any results; the full MATH-500 run stays the headline
measurement.

**Plot.** Two panels, accuracy and finish rate against α, with one line per variant
(L15, L19, both) and bootstrap bands.

**Cost / prerequisites.** About 10 GPU-hours on the RTX 3090 setup. Needs a
scaled-delta export (a small extension of `reconstruct`) and #1's runner state. PR #4's
placement tooling would let two cards run arms in parallel.

---

## 7. Locate-and-repair: a difficulty scaling curve

> **Status, November 16: offline half done** ([write-up](../experiments/locate-repair/README.md)).
> Model-free strategies only. The collapse hypothesis fails for *passing*: an editor not
> told the location still passes about half the time up to 642 parameters. It holds for
> *finding*: that editor edits the corrupted entry in 3% of cases at 642 parameters, and
> its passes undo a median 75–79% of the damage. Still to do: model editors (needs paid
> API or CLI allowance and approval).

**Hypothesis.** Benchmark success collapses once the editor must *find* the corrupted
entry instead of being told it. The drop grows with the log of the parameter count,
and random search collapses faster than model editors.

**Why it could be a post.** Today's tasks give the location away (`describe` names the
entry), so they measure arithmetic plus tool discipline. A scaling curve is the
standard figure for showing where a capability breaks.

**Method.** Add a new family based on `hidden`, with width H ∈ {2, 8, 32, 128}. One
corrupted entry anywhere in the weights, any single entry editable, the same success
rule. Run two conditions per editor: location given versus location hidden. Controls:
feasible; random (random entry and value); and a "gradient-free sensitivity probe"
heuristic declared in advance. Use the same budgets as the pilot config.

**Plot.** Success rate against parameter count (log x), solid lines for location
hidden, dashed for location given, one colour per editor, with the random control
in grey.

**Cost / prerequisites.** A new generator and controller support for a larger editable
set (about a day of work). API cost scales with editors × widths × seeds; start with
2 editors × 4 widths × 5 seeds.

---

## 8. Would a better guess keep the Breakout network alive longer?

*Added November 16 after #2.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/TARGETS.md)).
> The hypothesis fails. Perfect ascending prediction scores 480.6 versus 502.9 for the
> published guesses; a more accurate brick-blind table scores 294.1. 97 of 100 deaths
> are last-moment arrivals. A pure oracle policy survives far longer but scores 231.7,
> because the network's catch choices aim the ball at bricks.

**Hypothesis.** The released network's ascending predictions are wrong by about two
columns half the time because it cannot see bricks. A better brick-blind guess
increases survival under sticky actions, and perfect ascending prediction increases it
substantially.

**Method.** Replace only the ascending target code (units 2 and 3) with: follow the
ball's column; always center; landing with no bricks; the best brick-blind lookup table
fitted on held-out episodes 5000–5049; and an oracle that feeds the true landing column.
Add a pure oracle policy with no network. Evaluate each at the full protocol.

**Plot.** Mean score against mean ascending-target error, one point per variant.

**Cost.** CPU, about 15 minutes. `experiments/breakout-anatomy/targets.py`.

---

## 9. Why do the author's rough guesses beat accurate ones?

*Added November 16 after #8.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/HEDGE.md)).
> The published guesses are the earliest-bounce landings (all 180 cells), and a table
> rebuilt from physics alone scores 493.0 of 502.9. The deadline hypothesis fails as
> modelled. Fitting for exact hits beats fitting for distance (398 vs 294) but leaves a
> 100-point gap that no on-play accuracy measure explains.

**Hypothesis.** The author's guesses assume the earliest possible brick bounce, so they
hedge the deadline rather than the average error.

**Method.** Rebuild the full-wall landing table from physics; fit deadline-aware and
most-likely-landing tables on held-out play; score each at the published protocol.

**Cost.** CPU, about 15 minutes. `experiments/breakout-anatomy/hedge.py`.

**Next.** Closed-loop fit: refit a table on its own play and repeat, to test whether the
gap is a distribution-shift effect.

---

## 10. How many examples does it take to find the broken weight?

*Added November 16 after #7.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/PRACTICE.md)).
> From 32 to 1,024 practice examples, the not-told editor's pass rate rises to 85–92% and
> its repairs become near-complete, but at 642 parameters it edits the corrupted entry
> only 34% of the time, and most passes still change another entry.

**Hypothesis.** #7's localization collapse is a data shortage: with enough practice
examples, only the corrupted entry admits a practice-perfect edit.

**Method.** Keep each #7 case's corruption and held-out test fixed; grow practice from
16 to 512 examples per stratum; rerun the min-change editor.

**Cost.** CPU, about 40 minutes. `experiments/locate-repair/practice_size.py`.

**Next.** Corruptions no other single entry can absorb, to make localization well posed.

---

## 11. Is the accuracy gap a distribution-shift effect?

*Added November 16 after #9.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/CLOSEDLOOP.md)).
> No. Refitting tables on their own play never scores above 340, and one refit of the
> author's table drops it from 497 to 297. Accuracy-fitting itself costs the score.

**Hypothesis.** Tables fitted on the published network's play lose because installing them
changes the states that occur; refitting on their own play closes the gap.

**Method.** Iterated refits (mean distance) on each table's own play from three starts.

**Cost.** CPU, about an hour. `experiments/breakout-anatomy/closedloop.py`.

**Next.** Swap accurate targets into some rows only, to find where the author's table wins.

---

## 12. Where on the board does the author's table win?

*Added November 16 after #11.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/ROWSWAP.md)).
> Rows 3–5 (the bottom of the brick wall and the two rows below) carry almost the whole gap:
> accurate guesses there alone score 284, the author's guesses there alone lift the accurate
> table from 297 to 448.

**Method.** Swap ascending guesses one ball row at a time, and in groups, between the
author's table and #8's accurate table; 300 episodes each.

**Cost.** CPU, about 45 minutes. `experiments/breakout-anatomy/rowswap.py`.

**Next.** Test the mechanism directly: for deaths under the accurate table, record the row
where the ball last ascended and how many steps it had left.

---

## 13. What kills the accurate table?

*Added November 16 after #12.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/DEATHS.md)).
> About 70% of deaths follow a bounce off the bottom row of the wall. On those flights the
> accurate table starts half a column further from the catch and dies 2.5 times as often.

**Method.** Trace every falling flight (turn-down row, steps left, paddle distance, death)
under the published and accurate tables; 300 episodes each.

**Cost.** CPU, about 15 minutes. `experiments/breakout-anatomy/deaths.py`.

**Next.** Record paddle velocity at the turn to explain the row-3 result.

---

## 14. Which weights stand in for the broken one?

*Added November 16 after #10.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/SUBSTITUTES.md)).
> Every output-weight fault has an exact twin, the same unit's weight into the other logit,
> which no behavioural data can tell apart. Counting it as found raises H = 32 localization
> from 46% to 61%. Other stand-ins sit in units pointing the same way (cosine > 0.9 for about
> half, against 15–16% of units).

**Method.** For every entry, the smallest practice-perfect edit on 1,024 examples; class the
near-complete passing ones against the corrupted entry; H = 32 and 128, 50 seeds each.

**Cost.** CPU, about 25 minutes. `experiments/locate-repair/substitutes.py`.

**Next.** Score localization up to the mirror symmetry in the benchmark itself, and test a
corruption that no single other entry can absorb (for example, two corrupted entries).

---

## 15. Does paddle motion explain the row-3 deaths?

*Added November 16 after #13.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/MOTION.md)).
> Yes. Under the accurate table 17% of row-3 flights start with the paddle moving away from the
> catch, and they carry 36 of 64 deaths. The author's paddle is often parked on the catch before
> the bounce; its deaths are mostly out of reach, the accurate table's mostly last-step overrides.

**Method.** Follow #13's flights step by step: motion at the turn, settled before arrival, late
overrides, out of reach; 300 episodes per table.

**Cost.** CPU, about 15 minutes. `experiments/breakout-anatomy/motion.py`.

**Next.** Hand-write a table that keeps the author's bet in rows 3–5 but corrects the rows where
it is out of reach, and check whether it beats 503.

---

## 16. Is the author's bet only a hedge against sticky actions?

*Added November 16 after #15.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/STICKY.md)).
> No. Without sticky actions nothing dies, yet the accurate table scores 569 against 834:
> fewer of its returns hit a brick. Its death rate is a steady two times the author's, and at
> the published setting slower scoring is over a third of the score gap.

**Method.** Three variants at six sticky probabilities, 200 episodes each; deaths and bricks per
10,000 steps.

**Cost.** CPU, about 25 minutes. `experiments/breakout-anatomy/sticky_sweep.py`.

**Next.** Measure the return angle at each catch to test whether the catch column explains the
scoring rate, then try a hand-written table that keeps the author's bet in rows 3–5 and fixes
its out-of-reach cells.

---

## 17. Why do the accurate table's returns miss bricks?

*Added November 16 after #16.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/RETURNS.md)).
> Corner catches. The paddle only decides whether the ball comes back up straight (under) or
> reversed (corner). The author's table rarely makes corner catches and its returns always
> break a brick; turning the accurate table's corner catches into under catches recovers the
> whole brick rate, but costs it the safe top-wall flights.

**Method.** Catch type, column, bricks and turn row for every catch, both tables, sticky 0 and
0.1 (200 episodes), plus a corners-as-under counterfactual.

**Cost.** CPU, about 30 minutes. `experiments/breakout-anatomy/returns.py`.

**Next.** Check whether the accurate table's paddle waits one column ahead of the ball before
its corner catches, then write a table that keeps the author's bet in rows 3–5 and fixes its
out-of-reach cells.

---

## 18. Does fitting the guesses to the under column fix the brick rate?

*Added November 16 after #17.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/UNDERFIT.md)).
> Yes, and it closes a third of the gap. #8's fit counted a corner catch as a hit. Refitted
> to the under column, the table scores 834 without sticky actions (like the author's) and
> 357 at the published setting, against 296 for #8's table and 482 for the author's.

**Method.** Refit every rising-ball cell to the nearest catch column, the under column or the
corner column; 200 episodes at sticky 0.1 plus the no-sticky cycle.

**Cost.** CPU, about 10 minutes. `experiments/breakout-anatomy/underfit.py`.

**Next.** Combine the two parts: the under-column table in rows the author's bet does not
cover, the author's guesses in rows 3–5, and the out-of-reach cells fixed.

---

## 19. Does a mix of the author's and the under-column guesses beat both?

*Added November 16 after #18.*

> **Status, November 16: done** ([write-up](../experiments/breakout-anatomy/COMBINED.md)).
> No. None of 24 row mixes beats the author's table; the one that seemed to (+8.5 ± 3.7)
> came out at +0.9 ± 1.8 on 1,000 fresh episodes. The author's rows 3–5 only work together:
> as a group they add 88 points to the under table, far more than their single effects.

**Method.** Every single-row swap in both directions plus groups; 200 episodes each, paired by
seed; the best swap checked on 1,000 held-out episodes.

**Cost.** CPU, about 25 minutes. `experiments/breakout-anatomy/combine.py`.

**Next.** The Breakout series now accounts for every gap it found. Candidates: write the
corner-catch and under-column results up as a follow-up post, or return to the editing and
repair experiments.

---

## 20. Can repairs find two broken weights at once?

*Added November 16 after #14.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/TWO_FAULTS.md)).
> One edit passes 10% of cases. Two greedy edits pass 30% at H = 8 and 56% at H = 32, but at
> H = 32 three in four passing repairs touch neither broken entry. Told the entries, a joint
> search over both values passes 96%; fitting them one at a time stalls at 64–80%.

**Method.** #7's cases with a second corrupted entry in another hidden unit; 1,024 practice
examples; H = 8 and 32, 50 seeds each; untold and told repairs, one-at-a-time and joint.

**Cost.** CPU, about 10 minutes. `experiments/locate-repair/two_faults.py`.

**Next.** Score localization in the benchmark itself (up to the mirror), and try a search
that ranks pairs of entries rather than one entry at a time, to see whether finding the
pair is possible from practice data at all.

---

## 21. Can a search over pairs find two broken weights?

*Added November 16 after #20.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/PAIR_SEARCH.md)).
> Only by searching every pair, and then only half the time. At H = 8 the all-pairs search
> passes 92% of cases but picks the broken pair in 56%; in most misses another pair fits
> the practice set equally well with a smaller change. Shortlists built from single-edit
> scores never contain both faults at H = 32.

**Method.** Joint fits of every pair in a shortlist (top 8 or 16 by single-edit score), or
of all pairs at H = 8; the 20 best re-fitted finely; #20's cases.

**Cost.** CPU, about 50 minutes. `experiments/locate-repair/pair_search.py`.

**Next.** Add examples where the tied candidate pairs disagree (active queries to the
teacher) and test whether a few dozen of them single out the broken pair.

---

## 22. Can a few targeted examples single out two broken weights?

*Added November 16 after #21.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/ACTIVE_QUERIES.md)).
> Mostly. 64 teacher-labelled examples, chosen where the candidate pairs disagree, raise the
> broken pair's win rate at H = 8 from 14 to 20 of 25 cases; 64 random ones change nothing.
> What stays tied is the exact output-layer mirror symmetry.

**Method.** #21's 20 candidate pairs per case; eight rounds of 8 targeted or random queries;
every candidate re-fitted each round; H = 8, 25 cases.

**Cost.** CPU, about 30 minutes. `experiments/locate-repair/active_queries.py`.

**Next.** Make candidate finding scale (pairs at H = 32 need a cheaper search than all
13,041 pairs), or turn queries into a localization score for the benchmark itself.

---

## 23. Can a cheap screen make pair search scale?

*Added November 16 after #22.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/PAIR_SCREEN.md)).
> The coverage screen does: at H = 32 it ranks the broken pair a median 338th of 13,041,
> about nine times better than a Newton-step screen, in about 6 seconds per case. Fits on
> its top 1,000 pairs still pick the broken pair in only 10% of cases, because many pairs
> fit the practice set perfectly.

**Method.** Newton and coverage screens over every pair; #21's joint fits on the coverage top
200 or 1,000; H = 8 and 32, 50 cases each.

**Cost.** CPU, about 50 minutes. `experiments/locate-repair/pair_screen.py`.

**Next.** Run #22's targeted queries on the screened candidates at H = 32, and re-fit more
than 20 coarse pairs finely.

---

## 24. Do targeted examples still work on a wider network?

*Added November 16 after #23.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/ACTIVE_SCREEN.md)).
> Yes. At H = 32, 64 targeted queries on the coverage screen's 40 best candidates raise
> found-the-pair from 5 to 12 of 25 cases, against 6 with random queries. When the broken
> pair is among the candidates, it wins in 12 of 13 cases. Candidate recall (13 of 25) is
> now the limit.

**Method.** #23's coverage screen and coarse fits pick 40 candidates per case; #22's eight
rounds of 8 targeted or random queries; H = 32, 25 cases.

**Cost.** CPU, about 45 minutes. `experiments/locate-repair/active_screen.py`.

## 25. Can candidate recall be raised cheaply?

*Added November 16 after #24.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/CANDIDATE_RECALL.md)).
> Yes. Coarse-fitting the screen's top 3,000 pairs and keeping 100 raises recall from 13 to 18
> of 25, and targeted queries then find the broken pair in all 18. Every visible fault is found
> in 21 of 25 cases; four cases have a silent fault that changes no practice label. The 4
> misses all ranked past 3,000 in the screen.

Half of #24's misses were pairs that passed the screen but lost at the coarse 61 × 61 fit;
the other half ranked past 1,000. Re-fit more screened pairs finely (or use a finer coarse
grid), measure how many of the 12 missing pairs that recovers and at what cost, then re-run
#24's targeted queries. A screen that scores pairs jointly, not from single edits, is the
second option.

**Cost.** CPU, about an hour.

## 26. A screen that ranks every visible pair in its top 3,000

*Added November 16 after #25.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/JOINT_SCREEN.md)).
> A 13 × 13 joint fit of every pair, ordered by accuracy, ranks the broken pair a median 5th of
> 13,041 (coverage: 327th). With its best 40 as candidates and 48 targeted queries, every visible
> fault is found in 25 of 25 cases, in about 54 seconds of screening per case.

All of #25's misses were cases where the coverage screen ranked the broken pair 3,500th to
10,175th of 13,041. Score pairs jointly instead of from single edits: for example a cheap
joint fit on a 9 × 9 grid over a subsample of the wrong examples, or a coverage score that
credits examples only the two edits together correct. Measure screen recall at 1,000 and 3,000
on the same 25 cases, then re-run #25's queries on any new hits.

**Cost.** CPU, about an hour.

## 27. Three faults, and a wider network

*Added November 16 after #26.*

> **Status, November 16: (b) done** ([write-up](../experiments/locate-repair/THREE_FAULTS.md)).
> At H = 16 with three faults, the joint pair screen still ranks a two-fault pair a median 21st
> of 3,321, and the true triple is among the extended triples in 24 of 25 cases. From practice
> data alone the winner holds all three faults in only 5, while the repair passes 17. (a), H = 128
> with two faults, is not run.

#26 locates two faults at H = 32. Two ways to push it: (a) H = 128 with two faults, about 200,000
pairs, to measure how the joint screen's time and rank scale; (b) three faults at H = 8 or 16,
screening pairs jointly and then extending the best pairs by a third entry, to see whether the
pair screen still ranks the right two entries near the top when a third fault is present.

**Cost.** CPU, one to two hours.

## 28. Targeted queries on triples

*Added November 16 after #27.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/TRIPLE_QUERIES.md)).
> 64 targeted examples raise all-three-found from 5 to 10 of 25. The true triple is among the 40
> candidates in 17 cases, and in 15 of those no candidate, the true triple included, fits all the
> examples: the grid fit, not the data, is the limit. The 9-point coarse grid was not tried.

#27's triple search found all three faults in 5 of 25 cases because several triples fit practice
perfectly. Run #22's targeted queries with the coarse top 40 triples as the committee, re-fitting
each finely every round, and score all three and visible faults. Also try a 9-point coarse grid,
which should keep more true triples in the top 40.

**Cost.** CPU, about an hour.

## 29. A fitter that keeps the true triple perfect

*Added November 16 after #28.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/WARM_TRIPLES.md)).
> With a warm-started continuous fit, the true triple fits every label in 16 of the 17 cases
> where it is a candidate, and all three faults are found in 12 of 25 (#28: 10). Candidate recall
> (8 cases) and 3 real ties are what is left.

#28's examples land where candidate repairs disagree, and there the fine grid fit of three values
falls short, so the true triple loses to look-alikes. Re-run #28 with each round's fit started from
the previous round's best values (local grids and coordinate passes around them, plus the full fit
as a fallback), so a triple that fit perfectly keeps fitting. Measure how often the true triple's
fit stays at 100%, and all-three and visible-found counts against #28.

**Cost.** CPU, about an hour.

## 30. More candidates and more examples for three faults

*Added November 16 after #29.*

> **Status, November 16: (a) done, (b) paused** ([write-up](../experiments/locate-repair/TRIPLE_RECALL.md)).
> The 9-point grid ranks the true triple worse (top 40 in 8 of 25, against 17), because #27's 7-point
> grid has a point at ±1.67 spreads, next to the ±1.6 corruption. Refining the coarse top 1,000 raises
> top-40 recall to 19. Part (b) waits on #32, since #31 showed the ranking is partly an artefact.

#29 leaves 8 of 25 cases without the true triple among the 40 candidates and 3 cases still tied
after 64 examples. Re-run #29 with (a) a 9-point coarse grid and the best 100 triples as
candidates, and (b) 16 rounds (128 examples). Measure candidate recall, all-three and
visible-found counts against #29, and whether the ties settle.

**Cost.** CPU, about two to three hours.

## 31. Do the screens only work because the grid matches the bug?

*Added November 16 after #30a.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/RANDOM_SIZES.md)). Yes, in part.
> With grid points moved off 1.6, #26's pair pipeline finds every visible fault in 21 of 25 cases
> (20 with random sizes), not 25. The coarse-7 triple ranking puts the true triple in the top 40 in
> 5 of 25 cases with random sizes, against 17. Pass rates hardly change.

Every grid from #20 on (61, 13, 7 and 200 points over ±2.5 spreads) has a point at or within 0.01 of
±1.6 spreads, the exact corruption size. #30a shows this inflates triple recall. Re-run #26's joint
pair screen with targeted queries at H = 32, and #27's coarse triple ranking at H = 16, with corruptions
drawn at random between 1.0 and 2.2 spreads per fault. Report recall and visible-found counts against
the fixed-size results. If they drop, fix the screens (offset grids, several grids, or refinement)
before going further.

**Cost.** CPU, about two hours.

## 32. A screen that does not depend on where its grid points fall

*Added November 16 after #31.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/DENSE_SCREEN.md)). A 25-point pair
> screen finds every visible fault in 23 of 25 random-size cases (13 points: 20). For triples an 11-point
> grid plus refinement reaches 14 of 25 in the top 40 (#31: 11); the coarse grid remains the limit.

#31 showed that the coarse pair and triple screens rank the true candidates well mainly when the
bug size sits next to a grid point. Replace each coarse score by the best of two or three grids
offset by a fraction of a step, or by #29's continuous margin fit started from the coarse optimum,
and re-run #31's random-size cases. Measure top-40 recall for pairs and triples and visible-found
counts after 64 targeted examples, against #31's 20 (pairs) and 5 (triples), and the extra cost.

**Cost.** CPU, about two to three hours.

## 33. Rank every extended triple by a continuous fit

*Added November 16 after #32.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/CONTINUOUS_TRIPLES.md)). The true
> triple now fits practice perfectly in 23 of 24 cases, but so do a median 160 others. Top-200 recall is 21
> of 25 with coarse-7 ties; top-40 only 13. The bottleneck is now underdetermination, not the fit.

#32 left the triple suspect list limited by the coarse grid: in 8 of 25 random-size cases the true
triple ranked below 1,000 on an 11³ grid, so refinement never saw it. Skip the grid and give every
extended triple (about 10,000 per case) #29's margin fit, started from the current values and from a
few coarse points. Measure top-40 recall on #31's random-size cases against #32's 14, then run #29's
targeted examples on the new list for all-three and visible-found counts.

**Cost.** CPU, about three hours.

## 34. Targeted examples on a 200-triple committee

*Added November 16 after #33.*

> **Status, November 16: done** ([write-up](../experiments/locate-repair/TRIPLE_COMMITTEE.md)). After 64
> targeted examples, all three faults are found in 16 of 25 random-size cases and every visible fault in 19
> (practice data alone: 3 and 3). Four cases lack the true triple in the committee; three end on a tie.

#33's continuous fit puts the true triple in the top 200 in 21 of 25 random-size cases, but a median
of 160 triples fit practice perfectly, so the top 40 holds it in only 13. Run #29's warm-started
targeted examples with the top 200 (coarse-7 ties) as the committee, 8 per round up to 64. Measure
all-three and visible-found counts against #29 and the cost of the larger committee.

**Cost.** CPU, about three to four hours.

## 35. Query where the lookalikes disagree, not uniformly

*Added November 16 after #34.*

> **Status, November 17: done** ([write-up](../experiments/locate-repair/BOUNDARY_QUERIES.md)). A pool near the
> tied members' boundaries ends at 17 of 25 all three and 19 every visible fault, against 16 and 19. It trails
> until the last rounds, fixes two cases and loses two; seeds 10 and 15 keep the same ties.

#34 ended 3 of 25 three-fault cases on a lookalike that tied the true triple on every label after 64
examples, and #22 found the same late-round problem for pairs: no uniform input among 20,000 split
the committee. Draw the query pool near the members' decision boundaries (for instance, by bisecting
between inputs that two tied members label differently) and re-run #34's last rounds on the tied
cases, then all 25. Measure ties left and all-three counts against #34's 16.

**Cost.** CPU, about one to two hours.

## 36. Are the last lookalikes wrong at all?

*Added November 17 after #35.*

> **Status, November 17: done** ([write-up](../experiments/locate-repair/LOOKALIKES.md)). Mostly not. The
> winner matches the original network on all but 0.1% of a million inputs in 22 of 25 cases, against 16 to 17
> that edit the exact broken entries. Only seed 14 is a clear miss beyond the fit's own limit. Mixed pool: no gain.

Seeds 10 and 15 end on the same two- and three-way ties under both query pools. For each case, save
the final fits of every tied member and measure how often each disagrees with the teacher on a
million uniform inputs, against the true triple's fit. If the lookalikes disagree with the teacher
no more than the true triple's fit does, they are equivalent repairs, and the benchmark should score
repairs by disagreement rate as well as by entries. Also try a mixed pool (half uniform, half
boundary), since the uniform pool led early and the boundary pool late.

**Cost.** CPU, about two hours.

## 37. Rescore every repair result by behaviour

*Added November 17 after #36.*

> **Status, November 17: done for two and three faults** ([write-up](../experiments/locate-repair/BEHAVIOUR_SCORES.md)).
> Three faults: behaviour arrives early, 16 of 25 after 16 labels against 9 by entries, and 22 against 19 at the
> end. Two faults: the behaviour score is stricter, 20 of 25 against 23, because four repairs pick the exact
> weights but set them about 0.02 spreads off, which costs 0.13% to 0.17%. The one-fault benchmark is not rescored.

#36 found that entry-based scoring undercounts three-fault repairs: 22 of 25 restore behaviour against
19 that find every visible fault. Rescore the two-fault results (#26, #31, #32) and the one-fault
benchmark the same way, with disagreement on a million inputs, and report both columns side by side.
Then check whether behaviour matches sooner: probe after every round, not only the last, to see how
many labels a behavioural repair really needs.

**Cost.** CPU, about two to three hours (re-runs with probing).

## 38. Queries that pin down the winner's values

*Added November 17 after #37.*

> **Status, November 17: done** ([write-up](../experiments/locate-repair/VALUE_QUERIES.md)). Value queries alone do
> nothing on the 200-point grid, because its 0.025-spread step is the limit. With a refined local fit they work:
> 24 of 25 repairs under 0.1% and 20 under 0.01%, against #37's 20 and 9, and lower disagreement than #37 in 24
> of 25 cases. Middle-of-tie on the coarse grid is worse than the edge.

#37 found four two-fault repairs that edit the exact broken weights and still disagree with the
original network on 0.13% to 0.17% of inputs, with values about 0.01 to 0.03 spreads off. The
targeted labels are chosen to split the committee over which weights to edit, so once the winner is
right they stop helping. After the winner leads, spend the remaining labels on inputs that its
nudged neighbours (each value moved by a small step either way) label differently, refit, and probe
again. Also try breaking the pair fit's ties at the middle of the tied region instead of the edge
nearest the corrupted value. Measure "under 0.1%" and "under 0.01%" against #37's 20 and 9.

**Cost.** CPU, about one to two hours.

## 39. Switch to value queries once the committee agrees, and carry the refined fit to three faults

*Added November 17 after #38.*

> **Status, November 17: done for two faults** ([write-up](../experiments/locate-repair/SWITCH_QUERIES.md)).
> Switching keeps every visible fault at 23 of 25 and ends level on behaviour (24 under 0.1%, 19 under 0.01%,
> against #38's 24 and 20), with a third fewer value queries, but reaches the 0.01% line about 24 labels later.
> The three-fault half moved to #40.

#38's best arm gives half of every round's labels to value queries, which costs one case of weight-finding
(22 against 23 every visible fault). Spend targeted labels until the committee's leader is clear (for example,
when the top pair's practice accuracy beats the runner-up's), then switch to value queries. Measure both columns
against #38's 24 and 22. Then try the same value queries on #34's three-fault triples, whose continuous margin
fit already resolves finer than a grid, against #37's 22 of 25 under 0.1% and 16 under 0.01%.

**Cost.** CPU, about two to three hours.

## 40. Value queries for three broken weights

*Added November 17 after #39.*

#38 and #39 closed the two-fault value gap with a refined fit plus labels aimed at the winner's values.
#34's three-fault repairs already use #29's continuous margin fit, so they may only need the queries. Give
half of each round's labels to inputs where the winning triple and a copy with one value nudged by 0.01
spreads disagree, and measure against #37's three-fault 22 of 25 under 0.1% and 16 under 0.01%, and its
19 every visible fault.

**Cost.** CPU, about three to four hours (the triple committee is slower).

**Status (November 17, done):** [TRIPLE_VALUES.md](../experiments/locate-repair/TRIPLE_VALUES.md). Half the labels as
value queries bring 24 of 25 three-fault repairs under 0.1% and 19 under 0.01% (#37: 22 and 16), but every
visible fault drops from 19 to 17. Seeds 17 and 20 move under 0.1%, and so does their true-triple fit, so in #37
those two were value-limited. Seed 14 still fails.

## 41. Switch rule for three broken weights

*Added November 17 after #40.*

#40 repeats #38's trade for three faults: value queries help behaviour (24 and 19 against 22 and 16) and cost
weight-finding (17 every visible fault against 19). For two faults, #39's rule (targeted labels while any pool
input splits the committee, value queries after) got the weight-finding back. Run that rule on #40's cases and
compare with both #37 and #40. Also refit the true triple from several starts, so the reference does not stick
as it did in seed 2.

**Cost.** CPU, about three hours.

**Status (November 17, done, negative):** [SWITCH_TRIPLES.md](../experiments/locate-repair/SWITCH_TRIPLES.md).
The three-fault committee almost never runs out of splits: 16 of 25 cases never switch and only 101 of 1,600
labels are value queries, so the result is #37's to the case (22 and 16 under 0.1% and 0.01%, 19 every visible
fault). Seeds 17 and 20, the two #40 fixed, never switch. The true triple fitted afresh gives the same reference
(23 and 18) as the chained fit.

## 42. Switch at the behaviour line for three broken weights

*Added November 17 after #41.*

#41's trigger waits until no pool input splits the committee, which three-fault committees almost never
reach. Switch instead once fewer than 0.1% of the pool (20 of 20,000 inputs) split it, the same line the
repairs are scored on, and compare with #37 (22, 16, 19 every visible fault) and #40 (24, 19, 17). Record
the number of splitting inputs every round, so the trigger can be re-read at other lines without rerunning.

**Cost.** CPU, about two hours.

**Status (November 17, done, negative):** [SWITCH_LINE.md](../experiments/locate-repair/SWITCH_LINE.md).
The trigger fires in 23 of 25 cases (472 of 1,600 labels become value queries), but behaviour does not improve:
22 under 0.1% as in #37, and 13 under 0.01% against #37's 16 and #40's 19. Every visible fault stays at 19. The
true triple refitted on these labels also does worse at 0.01% (13 against 18), so dropping all committee labels
at the line costs value information. Seed 20 never reaches the line.

## 43. Half value labels from the line on, for three broken weights

*Added November 17 after #42.*

#42 switched all 8 labels to value queries at the 0.1% line and lost precision; #40's half-and-half mix from the
first round has the best behaviour (24 and 19) but finds every visible fault in 17 cases against 19. Keep all 8
labels on committee splits until fewer than 20 pool inputs split the committee, then use #40's mix (4 value
queries, 4 splits). If the visible-fault loss in #40 comes from value queries asked before the committee has
found the weights, this should keep 19 visible faults and most of #40's behaviour gain.

**Cost.** CPU, about two hours.

**Status (November 17, done, negative):** [HALF_LINE.md](../experiments/locate-repair/HALF_LINE.md).
With the same trigger as #42 (23 of 25 cases cross it; 239 of 1,600 labels become value queries), the result is
#37's at both lines: 22 under 0.1% and 16 under 0.01%, the same cases. #42's loss at 0.01% is undone and the
true-triple refit is back at 23 and 18, but every visible fault drops to 18 (seed 11). Seeds 17 and 20, the
two #40 fixes, reach the line late or never, so a trigger that waits for the committee to settle cannot help them.

## 44. Value labels early, committee splits late, for three broken weights

*Added November 17 after #43.*

#43 shows value queries asked late change nothing, and #40's gains are in cases that never get near the line.
#40 is also ahead on visible faults early (14 against 12 after 32 labels) and falls behind only between 32 and
40 labels. Reverse the order: #40's mix (4 value queries, 4 splits) for the first four rounds, then all 8 on
committee splits. Compare with #37 (22, 16, 19 every visible fault) and #40 (24, 19, 17). If the early value
labels carry the behaviour gain and the late splits win back the weights, this should beat both.

**Cost.** CPU, about two hours.

**Status (November 17, done):** [EARLY_VALUES.md](../experiments/locate-repair/EARLY_VALUES.md).
It keeps both gains: 24 of 25 under 0.1% (as #40) and 19 every visible fault, 16 exact (as #37, the same
cases), with 400 value queries against #40's 797. Under 0.01% it has 18, between #37's 16 and #40's 19.
Seeds 17 and 20 end under 0.1%. Seed 14 still fails because its committee never held the true triple.

## 45. Where to switch from value labels to committee splits

*Added November 17 after #44.*

#44 used #40's mix for four rounds and then splits only, and got #40's behaviour with #37's weight-finding.
Four was a guess. Run the same rule with 2 and with 6 early rounds. With 2, fewer value labels may lose
seeds 17 and 20 again; with 6, the late weight catch-up (every visible fault only reaches 19 in the last
round) may not finish. If 4 is near the best, the rule is ready to write up; if 6 matches it at 0.1% and
beats it at 0.01%, the switch should come later.

**Cost.** CPU, about three and a half hours (two arms).

**Status (November 17, done):** [SWITCH_POINT.md](../experiments/locate-repair/SWITCH_POINT.md).
Two early rounds is the best three-fault arm so far: 24 of 25 under 0.1%, 20 under 0.01% (#44: 18, #40: 19),
median 0.0041%, with only 200 value queries. Its weights are #37's exactly (19 every visible fault, 16 exact),
and they arrive by 48 labels as in #37. Six early rounds is worse on every count (23, 16, 18 every visible fault).
Seed 14 still fails in every arm.

## 46. One early round, and two early rounds on fresh cases

*Added November 17 after #45.*

#45 found that 2 early rounds beat 4 and 6, but by two or three cases out of 25. Two checks. First, run 1 early
round on the same 25 cases, to see whether 8 value queries per case are already enough. Second, run 0, 2 and 4
early rounds on 25 fresh cases (seeds 25 to 49), where 0 is #37's rule, to see whether 2 rounds still lead when
nothing was tuned on those cases. If 2 still beats 0 and 4 on fresh cases at the 0.01% line, it becomes the
default for three faults; if not, #45's lead was noise.

**Cost.** CPU, about seven hours (four runs in parallel).

**Status (November 18, done):** [FRESH_SWITCH.md](../experiments/locate-repair/FRESH_SWITCH.md).
#45's lead was noise. On fresh cases 0, 2 and 4 early rounds give 23, 24 and 24 under 0.1% and 15, 15 and 16
under 0.01%, with the same weights. Over all 50 cases the mixes have 48 under 0.1% against 45 with no value
labels, and 2 and 4 rounds are level at 0.01% (35 and 34). One early round loses seed 17 (23 under 0.1%).
Two rounds stays the default because it needs half the value queries. The committee misses the true triple
in 8 of the 25 fresh cases (4 of the original 25); seed 49 is the one of them that fails in every arm.

## 47. Why the screen misses the true triple

*Added November 18 after #46.*

The committee of 200 triples misses the true triple in 12 of 50 cases (3, 5, 14, 17, 28, 30, 32, 37, 41, 44,
47, 49). Both clear failures left are among them (seeds 14 and 49), and in seed 49 the true triple, fitted
the committee's way, gets every practice label right while the winner gets 99.9%. For each of the 12, find
where the true triple ranks among all extended triples and at which stage it drops out: the pair screen that
seeds the triples, the coarse 7-point fit, or the cut to 200. Then try the cheapest fix that brings it back
(a larger committee, or ranking by the continuous fit before the cut) and rerun the 12 cases.

**Cost.** CPU, about an hour for the ranks; a few hours for the rerun.

**Status (November 18, done):** [SCREEN_MISSES.md](../experiments/locate-repair/SCREEN_MISSES.md).
Two ways to lose it. In 9 of the 12 the true triple is extended and fits every practice label but ties with
the 200th member and ranks 206th to 1,175th. In 3 (14, 47, 49) none of its pairs is among the best 200, so it
is never extended; in 14 and 49 it would have been the only triple to fit every label. Extending the best
1,000 pairs fixes both clear failures (seed 14: 0.61% to 0.0092%; seed 49: 0.34% to 0.0069%, exact weights)
and leaves 5 controls unchanged in kind, at about 3.5 times the cost per case.

## 48. The wider screen on all 50 three-fault cases

*Added November 18 after #47.*

#47 extended the best 1,000 pairs instead of 200 on 8 cases and fixed seeds 14 and 49, the two clear
failures. Run it with #45's 2-early-round rule on all 50 cases (seeds 0 to 49) and compare with the 200-pair
runs from #45 and #46. If it reaches 50 of 50 under 0.1% without losing a case elsewhere, it becomes the
three-fault default. Also record how often the true triple is in the committee.

**Cost.** CPU, about 12 hours (four runs in parallel, about 3,300 seconds per case).

**Status (November 18, done):** [WIDE_SCREEN.md](../experiments/locate-repair/WIDE_SCREEN.md).
50 of 50 under 0.1% (from 48), with no case lost at that line; under 0.01% 37 (from 35), with 6 gained and 4
lost, a shuffle that cases ending on the same weights show too. The true triple enters 3 more committees (14,
47, 49) and leaves none. Cost is about 4 times as much (51.5 against 12.3 CPU hours). An adaptive rule, widen
only when no committee member fits every practice label, fires on exactly seeds 14 and 49 and gets the same
50 of 50 at 14.0 CPU hours.

## 49. Does the adaptive widening flag hold on fresh cases?

*Added November 18 after #48.*

On seeds 0 to 49 the flag "the 200-pair committee's best member misses a practice label before any targeted
label" fires on exactly the two cases the 200-pair screen fails. Build the rule into `early_values.py`
(`--widen-to 1000`: run 200 pairs, and restart the case with 1,000 when the flag fires), then run it on 50
fresh three-fault cases (seeds 50 to 99) and record how often the flag fires, whether every case it skips
ends under 0.1%, and the cost against 200 pairs alone.

**Cost.** CPU, about 13 to 16 core-hours (four runs in parallel).

**Status (November 18, done):** [ADAPTIVE_WIDEN.md](../experiments/locate-repair/ADAPTIVE_WIDEN.md).
The flag never fired on the 50 fresh cases, and 5 of them (61, 63, 76, 83, 96) miss 0.1%, every one with the
true triple missing from a committee whose best member still fits every practice label. 45 of 50 under 0.1%,
28 under 0.01%, 13.9 CPU hours. A different flag, "the finished 200-pair repair still misses a label after
all 64 labels", catches all 7 failures across seeds 0 to 99 (16 fresh cases flagged, 11 old); on seeds 0 to 49
it keeps 50 of 50 under 0.1% at 22.8 CPU hours.

## 50. Widen when the finished repair still misses a label

*Added November 18 after #49.*

Rerun with 1,000 pairs the 16 fresh cases (seeds 50 to 99) whose finished 200-pair repair misses a label, and
score the end-flag rule on all 100 three-fault cases: 200-pair run everywhere, 1,000-pair result where flagged.
If it puts all five fresh failures under 0.1% without losing a flagged case that was already fine, it replaces
#48's start flag as the three-fault default.

**Cost.** CPU, about 16 core-hours (four runs in parallel, about 3,500 seconds per case).

**Status (November 19, done):** [END_FLAG.md](../experiments/locate-repair/END_FLAG.md).
The rerun fixes 2 of the 5 fresh failures (61 and 83, both exact); 63, 76 and 96 stay above 0.1% with the true
triple still missing from the 1,000-pair committee. No flagged case crosses 0.1% upward. Across all 100 cases
the rule gives 97 under 0.1%, 66 under 0.01% and 59 exact in 49.3 CPU hours, against 93, 63 and 55 in 26.2
for 200 pairs everywhere. The bar was all five fixed, so #48's start flag is not replaced; the end flag is
still the better alarm.

## 51. Where the three remaining failures lose the true triple

*Added November 19 after #50.*

Seeds 63, 76 and 96 end above 0.1% even with 1,000 pairs, and their true triple never reaches the committee.
Run #47's `screen_ranks.py` on the nine fresh cases whose 1,000-pair committee lacks the true triple (52, 55,
62, 63, 66, 68, 76, 93, 96) and record where each true triple's pairs rank in the joint pair screen. If the
best pair ranks beyond 1,000, the pair score is the weak step; if within, the triple is extended and lost at
the cut to 200, as in 9 of #47's 12 cases.

**Cost.** CPU, about an hour.

**Status (November 19, done):** [RANK_MISSES.md](../experiments/locate-repair/RANK_MISSES.md).
All nine are lost in the cut, none in the pair screen: a true pair ranks in the best 70 of 3,321, the true
triple is extended at 200 pairs and fits every practice label, and it ranks 204th to 4,839th among at least
200 other perfect fits. Extending more pairs cannot help these cases. A committee of 1,000 triples would hold
the true triple in 8 of the 9, all three failures included.

## 52. A larger committee instead of a wider screen

*Added November 19 after #51.*

Keep 200 pairs, but keep the best 1,000 extended triples in the committee instead of 200, on the 16 fresh
end-flag cases (seeds 50 to 99). Record whether 63, 76 and 96 go under 0.1%, whether the true triple reaches
the committee in the eight cases #51 predicts, what happens to the cases that were already fine, and the cost
per case against #50's wider screen (about 2,800 seconds per case).

**Cost.** CPU, unknown until one case is timed; the targeted rounds score every member, so up to about five
times #45's per-case time.

**Status (November 19, done):** [BIG_COMMITTEE.md](../experiments/locate-repair/BIG_COMMITTEE.md).
Keeping 1,000 triples fixes 4 of the 5 fresh failures (63, 76 and 96 on the exact weights, 83 on others) and
brings the true triple in for exactly the 8 cases #51 predicted, at a median 1,103 seconds per rerun against
2,808 for the wider screen. It gives 49 of 50 under 0.1% and 33 exact in 18.8 CPU hours. The one it misses
(61) is a never-extended case the wider screen fixes, and the end flag stays up on only 5 of the 16 after the
larger committee, so "larger committee, then wider screen if still flagged" puts all 50 fresh cases under
0.1%, 35 exact, in 22.5 CPU hours.

## 53. The two-step rule on all 100 three-fault cases

*Added November 19 after #52.*

Run the larger committee (`--committee 1000`, 200 pairs) on the 11 end-flag cases of seeds 0 to 49 (6, 14,
17, 21, 28, 30, 32, 37, 41, 47, 49). Score "200 triples; if the finished repair misses a label, 1,000
triples; if it still does, 1,000 pairs" on all 100 cases, using #48's 1,000-pair runs for seeds 0 to 49 and
#50's for seeds 50 to 99. If it keeps 100 of 100 under 0.1%, it becomes the three-fault default.

**Cost.** CPU, about 3.5 core-hours (11 cases at about 1,100 seconds).

**Status (November 19, done):** [TWO_STEP.md](../experiments/locate-repair/TWO_STEP.md).
All 100 under 0.1% (worst 0.088%), 75 under 0.01% and 69 exact in 42.9 CPU hours, against 93, 63 and 55 in
26.2 for the plain run and 97, 66 and 59 in 49.3 for #50's rule. On seeds 0 to 49 the larger committee alone
fixes neither failure and breaks seed 17, but the second flag catches all three. The 1,000-pair screen runs
on 10 of 100 cases. The worst case is one the flag never sees (seed 70).

## 54. The two-step rule end to end on unseen cases

*Added November 19 after #53.*

The rule was designed on seeds 0 to 99. Build it into `early_values.py` as one option (`--two-step`: run, and
rerun with `--committee 1000` and then `--pairs 1000` while the finished repair misses a label), and run it on
50 unseen three-fault cases (seeds 100 to 149). Record under 0.1%, under 0.01%, exact, how often each step
fires, and the cost against the plain run.

**Cost.** CPU, about 23 core-hours by #53's rates (four runs in parallel, about six hours).

**Status (November 19, done):** [UNSEEN.md](../experiments/locate-repair/UNSEEN.md).
All 50 unseen cases under 0.1% (worst 0.069%), 35 under 0.01% and 27 exact in 19.8 CPU hours, against 46, 31
and 20 in 10.9 for the plain run. The first flag fires on 15 cases and the second on 6. Three cases still miss
a label after both reruns; their true triple never reaches the committee, but their repairs stay under 0.1%.

## 55. Where the full rule still loses the true triple

*Added November 19 after #54.*

On seeds 124, 126 and 149 the true triple is missing from the committee after the 200-pair run, the
1,000-triple rerun and the 1,000-pair rerun. Rank the true triple at each cut, as #51 did (pair screen, cut to
the kept pairs, extension to triples, cut to the committee), to find where it drops out, and check whether one
of its three pairs is ranked below 1,000.

**Cost.** CPU, under an hour.

**Status (November 19, done):** [UNSEEN_MISSES.md](../experiments/locate-repair/UNSEEN_MISSES.md).
With 1,000 pairs all three true triples are extended, so the pair screen is not the problem. All three are cut
among perfect practice fits (ranks 1,441, 9,277 and 298 at 1,000 pairs, with 1,411, 7,766 and 246 triples tied
or better). Only 1,000 pairs and 1,000 triples together would keep one of them (149). The cost was about
3 core-hours, not under one, because the 1,000-pair committees fit 43,000 to 49,000 triples.

## 56. A third rerun with 1,000 pairs and 1,000 triples

*Added November 19 after #55.*

#55 predicts that rerunning with both `--pairs 1000` and `--committee 1000` keeps seed 149's true triple and
not 124's or 126's. Run it on the three cases the two-step rule leaves flagged, and record whether 149 ends on
the exact broken weights, whether any repair stops missing a label, and the cost per case. If it fixes 149
alone, add it as an optional third step only if its cost per flagged case is under about 1.5 CPU hours.

**Cost.** CPU, about 5 core-hours (three cases, side by side).

**Status (November 19, done):** [BOTH_WIDE.md](../experiments/locate-repair/BOTH_WIDE.md).
As predicted, it fixes seed 149 on the exact weights (0.045% to 0.0055%) and not 124 or 126. Across the 50
unseen cases it adds one case under 0.01% and one exact for 12% more CPU (about 0.8 hours per flagged case).
Seed 124's repair gets worse (0.012% to 0.029%) while starting to fit every label. Kept as the option
`--three-step`, not the default. Cost was 2.4 core-hours.

## 57. A better order among perfect practice fits

*Added November 19 after #56.*

In every case where the rules lose the true triple (#51's nine, #55's three), it fits every practice label
and is cut on the tie-break among hundreds to thousands of other perfect fits. For those 12 cases, record
every perfect fit's features (total edit size, largest single edit, which kinds of entry it edits, coarse-fit
accuracy) and test whether a different tie-break ranks the true triple inside the first 200. Analysis only;
any tie-break that wins must then be checked on all 150 cases.

**Cost.** CPU, about 6 core-hours (rebuilding 12 committees at 1,000 pairs).

**Status (November 19, done):** [TIE_BREAKS.md](../experiments/locate-repair/TIE_BREAKS.md).
Preferring the perfect fit with the smallest largest single edit keeps 5 of the 12 lost triples inside 200;
the current coarse-fit order keeps none. No tie-break keeps all 12, and seeds 52, 62, 68 and 124 are lost
under all seven tried. All 8 controls stay kept under every tie-break. Cost was about 11 core-hours.

## 58. The largest-edit tie-break in the full repair

*Added November 19 after #57.*

#57 shows that ordering perfect practice fits by their largest single edit keeps 5 of the 12 lost true triples
in the committee. Add it as an option to `early_values.py` and rerun the two-step rule with it on the 50
unseen cases (seeds 100 to 149) and on #51's nine lost cases. Record whether the five cases now end on the
exact weights, whether any case the current order gets right gets worse, and the totals against #54 (50 under
0.1%, 35 under 0.01%, 27 exact). Make it the default only if it loses no case under 0.1%.

**Cost.** CPU, about 25 core-hours.

**Status (November 19): done, see [LARGEST_EDIT.md](../experiments/locate-repair/LARGEST_EDIT.md).** On seeds
100 to 149 the new order ends 49 of 50 under 0.1% (coarse: 50), 34 under 0.01% (35) and 28 exact (27), and
costs 24.3 CPU hours against 19.8. Seed 104 crosses the line, so the coarse order stays the default. #51's nine
cases were not rerun.

## 59. The largest-edit tie-break only in the widest rerun

*Added November 19 after #58.*

#58 shows the new order hurts when it picks the first committee: 41 cases under 0.1% after the first stage
against 46. #57's five wins were all at 1,000 pairs, the last stage. Add a per-stage tie-break to
`early_values.py` so the first two stages keep the coarse order and only the 1,000-pair rerun uses the
largest-edit order. Run it on #51's nine lost cases and on the six unseen cases that reach that stage (102,
109, 124, 126, 129, 149), reusing the saved first two stages. Record whether any of #57's five (55, 63, 76, 93,
96) now ends on the exact weights, and whether any of the six gets worse. Every other case is unchanged by
construction.

**Cost.** CPU, about 12 core-hours.

**Status (November 20): done, see [LATE_EDIT.md](../experiments/locate-repair/LATE_EDIT.md).** #51's nine
cases never reach the 1,000-pair stage under the two-step rule, and #57's five already end exact at the second
stage. On the six that reach it, four end closer and two further, none crosses 0.1%, and seed 124 loses its one
true weight. 8 of 15 under 0.01% against 7, for 4% more last-stage CPU. The coarse order stays the default. Cost
was about 9 core-hours.

## 60. The two-step rule on a network twice as wide

*Added November 20 after #59.*

Every three-fault result since #28 is at H = 16. Run the full two-step rule (`--early 2 --two-step`, coarse
order) on fresh three-fault cases at H = 32 (`--width 32`), where there are about twice as many entries and four
times as many pairs. Pilot one seed first to measure cost; run 12 seeds if a seed costs under about 2 CPU hours,
6 otherwise. Record the share under 0.1% and 0.01%, the share exact, how many cases each stage flags, and the
CPU per case against #54 (50 of 50 under 0.1%, 35 under 0.01%, 27 exact, 0.4 hours per case).

**Cost.** CPU, 10 to 25 core-hours.

**Status (November 20): done, see [WIDTH32.md](../experiments/locate-repair/WIDTH32.md).** The pilot seed cost
1.8 CPU hours, so 12 seeds ran (200 to 211). All 12 end under 0.1% and 8 under 0.01%, about the same shares as
width 16, but only 5 end exact. Stages flag 7 / 1 / 4 cases. 2.06 CPU hours per case, 5.2 times width 16. The
third stage makes all four cases it reaches worse and is 45% of the CPU. Cost was about 25 core-hours.

## 61. Keep the third stage only when it fits every practice label

*Added November 20 after #60.*

At width 16 the 1,000-pair third stage ends exact in the three cases where it reaches a perfect practice fit and
is slightly worse in the other three. At width 32 it is worse in all four cases it reaches. Score a final-choice
rule from the saved per-stage results (`two_step` in each case): keep the third stage's repair only when it fits
every practice label, otherwise keep the second stage's. Also score "keep the stage with the best practice
accuracy". Report the shares under 0.1% and 0.01% and exact at both widths, against the current rule, with seeds
100 to 149 and 200 to 211. If a rule never loses, test whether it should also skip the third stage at width 32
when the second stage is already close.

**Cost.** No new runs; minutes of CPU.

**Status (November 20): done, see [FINAL_STAGE.md](../experiments/locate-repair/FINAL_STAGE.md).** Keeping the
third stage only when it fits every practice label never loses a line in three runs: width 16 stays 50 / 35 / 27
with a worst case of 0.047% (was 0.069%), and width 32 goes from 8 to 10 of 12 under 0.01%, matching the oracle.
"Best practice" never ends further but gains less. Skipping the third stage saves 45% of the CPU at width 32 but
loses three exact wins at width 16. The rule is in-sample, so it needs fresh seeds.

## 62. The third-stage rule on fresh cases

*Added November 20 after #61.*

#61 picked "keep the third stage only when it fits every practice label" after seeing the runs it was scored on.
Add it to `early_values.py` as `--final-choice third-if-perfect` (default `last`, which leaves every earlier run
unchanged), keeping every stage's result in `two_step` as now. Run the two-step rule on 50 fresh width-16
three-fault cases, seeds 150 to 199. Score both final choices from the same run. Record how many cases reach the
third stage, whether the rule ever ends further than `last`, and the shares under 0.1% and 0.01% and exact
against #54.

**Cost.** CPU, about 20 core-hours.

**Status (November 20): done, see [THIRD_RULE.md](../experiments/locate-repair/THIRD_RULE.md).** The rule did not
hold on seeds 150 to 199: it drops five imperfect third stages, one ends closer and four further, and seed 197
crosses 0.1% (48 / 32 / 31 against `last`'s 49 / 32 / 31). The two perfect third-stage fits are exact and both
rules keep them. Pooled over 100 cases, an imperfect third stage ends closer in 4 of 8. `last` stays the default.
17 of the 33 pooled cases over 0.01% already edit exactly the three broken weights.

## 63. Refine the values once the repair has the right weights

*Added November 20 after #62.*

17 of the 33 width-16 cases over 0.01% in seeds 100 to 199 edit exactly the three broken weights and fit every
practice label, so their gap is in the values, not in the location. For each of the 58 exact cases, refit the
three values on the same practice labels, starting from the repair's values, and keep the point that fits every
label with the widest margin (the middle of the region that fits) instead of the first point the fit reaches.
No new labels and no new search. Report how many exact cases cross under 0.01%, and check that none of the
cases already under it moves over.

**Cost.** CPU, minutes per case.

**Status (November 21): done, see [REFINE_VALUES.md](../experiments/locate-repair/REFINE_VALUES.md).** The final
labels were not saved, so each exact case's kept stage was replayed (about 25 core-hours; all 58 match). The middle
of the region that fits is a little closer on average (geometric mean 0.88 to 0.96 across three walks) but moves
the 0.01% line by 0 to 2 of 58, and one or two cases under it move over. The gain is in the 14 cases whose region
is widest (11 closer in every walk, mean change 0.73). Where the region is narrow the labels already pin the
values.

## 64. Take the middle only when the labels leave room, on fresh cases

*Added November 21 after #63.*

#63 found that taking the middle of the region that fits helps only where that region is wide, but chose the
width bands after seeing the results. Fix the rule beforehand: after the repair's last round, run the 1,600-step
walk on the winner's values, and keep the middle only when the mean chord is at least 0.005 spreads and the
middle fits every label the repair fits. Save the final labels as part of every run (`keep`), so no replay is
needed. Run it inside the two-step repair on 50 fresh width-16 three-fault cases, seeds 200 to 249, for every
case, not only exact ones. Report the shares under 0.1% and 0.01% and exact with and without the step, and how
many cases cross the 0.01% line each way.

**Cost.** CPU, about 30 core-hours.

**Status (November 21): done, see [MIDDLE_STEP.md](../experiments/locate-repair/MIDDLE_STEP.md).** On seeds 200
to 249 the rule fixed beforehand uses the middle in 8 cases (6 closer, 2 further) and takes under 0.01% from 36 to
37. The middle whenever it fits as many labels does better: closer in 32, further in 15, and 39 under 0.01% with
none moving over. #63's width bands did not hold (0.82, 0.99, 0.81 from narrowest to widest). The final labels were
used but not written to disk.

## 65. Ask the next labels where the region of values is still unsure

*Added November 21 after #64.*

After the middle step, 11 of 50 fresh cases are still over 0.01%. Only 3 of them edit exactly the three broken
weights, so most still have a wrong weight among the three. One more round of labels could help either way. The
walk that finds the middle gives a few hundred value triples for the winner that all fit every label. Pick 8 new
inputs from the usual uniform pool of 5,000 where those sampled networks split most evenly, label them, refit every
committee member on all 1,096 labels, pick the winner again and take its middle. Compare with 8 more labels chosen
the usual way (one more round of committee splits) and with no extra labels, on 50 fresh width-16 cases, seeds 250
to 299. Write the final labels to disk this time. Report the shares under 0.01%, exact repairs, and the cases that
cross each way.

**Cost.** CPU, about 35 core-hours.

**Status (November 22): done, see [REGION_LABELS.md](../experiments/locate-repair/REGION_LABELS.md).**

## 66. Rebuild the committee from the final labels

*Added November 22 after #65.*

After #65's region-split labels, 9 of 50 cases are still over 0.01%. 7 of them edit at least one wrong weight, and
none of those 7 committees contains the true triple, so more labels on the same committee cannot help. The
committee was screened from the 1,024 practice labels alone. #65 wrote every case's final labels to disk: those
1,024, the 64 asked during the repair, and the 8 region-split inputs, 1,096 in all. Rerun the pair screen and the triple committee on those labels for all 50 cases of
seeds 250 to 299, refit, pick the winner and take its middle. Report how often the true triple enters a committee
that lacked it, exact repairs, the shares under 0.01%, and any case that gets worse.

**Cost.** CPU, about 10 core-hours.

**Status (November 23): done, see [REBUILD_COMMITTEE.md](../experiments/locate-repair/REBUILD_COMMITTEE.md).**

## 67. Choose among the members that fit every label

*Added November 23 after #66.*

Rebuilt on the 1,096 final labels, 46 of 50 committees hold the true triple, but the repair picks it in only 30.
In 12 of the 16 misses the true triple fits every label as well as the winner and loses the tie-break by fit cost;
in 4 its fit reaches fewer labels than the winner's. Rerun #66's rebuild on seeds 250 to 299, saving every
member's fit. Score three tie-breaks on the same members: smallest fit cost (current), the coarse 7-value order
(the committee's own first member), and the smallest largest single edit (#58). Also refit the members within 1%
of the best accuracy from 5 more starts and pick again. Report exact repairs, the shares under 0.01% and 0.1%, and
how many members tie at the top.

**Cost.** CPU, about 25 core-hours.

**Status (November 23): done, see [TIE_BREAK_FINAL.md](../experiments/locate-repair/TIE_BREAK_FINAL.md).**

## 68. Ask labels where the tied members disagree

*Added November 23 after #67.*

In #67 the true triple's values fit every final label in all 50 cases, and the median case has 4 committee members
tied on every label. Most ties are between near-identical networks, but in seed 255 three members tie and the pick
has one right weight (0.062%). Take each case with two or more members tied at the top after #67's refit, draw a
fresh uniform pool of 5,000 inputs, and label the 8 where the tied members' networks (at their middles) split most
evenly. Refit the tied members on all 1,104 labels, pick again by coarse order, take the middle, and compare exact
repairs, the 0.01% and 0.1% lines and the geometric-mean change with #67's refit-coarse rule on the same seeds.

**Cost.** CPU, about 10 core-hours.

**Status (November 23): done, see [TIED_LABELS.md](../experiments/locate-repair/TIED_LABELS.md).**

## 69. Find out why the tied members agree

*Added November 23 after #68.*

In #68, 41 of 50 cases still have two or more committee members that fit every label after 24 more labels, and in 87
of 127 rounds no input in a pool of 20,000 split them. In 28 of the 41 the tie count is exactly 2^k for a winner
with k output-layer edits, the mirror symmetry (#22). The other 13 tie more members than that, up to all 200. Rerun
#68 on the 41 tied cases, keep every tied member, and compare each with the winner: whether it is one of the
winner's mirrors, which other weights it edits, whether those weights sit on a hidden unit that is never active on
the 1M-point probe or never changes sign there, and how often the two networks disagree on the probe itself.

**Cost.** CPU, about 20 core-hours to rebuild the committees and fits, since #68 saved only each round's winner; the
comparison itself is cheap.

**Status (November 23): done, see [TIED_AGREE.md](../experiments/locate-repair/TIED_AGREE.md).**

## 70. Fix the six cases still over 0.01%

*Added November 23 after #69.*

#69 showed that every member tied at the top is within 0.01% of the winner, so ties no longer cost anything. What
keeps the repair off the 0.01% line is the 6 cases #68 left over it: in 258, 261 and 296 the committee lacks the
true triple, in 259 the true triple fits fewer labels than the winner, and in 260 and 269 the repair is exact but its
values are off. For the first three, rebuild the committee twice as large with twice as many pairs kept and check
whether the true triple enters and wins. For 259, refit the true triple and the winner from 20 random starts each and
see whether the fit, not the labels, is what loses it. For 260 and 269, keep the winning triple fixed, ask 64 labels
in a band around the fitted network's decision boundary and refit only the values. Score each against #68's final
pick on the 1M-point probe.

**Cost.** CPU, about 5 core-hours.

**Status (November 24): done, see [FIX_SIX.md](../experiments/locate-repair/FIX_SIX.md).**

## 71. Choose among the fixes from labels alone

*Added November 24 after #70.*

#70 found that each of its three fixes (a bigger committee, restarts, boundary labels) rescues a different one of the
six leftover cases, and none makes any case worse on the probe, but choosing the best fix per case used the probe.
Run restarts and boundary labels on all 50 of #67's cases (the bigger committee only where the committee lacks the
true triple), hold out 64 fresh uniform labels per case that no fit sees, and pick among #68's repair and each fix's
by held-out accuracy, then by fewest edits. Score the picked repair on the 1M-point probe: how many of the 50 are
under 0.01%, and whether any of #68's 44 cases under the line crosses back over.

**Cost.** CPU, about 20 core-hours.

**Status (November 24): done, see [PICK_FIXES.md](../experiments/locate-repair/PICK_FIXES.md).**

## 72. Run the whole repair on 50 fresh cases

*Added November 24 after #71.*

#71's gate for the bigger committee and its disagree chooser were designed after seeing #67's 50 cases, so its 47 of
50 under 0.01% is not a held-out result. Build 50 fresh width-16 three-fault cases (seeds 300 to 349) and run the
whole repair as it now stands: #68, restarts, boundary labels, the bigger committee where #68's pick misses a label,
and the disagree chooser. Score on the 1M-point probe, change nothing after seeing the results, and report how many
are under 0.01% against #68 alone on the same cases.

**Cost.** CPU, about 20 core-hours.

**Status (November 24): done, see [FRESH_REPAIR.md](../experiments/locate-repair/FRESH_REPAIR.md).**

## 73. Find where the repair's time goes, and cut it

*Added November 24 after #72.*

The whole repair now holds up on fresh cases (49 of 50 under 0.01%), but it costs about 0.75 core-hours a case, nearly
twice the estimate, and nobody has measured which part costs what. Across the 100 cases of #71 and #72, 6 rescues
came from three different fixes (boundary labels 4, restarts 1, bigger committee 1), so no fix can simply be dropped.
Rerun the 100 cases with the time of each stage recorded (#65's stages, the rebuilt committee, #68's rounds, each fix
and the chooser), then try a cheaper order: build the boundary candidate first, and build restarts and the bigger
committee only where the chooser's split inputs show #68's pick and the boundary candidate disagreeing, or where the
pick misses a label. Report core-hours a case and cases under 0.01%, against #72's pipeline on the same cases.

**Cost.** CPU, about 60 core-hours for the timed rerun of 100 cases, less for the cheaper order.

**Status (November 25): done on a 35-case sample, see [TIME_REPAIR.md](../experiments/locate-repair/TIME_REPAIR.md).** A pilot showed the fixes cost under a second each, so the cheaper order was scored only as a side note (it saves 0.06 of 51.5 core-hours). The margin refine takes 87% of #71's stage.

## 74. Cut the margin refine's cost and check the picks hold

*Added November 25 after #73.*

#73 found that one step, the margin refine (Nelder-Mead on the label margins, `xatol` 1e-6, `fatol` 1e-9, up to 800
iterations, after every fit and twice in every refit), takes 87% of #71's stage on every case, and #65's stage uses
the same functions. Try cheaper versions on #73's 35 cases, from the same labels: looser tolerances and fewer
iterations; refining only committee members within reach of the top score; and skipping the refine when the grid fit
already fits every label. For each, report core-hours, how often the disagree chooser's pick changes, and cases under
0.01% against #73's 34 of 35. Then time #65's stage by step on a few cases to confirm the refine dominates it too.

**Cost.** CPU, about 10 to 25 core-hours: each cheaper version should cost a fraction of #73's 21.5.

**Status (November 25): done for #71's stage, see [CUT_REFINE.md](../experiments/locate-repair/CUT_REFINE.md).** A loss that recomputes only the touched hidden units cuts the stage from 21.5 to 15.3 core-hours with the same pick in 35 of 35; adding looser stopping cuts it to 12.0 and keeps 34 under 0.01% and 28 exact, with 3 picks changed. The refine is still 70% to 76% of the stage. Timing #65's stage moved to #75.

## 75. Cut #65's stage the same way, and batch the refine

*Added November 25 after #74.*

#74 cut #71's stage 1.40x (columns, same answers) and 1.79x (loose, same line, 3 picks changed), but #65's stage is
58% of the whole repair (30.0 of 51.5 core-hours on #73's cases) and was never timed by step. Rerun #65's stage on a
handful of #73's cases with the original, columns and loose refines, timed by step, and check that its fixes and
#71's picks on top hold. Then try one Nelder-Mead run over many triples at once in arrays, in place of a Python loop
per triple, on #71's stage, and report the speedup and whether the picks match columns.

**Cost.** CPU, about 15 to 30 core-hours: #65's stage costs about 0.86 core-hours a case at the original speed.

**Status (November 25): timing done on 8 cases, see [TIME_65.md](../experiments/locate-repair/TIME_65.md).** The refine is 86% of #65's stage; columns cuts the stage 1.65x and loose 2.03x, with 7 of 8 under 0.01% in every run. columns ends with different labels in 4 of 8 cases, so #71's picks on top are not yet checked; that and the batched refine moved to #76.

## 76. Check the cheaper refines end to end, and batch the refine

*Added November 25 after #75.*

#75 found that columns, which changed nothing in #71's stage, ends #65's stage with different labels in 4 of 8 cases
(loose in all 8), though #65's repair keeps its disagreement. So #74's identical picks do not yet cover the whole
repair. Run #71's stage with columns and with loose on the labels #75 saved for its 8 cases, and compare the disagree
chooser's pick and the chosen network's disagreement with #73's. Then try one Nelder-Mead run over many triples at
once in arrays, in place of a Python loop per triple, and report the speedup on #71's stage and whether the picks
match columns.

**Cost.** CPU, about 6 to 12 core-hours: #71's stage on these 8 cases took 3.0 (columns) and 2.4 (loose) in #74.

**Status (November 25): done on 8 cases, see [END_TO_END.md](../experiments/locate-repair/END_TO_END.md).** On the new labels columns cuts the whole repair 1.54x with #73's pick in 8 of 8; loose cuts it 1.96x and keeps all 8 under 0.01% but changes 3 picks. A batched Nelder-Mead that follows SciPy's steps exactly cuts #71's stage a further 1.39x over columns with the same picks in 8 of 8. Running it in #65's stage and on fresh seeds moved to #77.

## 77. Run the whole cheaper repair on fresh seeds

*Added November 25 after #76.*

#76 checked columns end to end and the batched refine in #71's stage, but only on 8 cases these refines were already
tried on. Install the batched refine in #65's stage too (its committee calls are the same), then run the whole repair
twice on 12 fresh seeds (350 to 361): once with the original refine and once with columns plus batching. Compare
core-hours, the disagree chooser's pick, the chosen network's disagreement on a million probe inputs, and how many
cases are under 0.01% and exact.

**Cost.** CPU, about 20 to 35 core-hours: #73's original repair averaged about 1.5 core-hours a case, and #76 suggests
the cheaper one takes a little over half of that.

**Status (November 25): done on 12 fresh cases, see [FRESH_CHEAP.md](../experiments/locate-repair/FRESH_CHEAP.md).** The cheap repair (columns plus batching in both stages) takes 6.24 core-hours against 13.67 (2.19x), keeps all 12 under 0.01% and makes the original's pick in 11 of 12; it loses one exact repair (seed 356, 2 of 3 faults at 0.0006%). A 50-seed held-out run with it moved to #78.

## 78. Run the cheap repair on 50 more fresh seeds

*Added November 25 after #77.*

#72 ran the whole repair on 50 fresh seeds (300 to 349) and #77 showed the cheap repair does the same job on 12 more
for under half the compute. Run the cheap repair on 50 seeds nothing has been run on (400 to 449) and report how
many chosen networks are under 0.01% and 0.1% disagreement on a million probe inputs, how many are exact, and the
cost per case, next to #72's held-out numbers.

**Cost.** CPU, about 25 to 30 core-hours: the cheap repair averaged 0.52 core-hours a case in #77.

**Status (November 26): done on 50 fresh cases, see [FRESH_50.md](../experiments/locate-repair/FRESH_50.md).** The cheap repair takes 20.88 core-hours against #72's 37.56 (1.80x), puts 48 of 50 chosen networks under 0.01% (49 in #72) and all 50 under 0.1%, and is exact in 39 (39 in #72). The original repair, rerun on the two cases over the line (400 and 414), returns the same networks. On 414 #71's stage threw away #65's exact repair at 0.0035%; keeping it in the running moved to #79.

## 79. Keep #65's own repair in the running

*Added November 26 after #78.*

On seed 414 #65's stage ended with an exact repair at 0.0035%, but #71's disagree chooser only picks among its own
candidates (pick68, restarts, boundary and, when the gate opens, the bigger committee), so it returned a one-fault fit
at 0.069%. Add #65's repair as a further candidate in the chooser, scored on the same asked labels, and rerun #71's
stage (cheap) on #72's and #78's 100 seeds. Report how often the chooser takes #65's repair, how many chosen networks
are under 0.01% and exact, and whether any case gets worse.

**Cost.** CPU, about 20 core-hours: #71's stage took 10.1 core-hours on #78's 50 cases with the cheap repair, and
#72's seeds need it rerun cheap as well (candidate networks are not saved, only their winners).

## Posts the current results already support

1. **"815 weights that beat PPO 20×: reproducing a hand-written Breakout network."**
   - The released network reproduces exactly on a fresh machine: 502.880 ± 7.747 over
     1,000 episodes, versus the published 517.411 ± 7.417. The PPO references are
     15.8 and 25.3.
   - It is a 6 → 8 → 2-unit circuit with 815 nonzero parameters.
   - Its score is mostly survival time: 52% of episodes hit the 10,000-step ceiling,
     and r = 0.992 between score and length.
   - A fresh tool-free model response matched it (511.5 ± 5.5, not significantly
     different).
   - *Caveats:* the research and the network belong to Takehi et al.; the fresh network
     is one attempt (roadmap #4); training-data exposure cannot be ruled out.
2. **"Almost right fails: the geometry of a one-number weight-editing benchmark."**
   - Across 400 synthetic cases, a perfect practice score passes the held-out rule only
     27–42% of the time.
   - Overshoot tolerance is 2–7× tighter than undershoot.
   - The smallest practice-perfect edit succeeds about 80% of the time; the scripted
     random control about 11%.
   - Every recorded model attempt (3 by Codex) shows the overshoot signature: full
     repair plus 1–6 regressions.
   - *Caveats:* synthetic one-entry tasks; three model attempts; the Claude pilot is
     harness-confounded (#3), so leave it out of any comparison.
3. **"Verifying a published LLM weight edit bit-for-bit."**
   - The p50c reconstruction matches exactly: 130,188 BF16 entries across two tensors,
     both hashes match, and 318 other tensors are bitwise unchanged.
   - The runner keeps every prompt, seed, token and grade, so the eventual full result
     can be audited.
   - The two-problem pilot is anecdotal but suggestive: capped samples went from 8/8 to
     1/8, and the original's only success was a capped sample. That sets up roadmap #1.
   - *Caveats:* present the pilot as a preview, not a benchmark result. Until the full
     run lands, this is a methods post.

## Small fixes that unblock the roadmap

- Commit the status of the full MATH-500 run (progress, job hash, environment). Add a
  capped-as-wrong and finished-only accuracy to `qwen/reproduce_qwen.py compare` (#1).
- Add `cli-structured-actions-v3` with client-default structured-output retries,
  leaving v2 unchanged (#3).
- Commit or share the fresh Breakout checkpoint `fresh.npz` as evidence (#2, #4).
- Align CI's NumPy pin (2.3.5) with `requirements.txt` (2.5.4).
- Show the analytic random-control expectation (about 11%) next to the drawn result
  in benchmark scorecards.
- Decide on PR [ZayneMason/weight-space-lab#4](https://github.com/ZayneMason/weight-space-lab/pull/4)
  before starting #6; it changes the Qwen runner's job identity.
