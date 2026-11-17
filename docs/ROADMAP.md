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

#36 found that entry-based scoring undercounts three-fault repairs: 22 of 25 restore behaviour against
19 that find every visible fault. Rescore the two-fault results (#26, #31, #32) and the one-fault
benchmark the same way, with disagreement on a million inputs, and report both columns side by side.
Then check whether behaviour matches sooner: probe after every round, not only the last, to see how
many labels a behavioural repair really needs.

**Cost.** CPU, about two to three hours (re-runs with probing).

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
