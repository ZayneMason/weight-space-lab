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
