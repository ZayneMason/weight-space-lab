# Lab notebook

A running record of what the lab has, what each experiment showed, and what is
broken or unfinished. Newest entry first. Every entry separates **measured** results
from **interpretation**. Numbers come from committed evidence or from commands that
were actually run on the date shown.
The [roadmap](ROADMAP.md) turns the open questions into ranked experiments.

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
