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
