# Lab notebook

A running record of what the lab has, what each experiment showed, and what is
broken or unfinished. Newest entry first. Every entry separates **measured** results
from **interpretation**. Numbers come from committed evidence or from commands that
were actually run on the date shown.
The [roadmap](ROADMAP.md) turns the open questions into ranked experiments.

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
