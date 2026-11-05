# Local verification record

Date: November 4, 2026. This records implementation checks and explicitly labeled
small CLI pilots; it is not a completed multi-model benchmark study.

## Automated checks

- Main Python suite: **87 tests run, 86 passed, 1 skipped**. The skipped legacy
  integration test expects a specific absent reference directory. The new bounded
  Breakout workflow separately fetched/verified the actual pinned reference.
- Qwen unit suite: **8 passed**; no GPU inference was launched by those tests.
- Node plugin suite: **2 passed**.
- Controller checks cover consumed rejection/failure budgets, counter-reset
  detection, locks, freeze ordering/failure records, forbidden post-freeze edits,
  one-time final evaluation, frozen-file tampering, no-selection closure, and regression-based failure.
- Editor-session checks cover global budgets across reconnects, pinned editor
  configuration, malformed/oversized input, duplicate IDs, concurrent writers,
  transcript tampering, forbidden paths/finalization and out-of-session mutation.
  Fault injection covers an operator feedback/freeze between request reads, plus
  a change after dispatch begins but before its mutation lock is acquired.
- Workflow checks cover offline behavior, collisions, source preservation,
  recomputed counts/predictions, incompatible comparisons, integrity failures,
  metadata export, and failure retention.
- Benchmark checks cover four feasible task families, byte-matched editor cases,
  independent practice/final draws, complete trial denominators, no-op controls,
  missing-key refusal, counters/artifact tampering, native provider-shaped loops,
  retained usage on malformed calls, bounded nesting, and billing-anomaly blocking.
- CLI fixtures cover secret-environment allowlisting, structured actions, usage
  retention on malformed/truncated output, real fake-child capture, timeout and
  cancellation cleanup, separate opt-ins/counters, and session-only key transfer.

Main CPU checks used Python 3.12.10, NumPy 2.5.4, SafeTensors 0.8.0 and MinAtar
1.0.15 on Windows. Qwen unit checks used the existing isolated Linux environment.

## Offline demonstration

Executed in a new directory:

```bash
uv run --script lab.py demo runs/sign-demo-final
python lab.py verify runs/sign-demo-final
```

Result: 0/6 to 6/6 correct, two of four entries changed, six fixed examples and zero
regressions. Source and candidate hashes match their files. The exact patch is
prewritten and the examples are illustrative. Repeated-output preservation is
covered by the collision test, which compares every file hash before and after.

The README SVG was generated from this actual run. It was rasterized with an
offline SVG renderer and visually inspected; nearby input labels were separated
after inspection. No browser screenshot is claimed.

## Controller controls

```bash
python scripts/check_controller.py runs/controller-validation
python lab.py experiment audit runs/controller-validation
```

Both manual candidates and both feedback calls are retained. Candidate c001 fixed
eight practice examples and regressed eight, leaving average accuracy unchanged.
Candidate c002 fixed eight and regressed zero. The second candidate was frozen
before final evaluation. The preserved baseline scored 75% and the frozen feasible
control scored 100% on the 256-example final set, with zero regressions.

This is a known feasible manual control, **not agent discovery or an external-agent
trial**. `evidence/controller-controls.json` contains the sanitized measurements;
private final examples and seeds are excluded. An external-agent study remains
unexecuted and needs operator-supplied isolation and editor provenance.

## Recorded stdio editor control

```bash
python scripts/check_editor_session.py runs/editor-transport-reviewed
python lab.py editor audit runs/editor-transport-reviewed scripted
python lab.py experiment audit runs/editor-transport-reviewed
```

The actual CLI subprocess completed five requests: describe, inspect, propose,
feedback and freeze. The declared editor was a prewritten manual control, with
provider `none`. One proposal and one feedback call (32 example exposures) were
consumed. The editor exited at `frozen` before the operator's separate final
evaluation. Both audits passed after finalization. The baseline scored 75% and
the known feasible control scored 100%, with zero regressions on the final set.
These measurements validate transport and accounting, not external-agent capability.

A fault-injection test also confirms that failure to create `final/` is journaled
as a terminal `final_failed` attempt, with a retained fallback failure file and no
retry opportunity. Version 0.5.0 ZIP/tar packages built with the bridge, guide and
check script; runtime session directories are excluded.
The ZIP was extracted into a fresh directory, its required editor files and lack
of runtime/model artifacts were checked, and its stdio control passed independently
of the repository source paths.

## Synthetic editing benchmark calibration

```bash
python scripts/check_benchmark.py runs/benchmark-validation
python lab.py benchmark audit runs/benchmark-validation
```

Four task families × three task seeds produced 12 cases. Every case supplied
identical input/model/data bytes to each editor/control. The 36 scripted trials
completed: privileged feasible control **12/12** successes, unchanged control
**0/12** with 50% candidate accuracy, and independently seeded random search
**1/12** successes in this retained draw. The random result is one calibration,
not an estimate of model capability or a universal random-search bound.

The campaign audit checked all **84 predeclared slots**: 36 controls plus 48 API
editor slots. All API slots remain pending, global reserved API calls/output tokens
are zero, and no paid/provider requests were sent. The offline HTML/JSON scorecards
keep those pending slots distinct from measured failures.

Provider-shaped fixtures separately exercised both Claude and Fireworks tool
envelopes and the complete freeze-before-final loop using patched local transports.
They do not establish real endpoint compatibility, account/model availability or
editor capability. Usage fields stay counted even when native tool envelopes are
malformed; reservations remain consumed without retries, and a usage anomaly
blocks subsequent paid calls. Both protocol and standards review findings were
resolved before publication.

Version 0.6.0 ZIP and tar packages contained the same 83 files, including the
benchmark modules/configuration/guide/calibration script and excluding runtime
records and model files. A fresh ZIP extraction ran a hidden-layer control campaign
and audit successfully with zero API calls.

## CLI integration and limited live pilots

Installed clients were Claude Code **2.1.285** and Codex CLI **0.164.0**. Both
returned valid inspection actions in live format checks using existing CLI
sign-ins. Claude's initial expired OAuth failure is retained separately; the live
checks continued after the user completed normal sign-in. No Fireworks or direct
paid API adapter calls occurred. CLI activity consumed account allowance.

The current structured interface is `cli-structured-actions-v2`. A fresh matched
threshold case (task seed 47; shared private final draw) was evaluated for both
CLIs under the same outer budgets, with low effort requested:

| Editor | Outcome | Final accuracy | Repair accuracy | Regressions | CLI rounds |
| --- | --- | --- | --- | --- | --- |
| Claude CLI (`sonnet` alias) | Structured-output execution failure before selection | Unmeasured | Unmeasured | Unmeasured | 2 |
| Codex CLI (`gpt-6.1-sol`) | Frozen and evaluated; success rule failed | 255/256 (99.61%) | 100% | 1 | 6 |

Claude returned four valid inspection/status actions before its next CLI response
failed with `error_max_structured_output_retries`. This is a recorded harness/client
execution failure, not evidence that Claude cannot edit weights. Its known usage
was 3,728 input and 109 output tokens. Codex used 10 tool requests, including one
tool error, and one feedback call; known usage was 95,532 input and 420 output
tokens. Codex's repair gain did not satisfy the zero-regression criterion.
The campaign audit passed; paired ranking was withheld because one trial failed
before evaluation. The report is local under `runs/cli-structured-pilot/`.

Earlier integration attempts used a different JSON-in-string interface and remain
under `runs/codex-cli-pilot/` and `runs/cli-paired-pilot/`. Codex's earlier final
accuracies were 253/256 with 3 regressions (seed 11) and 250/256 with 6 regressions
(seed 23), both failing the same strict success rule. Claude's seed-23 attempt
stopped with a structured-output failure before feedback or final evaluation.
These attempts are retained, labeled v1, and are not pooled with the v2 pilot or
used to claim a general model ranking. No failed slot was overwritten or rerun
in place.

The Python/Tk key dialog was confirmed open and responsive by window title only.
Its masked field was not read by the assistant. An offline fake-child test confirmed
that the key goes through a new terminal's environment, not arguments or a file,
without changing the parent environment. The initial PowerShell helper was blocked
by the machine's unsigned-script policy and was replaced; no execution policy was
changed.

## Bounded MinAtar Breakout run

```bash
uv run --with minatar==1.0.15 --script lab.py breakout runs/breakout-check --mode smoke --episodes 8 --max-steps 1000 --workers 2 --reference runs/breakout-smoke/reference
python lab.py export-evidence runs/breakout-check evidence/breakout-smoke
python lab.py verify evidence/breakout-smoke
```

A preceding smoke run downloaded and verified the six pinned assets; the command
above reused their verified copy. Both bounded runs used seed 0 and produced a
mean score of **77.875**, SE **3.090524**, over eight episodes. All eight reached the
1,000-step ceiling. This is a smoke result, not the published 1,000-episode benchmark.

Episode zero was replayed with the evaluation's exact policy, environment/action
seeds and step ceiling. Its result and recorded prefix matched the evaluated
episode. The local HTML includes a playable 200-frame prefix and raw replay JSON.
Published research attribution and asset licenses are retained.

## Evidence and report checks

- Sanitized existing Qwen pilot samples agree with both stored aggregate summaries.
  Generated text, token IDs, questions, answers and private paths are omitted.
- New workflow ledgers retain relative file hashes and snapshots of measured source.
- Demo and Breakout metadata exports passed integrity checks.
- HTML parsing found no external resources. Artifact links in all three report
  types resolved locally; the Breakout inline JavaScript passed `node --check`.
- Direct browser rendering was **not verified**: the browser tool rejected local
  `file:` URL navigation under its security policy. No workaround was used. Static
  structure, links, script syntax, and the separately rendered SVG were checked.

The CLI pilots above used hosted clients through existing sign-ins. No Fireworks
API call, new GPU benchmark, or active-profile plugin installation was performed
for this change. The prior GPU experiment is separate.
