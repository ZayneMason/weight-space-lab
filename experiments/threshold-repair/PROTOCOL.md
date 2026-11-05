# Bounded threshold repair: independent experiment protocol

**Status:** Controller and manual controls implemented. An external-agent trial has
not been executed. This experiment is independent of the attributed Breakout and
Qwen reproductions. It makes no claim of a new general model-editing method.

## Research question and falsifiable outcome

Can an external editor repair a shifted decision threshold with at most one
changed bias entry, five proposed candidates and three practice-feedback calls,
while introducing no regressions on protected examples?

The model is deliberately small: a one-input, two-logit classifier. A feasible
positive control exists analytically. The study measures an editor's ability to
form and select an effective edit under explicit information/feedback constraints,
not whether this architecture can represent the task. A failure after exhausting
the budget is a limitation of that editor/protocol combination, not proof that no
repair exists or that language models cannot edit weights.

The final success criterion is declared before editing:

1. At least 90% accuracy on the held-out repair region.
2. Zero protected-region regressions relative to the untouched baseline.
3. Zero overall regressions across the complete final set.

Accuracy, fixed IDs, regressed IDs and candidate failures remain visible even when
the criterion is not met. Report every run and every candidate; do not report only
the best trial. Repeated independent trials require a separately declared trial
count, editor settings and aggregation plan.

## Data and constraints

- Source logits are `[-x, x - 0.6]`; only flat index 1 of `bias` may be set, added
  to, or scaled. At most one of the four stored entries may differ.
- Practice data have 32 examples: eight each from repair, protected, positive-anchor
  and negative-anchor regions. Inputs are visible in the editor workspace; labeled
  predictions are returned only through metered controller feedback.
- Every feedback request, including a repeat or failed evaluation, consumes one
  call and 32 example exposures. Rejected/malformed proposals consume proposal budget.
- The final set has 256 independent draws from the same four predefined regions.
  A fresh private seed and dataset-hash commitment are recorded before editing;
  the seed and labels remain in `private/`. Retained files permit exact repetition.
- No gradient training or out-of-band model execution is part of the protocol.
- The source, architecture, protocol, data, software versions and evaluator code
  are hash-bound. Changing them requires a new run.

## Controller interface

Install the CPU dependencies from `requirements.txt`. No hosted model is invoked
by any controller command.

```bash
python lab.py experiment create runs/threshold-study --proposal-budget 5 --feedback-budget 3
python lab.py experiment status runs/threshold-study
```

The default actor type is `manual-validation`. For an actual externally managed
editor, create a fresh run with `--actor-type external-agent --editor-id local-editor-01`;
that identifier is caller-declared, not authenticated evidence of a model run.
The operator must separately retain exact editor model/revision, prompt, settings,
tool restrictions, transcript and any allowance usage. Do not expose manual positive
controls or this researcher protocol to a purported blind editor session.

The [recorded JSONL editor bridge](../../docs/EDITOR_SESSIONS.md) supplies an
allowlisted tool interface and retained prompt/configuration/tool records. It
preserves global budgets across sessions and excludes final evaluation. External
isolation, authenticated provider provenance and usage records remain operator
responsibilities. The bundled stdio check is prewritten validation only.

After an editor writes a patch file using the source hash and a reason:

```bash
python lab.py experiment submit runs/threshold-study proposed-patch.json
python lab.py experiment feedback runs/threshold-study c001
python lab.py experiment freeze runs/threshold-study c001
python lab.py experiment finalize runs/threshold-study
python lab.py experiment audit runs/threshold-study
```

`c001` denotes the first proposal; use the accepted ID returned by the controller.
Only a candidate with completed practice feedback can be frozen. Freezing copies
and hashes it, records the selection and budgets, and closes editing/feedback.
The final evaluator accepts no alternate candidate and runs the frozen candidate
and preserved baseline on identical final examples and settings. A second final
attempt is denied. The final HTML report and all JSON records remain under `final/`.

If an editor selects no candidate, record that outcome without running the final
set: `python lab.py experiment close runs/threshold-study --reason "Editor abstained"`.
The record distinguishes exhausted budget from early termination. It makes no
final capability claim and prevents reopening the trial for more feedback.

## Manual infrastructure verification

```bash
python scripts/check_controller.py runs/controller-check
```

This deliberately submits a regressing control and an analytically feasible
control, records both feedback results, freezes the feasible one, and runs the
matched final evaluation. It validates accounting and measurement only. It is not
agent discovery and must never be counted as an external-agent trial.

## Trust boundary and recovery

The controller enforces logical quotas, phase transitions, code/input integrity,
candidate identity, exclusive writers, and an unsigned event hash chain. It is
**not an OS sandbox**. An agent with unrestricted filesystem or shell access could
read private data or evaluate outside the budget. A credible agent study must
expose only `editor/` and a constrained controller tool, with operator-supplied
filesystem/compute isolation and retained tool audit records.

An interrupted action keeps its budget reservation and pending state. Failed freeze
or final attempts are terminal and retain partial files and failure records. A
stale exclusive lock requires operator investigation, never automatic unlocking or
budget resetting. Preserve a failed run and start a separately recorded run.

The `audit` command checks retained source snapshots and local records even after
the installed toolkit changes. Continuing an active trial still requires its
original code and dependency versions. Its `verified` flag is not an
independent scientific attestation; `final_complete` separately states whether the
final evaluation completed. No private data or controller runtime directories
should be committed or automatically published.
