# Recorded editor sessions

The JSONL bridge connects an externally managed editor to the bounded threshold
repair controller. It records the declared editor identity, prompt, settings,
requests, responses, failures and controller transitions. It invokes no model,
opens no network listener and uses only installed CPU dependencies.

**Status:** A prewritten control verifies the transport. No external-agent study
has been executed. This interface supports `bounded-threshold-repair-v1` and the
four synthetic `bounded-mlp-edit-v1` tasks in the [benchmark](BENCHMARK.md).
It does not evaluate arbitrary transformer edits.

## Try the complete transport locally

After installing `requirements.txt`, run:

```bash
python scripts/check_editor_session.py runs/editor-transport-check
python lab.py editor audit runs/editor-transport-check scripted
python lab.py experiment audit runs/editor-transport-check
```

The script starts the real CLI as a child process, inspects the original checkpoint,
submits a **prewritten** `bias[1]=0` patch, requests one practice feedback call and
freezes the candidate. The editor process exits before the script separately runs
the operator's final evaluation. The final report is
`runs/editor-transport-check/final/report.html`. A known answer validates the
infrastructure; it cannot establish independent editor capability.

## Operator setup for an external harness

Create a fresh run and retain a local metadata file:

```bash
python lab.py experiment create runs/editor-study --actor-type external-agent --editor-id editor-01 --proposal-budget 5 --feedback-budget 3
python lab.py editor serve runs/editor-study session-01 --metadata editor-metadata.json --max-requests 32
```

`editor-metadata.json` has exactly these fields:

```json
{
  "editor_id": "editor-01",
  "provider": "actual-provider",
  "model": "actual-model",
  "model_revision": "actual-revision-or-provider-unavailable",
  "settings": {"temperature": 0, "tool_policy": "stdio-tools-only"},
  "prompt": "The complete initial instructions actually supplied to the editor."
}
```

Use actual values and explicitly record unavailable provider revision information.
Supported settings are `temperature`, `reasoning_effort`, `seed` and `tool_policy`;
values must be bounded JSON scalars. Credentials are not supported fields. These
records are caller declarations, not provider authentication. Retain the provider
conversation, exact tool mapping, model usage/cost and any harness settings outside
this interface as well. A tool journal cannot prove which model generated a patch
or capture reasoning/messages that the harness never sent to it.

The operator launches this process and connects its pipes to the isolated editor's
tool adapter. **Do not give the editor the run directory, researcher protocol,
control scripts, shell access or operator stderr.** The current toolkit supplies
a logical tool boundary, not filesystem, process or network isolation. An actual
blind study requires an operator-supplied container/VM or equivalent isolation.
Starting this command in a shared unrestricted checkout does not provide it.

## Wire contract

The first stdout line is a `hello` object advertising methods and limits. Each
UTF-8 input line must contain exactly `id`, `method` and `params`:

```json
{"id":1,"method":"describe","params":{}}
{"id":2,"method":"inspect","params":{}}
{"id":3,"method":"tensor","params":{"name":"bias","start":0,"count":2}}
```

Each response has the matching `id` and either `result` or `error`. Invalid JSON
may have a null ID. IDs must be unique within a session: a nonempty string of at
most 64 characters or an integer from 0 through 2147483647. No file paths or
arbitrary commands are accepted in any method.

| Method | Parameters | Effect |
| --- | --- | --- |
| `describe` | `{}` | Question, editable entries, budgets, success rule and feedback policy; no control answer or final data. |
| `inspect` | `{}` | Original checkpoint hash/tensor metadata and architecture. |
| `tensor` | `name`, optional `start`, `count` | Bounded values from an existing original tensor, including benchmark hidden-layer tensors. |
| `practice_inputs` | `{}` | Unlabeled practice inputs. |
| `propose` | `patch` | Inline patch with `base_sha256`, `reason`, `operations`; consumes the global proposal allowance, including rejected patches. |
| `feedback` | `candidate_id` | Metered practice evaluation, labels, predictions, fixes and regressions. |
| `freeze` | `candidate_id` | Freeze a candidate with completed feedback and end the session. |
| `abstain` | `reason` | Record no selection, close the trial and end the session. |
| `status` | `{}` | Phase, global quotas and cumulative usage. |

The [CLI reference](../skills/weight-space-lab/references/cli.md) describes patch
syntax; the original experiment restricts edits to `bias` index 1, while each
benchmark task declares its own permitted tensor/index in `describe`. An invalid
tool envelope consumes a transport request; it does not count as a submitted
patch. Invalid candidates reaching the controller consume proposal budget. Global
feedback accounting remains exactly the [protocol](../experiments/threshold-repair/PROTOCOL.md)
rule; denied requests after exhaustion cannot create additional evaluations.

Every input line, including malformed/repeated requests, consumes the session's
request allowance (default 32; allowed range 1–256). Each line is limited to 65,536
bytes including its newline. Oversized input terminates the session immediately;
the journal retains only the bounded prefix actually read, explicitly labeled as
such. It does not claim to hash the unconsumed remainder. EOF and transport limits
stop the session without automatically selecting or closing the experiment.

A fresh session ID can resume an editing trial. Proposal and feedback budgets are
global and never reset. Editor/provider/model/revision/settings are pinned across
sessions; each session retains its own actual prompt, including handoff context.
The bridge source is pinned with the controller code at experiment creation;
changing the installed implementation requires a fresh run or the recorded code.
Only one editor session may be active. Out-of-session controller changes invalidate
the session audit. Mutations check the expected event prefix under the controller
lock; audits also match request candidate IDs and response outcomes to transitions.
Stale locks or incomplete records require operator investigation;
preserve them rather than deleting records and restarting the same session ID.

## Final evaluation and records

There is no final-evaluation method in the editor interface. After the editor has
exited and selected a frozen candidate, the operator runs:

```bash
python lab.py editor audit runs/editor-study session-01
python lab.py experiment finalize runs/editor-study
python lab.py experiment audit runs/editor-study
python lab.py editor audit runs/editor-study session-01
```

Audit **every** session, including reconnects and failures. Finalization compares
the frozen candidate and untouched baseline on identical held-out examples once.
Even failure to create the final output directory consumes that final attempt and
retains a terminal failure record. A session's historical controller anchors remain
verifiable after this separate final evaluation.

Each `sessions/SESSION_ID/` retains `header.json`, `prompt.txt`, the bridge source
and license, `events.jsonl`, copied inline proposal requests and `summary.json`.
The journal binds exact request bytes, responses and controller event prefixes;
the summary hashes retained artifacts. `editor-provenance.json` pins the declared
configuration across sessions. Controller audit and session audit are complementary:
the former validates the experiment; the latter also validates the transport record.

These are unsigned local integrity records. They detect changed/missing artifacts
under the retained ledger; they cannot attest against a party able to rewrite all
records, authenticate a provider, prove isolation or certify scientific validity.
Session files retain private prompts and operator errors and are **not sanitized
publication exports**. Review them before sharing. Runtime directories stay ignored
by Git and distribution packaging. Report the predeclared trial count, all outcomes
and matched baselines; do not count scripted validation as an external-agent result.
