# Comparing models as weight editors

The benchmark measures how API models use Weight Space Lab's tools to repair small
neural classifiers. The **editor** is Claude or a model served by Fireworks. The
**target** is a separate CPU MLP whose numerical parameters the editor changes.

**Status:** Task generators, matched campaigns, provider adapters and scorecards
are implemented. Local controls and provider-shaped fixtures validate the runner.
Native API model trials remain unrun. Limited CLI integration pilots, including
failures, are recorded in [verification](VERIFICATION.md) and use a separate
[CLI transport](CLI_EDITORS.md); they do not establish a model ranking. The example model IDs
and prices were checked on November 4, 2026; availability and pricing can change.

## Run the suite without spending

Install `requirements.txt`, then:

```bash
python lab.py benchmark create runs/editing-suite --config experiments/editing-benchmark/pilot.example.json
python lab.py benchmark run runs/editing-suite
python lab.py benchmark audit runs/editing-suite
```

Open `runs/editing-suite/scorecard.html`. The default run executes **only** the
three scripted baselines on four tasks. The example reserves 16 future API trial
slots (four editor models × four task families × one task seed), which remain
visibly pending. Preparation and offline controls require no API keys or network.
Creation requires a fresh directory; it never overwrites an existing campaign.

For a larger **scripted calibration** across three task seeds and all four families:

```bash
python scripts/check_benchmark.py runs/benchmark-validation
```

It validates 12 feasible controls and 12 unchanged controls, retains the 12 random
controls, audits every planned slot and generates the same scorecard. API slots
remain pending and no network calls occur.

## Tasks and controls

| Family | Target defect | Permitted change |
| --- | --- | --- |
| `threshold` | Shifted decision threshold | One output bias entry |
| `slope` | Attenuated class slope | One input-to-logit weight |
| `routing` | Corrupted contribution from a second feature | One input-to-logit weight |
| `hidden` | Corrupted input routing through a ReLU hidden unit | One hidden-layer weight |

Each task seed varies coefficients and corruption. A fresh private held-out seed
is drawn once per case, then the **same source bytes, architecture, practice set
and held-out set** are supplied to every editor and control. Hashes are checked
at preparation. Public task seeds are variation identifiers, not claims of novel
or contamination-resistant tasks; exact final draws remain private.

Each practice set has 32 examples, balanced between initially wrong (`repair`)
and initially correct (`protected`) predictions. The independent final draw has
256 examples with the same 50/50 stratification. Inputs are sampled from [-2,2];
labels come from the uncorrupted teacher. These artificial strata measure repair
and preservation directly; their accuracy is not a natural population estimate.
Example IDs reveal their strata. This is deliberately a tool/parameter reasoning
test, not a blind inference of which examples were initially wrong.

Every case has a known feasible one-entry repair. Three scripted controls run on
the identical cases and under the same proposal/feedback upper limits:

- `feasible`: privileged access to the operator's known repair value; validates
  reachability and accounting. It is not an agent and has extra information.
- `no-op`: freezes an unchanged checkpoint, establishing the 50% matched baseline.
- `random`: samples up to the minimum of the proposal and feedback limits, uniformly
  from [-3,3] for the permitted entry; selects by practice accuracy, then fewer
  regressions. Its independently drawn private seed is retained; it does not reuse
  the task generator's random stream. This is a declared naive search policy, not
  a universal random bound.

No-change proposals are explicitly enabled for benchmark tasks and consume normal
proposal/feedback budgets. Their receipts record zero changed entries. The generic
CLI and the original threshold protocol keep rejecting no-change patches by default.
An API editor can spend a no-change proposal/feedback call to inspect baseline
labels before deciding on a repair. No gradients or external evaluator are exposed.

## Trial protocol and fair comparison

The frozen configuration declares editors, families, unique task seeds and budgets
before any trial runs. It also retains one common editor prompt and snapshots the
runner, adapters, controller, generators and evaluator. The same initial prompt,
native tool descriptions, example count and proposal/feedback limits apply across
editors. Model-specific native API envelopes differ and are retained exactly.
There are no model-specific hints, automatic repair of failed outputs, fallbacks
to another editor, or automatic HTTP retries.

The pilot allows five proposals, three practice-feedback calls (96 example
exposures), 32 tool requests, eight API turns and 1,024 maximum output tokens per
API call. A turn can contain multiple native tool calls; those run sequentially
and individually consume tool/controller allowances. Unexecuted calls after freeze
are retained in the provider response. A model that ends without freezing records
no selection; it receives no final score. Malformed/rejected calls remain visible.

Freezing requires completed feedback and permanently closes editing. The editor
session ends and passes audit before the operator evaluates the frozen candidate
and untouched baseline on the same final examples once. Success requires at least
90% repair-region accuracy and zero overall regressions (therefore zero protected
regressions). The suite reports continuous outcomes even when that rule fails.

All planned trial slots remain in the denominator. Pending rates are withheld.
API failures, incomplete records and abstentions are not discarded or silently
rerun. Groups with provider/execution failures are excluded from paired ranking;
paired output reports only-A, only-B, both and neither success counts when both
groups are eligible. Completed-only accuracy is explicitly secondary and must not
be used to hide failures. Scorecards include task-level outcomes, repair accuracy,
regressions, feedback use, tool errors, API calls, known token usage and latency.

A one-seed pilot is a smoke test, not a broad model ranking. For a substantive
comparison, predeclare multiple task seeds, sampling settings and an aggregation
plan. Seeds vary the target problem; they do **not** seed provider generation.
Unset sampling/thinking settings mean provider defaults, recorded as such; equal
output caps do not imply equal reasoning effort. API adapters currently forward
only optional `temperature`; omit it for models that reject that setting. In
particular, the checked Claude Sonnet model rejects temperature settings.
[Claude model documentation](https://platform.claude.com/docs/en/models/sonnet-5-5/overview).

## Live API runs are separate opt-in operations

The example lists Claude Sonnet and Fireworks-hosted DeepSeek, Kimi and MiniMax.
Review exact IDs, tool support, current prices and account access before a live
run. Sources: [Claude](https://platform.claude.com/docs/en/models/overview),
[Fireworks catalog](https://fireworks.ai/models?serverless=true),
[DeepSeek](https://fireworks.ai/models/deepseek-ai/deepseek-v4p1-flash),
[Kimi](https://fireworks.ai/models/fireworks/kimi-k3p1-pro),
[MiniMax](https://fireworks.ai/models/fireworks/minimax-m3p1-flash).
Copy the example and change entries to compare other models or additional variants.
The configuration contains environment-variable names indirectly through provider
selection, never API credentials. Model aliases and returned model IDs are retained;
they cannot establish an unavailable exact server-side revision.

When separately authorized, expose `ANTHROPIC_API_KEY` and `FIREWORKS_API_KEY` only
to the operator process, and explicitly opt in:

```bash
python lab.py benchmark run runs/editing-suite --allow-paid-api
python lab.py benchmark audit runs/editing-suite
python lab.py benchmark report runs/editing-suite
```

This can continue a controls-only campaign using its pinned code and environment.
A live campaign cannot be replayed in place; an interrupted campaign is preserved
and requires a separately declared run. Stale locks require investigation. Missing
keys fail before any provider call. Endpoints are fixed HTTPS Claude Messages and
Fireworks Chat Completions; redirects are refused. No model is given shell, file
or network tools. The runner forwards only the advertised bounded tools and never
the run directory, private teacher/control value, final examples or provider key.
Native formats follow [Claude tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)
and [Fireworks tool calling](https://docs.fireworks.ai/guides/function-calling).

## API budgets, usage and private records

Global call and maximum-output reservations must cover every predeclared trial's
upper limits. Each call reserves its full output allowance before HTTP; failures
and timeouts retain reservations and are never retried automatically. Requests
are byte-bounded and responses capped at 4 MiB, with a 90-second HTTP timeout.
Native tool argument JSON is limited to 64 nesting levels; malformed arguments
remain rejected tool requests rather than being silently repaired.
This is a per-request transport timeout, not a whole-campaign wall-clock deadline.

The dollar guard uses supplied input/output prices and a conservative byte-based
input estimate plus maximum output. Reservations are not refunded after short or
uncertain responses. Known reported usage and estimated dollar use are reported
separately. Unexpected usage exceeding a reservation blocks all further paid calls
in that campaign. Output truncations are counted. **This is
an estimated spend guard, not an invoice-level hard cap**: provider tokenization,
pricing changes, caching or additional fees can differ. Use provider/account
spending limits when a hard monetary ceiling is required. The example's $10 guard
is configuration, not authorization to spend.

`plan.json` fixes every trial and its matched input hashes. `private/` retains
held-out seeds, control values and source snapshots. Each trial retains its normal
controller/session records plus credential-free API request bodies, raw responses,
reservations, usage, returned model identity and failure markers. Valid reported
usage is recorded before tool decoding, including turns with malformed tool calls.
The campaign
cross-checks usage counters against those records and hashes completed-trial
artifacts. Root `state.json` records phases and every started/completed outcome.
`scorecard.json` and `scorecard.html` are regenerable derived views, not signed
research attestations. Runtime records include private prompts and held-out data;
they stay Git-ignored and are not publication-ready exports.

The hosted editor sees only the tool messages, but the local runner still needs
normal operator security. It is not an OS sandbox against a process already able
to read the operator's filesystem. Native provider transcripts are retained; model
identity, billing, isolation and scientific validity are not independently attested.
Synthetic one-entry MLP success would establish a narrow tool-use capability,
not general neural programming or successful transformer editing.
