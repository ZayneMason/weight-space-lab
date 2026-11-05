# Claude CLI, Codex CLI and a local Fireworks key

The benchmark can use existing Claude Code and Codex CLI sign-ins instead of
requiring Anthropic/OpenAI API keys. Fireworks remains a separate API adapter.
These CLI runs consume account allowance and may have account-specific costs.

## Prepare and run

```bash
python lab.py benchmark create runs/cli-comparison --config experiments/editing-benchmark/cli.example.json
python lab.py benchmark run runs/cli-comparison --allow-cli
python lab.py benchmark audit runs/cli-comparison
```

The example declares Claude CLI, Codex CLI and three Fireworks editors on the same
four task instances. `--allow-cli` runs the two CLI editors plus offline controls;
Fireworks stays pending. The default command without an opt-in runs controls only.
`--allow-paid-api` separately enables the API editors and requires their keys.
Existing completed/failed trial slots are retained and never silently rerun.

Use a fresh campaign when changing configuration, software or trial design. The
example requests `sonnet` for Claude and `gpt-6.1-sol` for Codex; edit these before
preparation if different models are wanted. CLI effort settings allow low, medium
or high. Exact CLI versions are recorded and must remain fixed when continuing a
campaign. Returned model IDs are retained where the CLI provides them; a requested
alias is not proof of a particular server-side deployment revision.

Sign in with the vendors' normal interactive flows if needed:

```bash
claude auth login
codex login
```

A cached “signed in” status can coexist with an expired OAuth session; a failed
live call remains a recorded failure, not a completed editing result.

## Paste a Fireworks key locally on Windows

```powershell
python scripts/fireworks_key.py
```

The dialog masks the key. **Open benchmark terminal** starts a local PowerShell
whose process environment contains `FIREWORKS_API_KEY`. The dialog then clears its
field and restores its own previous environment. No key file, registry entry,
command-line argument or shell-history assignment is created. Closing that
terminal discards this copy. The key must not be pasted into chat or source files.
This is session-only input, not a persistent credential manager, and the running
assistant process does not gain the key.

The terminal prints preparation/run commands but starts no benchmark or provider
request automatically. When API spending is separately authorized, the API command
must run in that terminal (or a child) to inherit its key. Other existing terminals
do not receive it. A console-only masked-input alternative is:

```powershell
python scripts/fireworks_key.py --console
```

On other platforms, use the normal environment-variable mechanism in a local
terminal; the GUI helper is Windows-only. The API example's estimated dollar guard
is configuration, not authorization to spend.

## How the CLI adapters use the tools

The CLI is the decision-maker; the operator-side benchmark remains the only
executor of weight tools. Each round starts a fresh noninteractive CLI process
in an empty temporary working directory and sends the tool definitions and
recorded conversation through stdin. The CLI returns a JSON batch:

```json
{"actions":[{"method":"inspect","params":{}}]}
```

There can be at most eight actions per round. The existing controller validates
ordinary JSON argument objects directly; tensor actions explicitly supply name,
start and count for the strict schema. There is no JSON-in-a-string argument layer.
The interface revision is recorded as `cli-structured-actions-v2` in the plan.
The controller meters every action, returns practice feedback, and stops at freeze or
abstention. The complete prompt, raw CLI stdout/stderr, exit metadata, normalized
response and reported usage remain in the private trial record. The final held-out
evaluation runs only after the editor session ends. No final data or feasible
control value is placed in the CLI prompt.

Claude runs with built-in tools disabled, an empty strict MCP configuration,
skills/hooks/memory discovery disabled, no session persistence and a two-internal-
turn limit. It retains the normal account sign-in; minimal `--bare` mode is avoided
because the installed CLI documents that mode as API-key-only authentication.
[Claude CLI reference](https://code.claude.com/docs/en/cli-reference).

Codex runs with ignored user config/rules, ephemeral execution, read-only sandbox,
zero project-instruction discovery, web search disabled, and command execution,
plugins/apps, hooks, memories, skill search, browser/computer tools and subagents disabled.
Observed native tool events invalidate a trial. These are invocation overrides;
the user's saved CLI configuration is not rewritten.
[Codex noninteractive mode](https://learn.chatgpt.com/docs/non-interactive-mode),
[Codex configuration reference](https://developers.openai.com/codex/config-reference/).

The supplied stdin prompt is retained, but vendor-generated system context is not
fully observable. These overrides do not independently prove that every form of
client-side context discovery is absent; CLI results remain harness-specific.

The adapters allowlist standard OS/locale/proxy/certificate paths plus the CLI home
directories. API keys, alternate authorization/custom-header overrides, arbitrary
household tokens and environment-based OAuth overrides are not inherited. No
authentication files are copied; each CLI reads its normal on-disk/native-keychain
sign-in. These controls are a logical tool boundary, not a new OS sandbox
or an attestation against a compromised CLI/operator. Version/flag incompatibility
fails the trial instead of falling back to a less restricted invocation.

## Accounting and comparison limits

CLI rounds have a separate `max_total_cli_calls` reservation counter, with the
existing per-editor `api_calls` setting reused as the maximum number of CLI
decision rounds. Every launch reserves a round before execution, and failures
remain consumed. Each process has a 120-second wall timeout and a 4 MiB cap per
stdout/stderr capture, including fast-exiting children. Timeouts kill the owned
process tree. Bounded prefixes and exit metadata are retained on capture failure.

The harness does not retry a failed CLI round. A vendor CLI may make internal
requests or retries; its one process is **not** claimed to equal one API request.
Known reported tokens are retained even when action decoding fails, including
completed usage emitted before a process times out. Missing usage is unknown,
not proof of zero consumption. CLI tokens do not enter the Fireworks/API output-
token reservation or estimated-dollar guard. The CLI price fields must be zero
placeholders and scorecard dollar cost is unavailable, not free.

The CLIs do not expose the same enforceable per-call output cap as these native
API adapters. Their built-in system prompts, internal turns and structured-action
transport also differ. The scorecard labels `cli-structured-actions-v2` separately from
`native-api-tools` and withholds paired ranking across the two transports. The two
CLI editors can be compared under the same outer tool/decision-round limits, but
results remain specific to their CLI harnesses and account settings.
