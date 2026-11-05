"""Paired, preregistered synthetic editing campaigns; paid calls require opt-in."""
from html import escape
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import time

from . import benchmark_cli, benchmark_providers as providers, benchmark_tasks as tasks, controller, editor_session
from .evidence import digest, new_directory, read_json, write_json
from .legacy import ROOT, helper

CONTROLS = ("feasible", "no-op", "random")
_NO_PARAMS = object()
PROMPT = """Repair the supplied neural classifier by using only the supplied weight-editing tools.
First read describe and inspect. Read tensor values and practice inputs as needed.
Use the declared editable entries and budgets; all patches are relative to the original.
You may submit a no-change patch to obtain labeled baseline feedback; it consumes
one proposal and one feedback call. Each feedback returns predictions/labels for
the same practice set. Protect initially correct predictions while repairing errors.
Select a candidate with completed feedback using freeze, or explicitly abstain.
Freezing ends editing before a separate held-out evaluation. Do not request final
scores or external tools. Provide a short reason in each patch. No shell, filesystem,
web, training, or out-of-band evaluator is available. You have no access to controls.
"""


def _validate(config):
    if set(config) != {"models", "families", "task_seeds", "budgets", "limits"}:
        raise ValueError("Configuration requires models, families, task_seeds, budgets, limits only")
    models, families, seeds, budgets, limits = (config[k] for k in ("models", "families", "task_seeds", "budgets", "limits"))
    if not isinstance(models, list) or not 1 <= len(models) <= 20:
        raise ValueError("Declare 1..20 editors")
    ids = set()
    for model in models:
        if set(model) != {"id", "provider", "model", "model_revision", "settings", "input_usd_per_million", "output_usd_per_million"}:
            raise ValueError("Model configuration fields differ from the documented schema")
        if (not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", model["id"]) or model["id"].startswith("control-") or model["id"] in ids
                or model["provider"] not in {*providers.KEYS, *benchmark_cli.PROVIDERS}):
            raise ValueError("Editor IDs must be unique safe names; choose a supported API or CLI provider")
        ids.add(model["id"])
        for key in ("model", "model_revision"):
            if not isinstance(model[key], str) or not 1 <= len(model[key]) <= 256:
                raise ValueError("Retain bounded model and revision identifiers")
        cli = model["provider"] in benchmark_cli.PROVIDERS
        if not isinstance(model["settings"], dict) or set(model["settings"]) - ({"reasoning_effort"} if cli else {"temperature"}):
            raise ValueError("API settings allow temperature; CLI settings allow reasoning_effort")
        if "reasoning_effort" in model["settings"] and model["settings"]["reasoning_effort"] not in ("low", "medium", "high"):
            raise ValueError("CLI effort must be low, medium or high")
        if "temperature" in model["settings"] and (type(model["settings"]["temperature"]) not in (int, float) or not 0 <= model["settings"]["temperature"] <= 2):
            raise ValueError("Temperature must be finite in [0,2]")
        for key in ("input_usd_per_million", "output_usd_per_million"):
            if type(model[key]) not in (int, float) or not math.isfinite(model[key]) or (model[key] != 0 if cli else model[key] <= 0):
                raise ValueError("API token prices must be finite and positive; CLI price placeholders must be zero")
    if (not isinstance(families, list) or not families or len(families) != len(set(families)) or set(families) - set(tasks.FAMILIES)
            or not isinstance(seeds, list) or not 1 <= len(seeds) <= 25 or any(type(s) is not int or not 0 <= s <= 2**32-1 for s in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Declare unique task families and 1..25 unique integer seeds")
    bounds = {"proposals": (1, 50), "feedback_calls": (1, 20), "tool_requests": (1, 256),
              "api_calls": (1, 32), "max_output_tokens": (1, 16384), "max_request_bytes": (4096, 1048576)}
    if set(budgets) != set(bounds) or any(type(budgets[k]) is not int or not lo <= budgets[k] <= hi for k, (lo, hi) in bounds.items()):
        raise ValueError("Invalid per-trial budgets")
    cases = len(families) * len(seeds)
    calls = sum(m["provider"] in providers.KEYS for m in models) * cases * budgets["api_calls"]
    cli_calls = sum(m["provider"] in benchmark_cli.PROVIDERS for m in models) * cases * budgets["api_calls"]
    if (set(limits) - {"max_total_api_calls", "max_total_output_tokens", "max_estimated_usd", "max_total_cli_calls"}
            or not {"max_total_api_calls", "max_total_output_tokens", "max_estimated_usd"} <= set(limits)
            or type(limits["max_total_api_calls"]) is not int or limits["max_total_api_calls"] < calls
            or type(limits["max_total_output_tokens"]) is not int or limits["max_total_output_tokens"] < calls * budgets["max_output_tokens"]
            or type(limits["max_estimated_usd"]) not in (int, float) or not math.isfinite(limits["max_estimated_usd"]) or limits["max_estimated_usd"] <= 0):
        raise ValueError("Global call/output limits must cover every predeclared trial; declare a positive estimated-dollar guard")
    if cli_calls and (type(limits.get("max_total_cli_calls")) is not int or limits["max_total_cli_calls"] < cli_calls):
        raise ValueError("Declare max_total_cli_calls covering all CLI decision rounds")
    json.dumps(config, allow_nan=False)


def _artifacts(path):
    return {p.relative_to(path).as_posix(): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def create(output, config):
    _validate(config)
    root = new_directory(output)
    (root / "private").mkdir()
    (root / "trials").mkdir()
    code = [ROOT / "lab.py", *sorted((ROOT / "weight_space_lab").glob("*.py")), ROOT / "skills/weight-space-lab/scripts/weight_lab.py", ROOT / "LICENSE"]
    hashes = {}
    for path in code:
        name = path.relative_to(ROOT).as_posix()
        target = root / "private/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[name] = digest(target)
    cases, trials = [], []
    editors = [{"id": "control-" + name, "provider": "control", "model": name} for name in CONTROLS] + config["models"]
    for family in config["families"]:
        for seed in config["task_seeds"]:
            final_seed = secrets.randbits(64)
            spec = tasks.generate(family, seed, final_seed)
            case = {"id": f"{family}-{seed}", "family": family, "seed": seed, "heldout_seed": final_seed,
                    "random_control_seed": secrets.randbits(64), "control_value": spec["control_value"]}
            cases.append(case)
            expected = None
            for editor in editors:
                name = f"{editor['id']}/{case['id']}"
                path = root / "trials" / name
                controller.create(path, proposal_budget=config["budgets"]["proposals"], feedback_budget=config["budgets"]["feedback_calls"],
                    actor_type="manual-validation" if editor["provider"] == "control" else "external-agent", editor_id=editor["id"], _task=spec)
                matched = {k: digest(path / v) for k, v in {"source": "editor/original.npz", "architecture": "editor/architecture.json",
                    "practice": "private/practice.json", "heldout": "private/heldout.json"}.items()}
                if expected is not None and matched != expected:
                    raise ValueError("Task instances are not byte-matched across editors")
                expected = matched
                trials.append({"id": name, "editor_id": editor["id"], "provider": editor["provider"], "case_id": case["id"],
                    "family": family, "path": "trials/" + name, "matched": matched,
                    "transport": benchmark_cli.TRANSPORT if editor["provider"] in benchmark_cli.PROVIDERS else "scripted" if editor["provider"] == "control" else "native-api-tools"})
    write_json(root / "private/cases.json", cases)
    initial = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file() and p.name != "state.json"}
    plan = {"schema": "weight-space-benchmark-v1", "config": config, "trials": trials, "cases": [c["id"] for c in cases],
            "code_sha256": hashes, "initial_artifacts": initial, "prompt": PROMPT,
            "interpretation": "Original synthetic one-entry MLP editing; no claim about general transformer editing. Controls are scripted and privileged where labeled."}
    write_json(root / "plan.json", plan)
    controller._save(root, {"schema": "weight-space-benchmark-state-v1", "phase": "prepared", "plan_sha256": digest(root / "plan.json"),
        "api_calls_reserved": 0, "cli_calls_reserved": 0, "cli_usage": {"input_tokens": 0, "output_tokens": 0},
        "output_tokens_reserved": 0, "estimated_usd_reserved": 0., "usage": {"input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.}, "outcomes": {}})
    return summarize(root)


def _load(root, runtime=False):
    plan, state = read_json(root / "plan.json"), read_json(root / "state.json")
    if state["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Benchmark plan changed")
    for name, expected in plan["initial_artifacts"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Benchmark input/source changed: {name}")
    if runtime:
        for name, expected in plan["code_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Benchmark implementation changed; use retained code or create a new campaign")
    reservations, cli_reservations, usage = [], [], {"input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.}
    cli_usage = {"input_tokens": 0, "output_tokens": 0}
    for trial in plan["trials"]:
        if (root / trial["path"] / "sessions/benchmark/header.json").exists() and trial["id"] not in state["outcomes"]:
            raise ValueError("Missing outcome for a started benchmark session")
        for reservation_file in sorted((root / trial["path"] / "provider-calls").glob("*/reservation.json")):
            reservation = read_json(reservation_file)
            cli = reservation.get("transport") == "cli"
            (cli_reservations if cli else reservations).append(reservation)
            usage_file = reservation_file.parent / "usage.json"
            if usage_file.exists():
                logged = read_json(usage_file)
                totals = cli_usage if cli else usage
                for key in totals:
                    totals[key] += logged[key]
    if (state["api_calls_reserved"] != len(reservations)
            or state["output_tokens_reserved"] != sum(r["max_output_tokens"] for r in reservations)
            or not math.isclose(state["estimated_usd_reserved"], sum(r["estimated_usd_reserved"] for r in reservations), abs_tol=1e-10)
            or any(not math.isclose(state["usage"][key], usage[key], abs_tol=1e-10) for key in usage)
            or state.get("cli_calls_reserved", 0) != len(cli_reservations)
            or state.get("cli_usage", {"input_tokens": 0, "output_tokens": 0}) != cli_usage):
        raise ValueError("API budget counters disagree with retained request/usage records; preserve interrupted reservations")
    return plan, state


def _metadata(model):
    control = model["provider"] == "control"
    return {"editor_id": model["id"], "provider": model["provider"], "model": model["model"],
        "model_revision": "prewritten-control-v1" if control else model["model_revision"],
        "settings": {"tool_policy": benchmark_cli.TRANSPORT if model["provider"] in benchmark_cli.PROVIDERS else "bounded-native-tools-v1", **model.get("settings", {})},
        "prompt": "Prewritten infrastructure control, not external-agent discovery. " + PROMPT if control else PROMPT}


def _tool(session, name, params=_NO_PARAMS):
    if not session.active:
        raise ValueError("Session tool budget is closed")
    wire = json.dumps({"id": session.requests + 1, "method": name, "params": {} if params is _NO_PARAMS else params}, allow_nan=False).encode()
    return session.process(wire[:editor_session.MAX_LINE_BYTES + 1])


def _control(session, name, case):
    import numpy as np
    _tool(session, "describe")
    original = _tool(session, "inspect")["result"]["checkpoint"]["sha256"]
    edit = session.protocol["editable"]
    current = helper("weight_lab").tensor_slice(session.run / "editor/original.npz", edit["tensor"], start=edit["flat_indices"][0], count=1)["values"][0]
    rng = np.random.default_rng(case["random_control_seed"])
    count = min(session.protocol["budgets"]["proposals"], session.protocol["budgets"]["feedback_calls"]) if name == "random" else 1
    best = None
    for _ in range(count):
        if not session.active:
            break
        value = case["control_value"] if name == "feasible" else current if name == "no-op" else float(rng.uniform(-3, 3))
        proposed = _tool(session, "propose", {"patch": {"base_sha256": original, "reason": f"Scripted {name} control; not a model result.",
            "operations": [{"op": "set", "tensor": edit["tensor"], "indices": edit["flat_indices"], "values": [value]}]}})
        if "error" in proposed or not session.active:
            continue
        candidate = proposed["result"]["candidate_id"]
        feedback = _tool(session, "feedback", {"candidate_id": candidate})
        if "result" in feedback:
            key = (feedback["result"]["candidate_accuracy"], -len(feedback["result"]["regressed_ids"]))
            if best is None or key > best[0]:
                best = key, candidate
    if session.active and best:
        _tool(session, "freeze", {"candidate_id": best[1]})


def _reserve(root, state, config, model, body):
    cli = model["provider"] in benchmark_cli.PROVIDERS
    if state.get("billing_anomaly") and not cli:
        raise ValueError("Provider usage anomaly stopped further paid calls; preserve the campaign")
    encoded = json.dumps(body, allow_nan=False).encode()
    if len(encoded) > config["budgets"]["max_request_bytes"]:
        raise ValueError("API request exceeds its declared byte limit")
    if cli:
        if state.get("cli_calls_reserved", 0) >= config["limits"]["max_total_cli_calls"]:
            raise ValueError("Global CLI decision-round budget exhausted")
        state["cli_calls_reserved"] = state.get("cli_calls_reserved", 0) + 1
        controller._save(root, state)
        return {"transport": "cli", "cli_call": state["cli_calls_reserved"], "max_output_tokens": None,
                "estimated_usd_reserved": None, "billing": "Existing CLI sign-in; account allowance/cost not metered by the API-dollar guard"}
    tokens = config["budgets"]["max_output_tokens"]
    # Deliberately conservative estimate, not an invoice guarantee or tokenizer.
    estimated = ((2*len(encoded) + 4096) * model["input_usd_per_million"] + tokens * model["output_usd_per_million"]) / 1e6
    limits = config["limits"]
    if (state["api_calls_reserved"] + 1 > limits["max_total_api_calls"] or state["output_tokens_reserved"] + tokens > limits["max_total_output_tokens"]
            or state["estimated_usd_reserved"] + estimated > limits["max_estimated_usd"]):
        raise ValueError("Global API reservation budget exhausted")
    state["api_calls_reserved"] += 1
    state["output_tokens_reserved"] += tokens
    state["estimated_usd_reserved"] += estimated
    controller._save(root, state)  # Reserve before any potentially billable request. Never refund/retry unknown calls.
    return {"api_call": state["api_calls_reserved"], "max_output_tokens": tokens, "estimated_usd_reserved": estimated}


def _api(root, state, config, model, session):
    conversation = providers.Conversation(model, PROMPT)
    (session.run / "provider-calls").mkdir()
    for number in range(config["budgets"]["api_calls"]):
        body = conversation.request(config["budgets"]["max_output_tokens"])
        reservation = _reserve(root, state, config, model, body)
        folder = session.run / "provider-calls" / f"{number+1:03d}"
        folder.mkdir()
        write_json(folder / "reservation.json", reservation)
        write_json(folder / "request.json", body)  # No headers or credentials.
        started = time.monotonic()
        try:
            cli = model["provider"] in benchmark_cli.PROVIDERS
            raw = benchmark_cli.send(model, body, folder) if cli else providers.send(model["provider"], body)
            write_json(folder / "response.json", raw)
            usage = providers.usage(model["provider"], raw)
            cost = (usage["input_tokens"] * model["input_usd_per_million"] + usage["output_tokens"] * model["output_usd_per_million"]) / 1e6
            choices = raw.get("choices")
            truncated = raw.get("stop_reason") == "max_tokens" or (isinstance(choices, list) and
                any(isinstance(c, dict) and c.get("finish_reason") == "length" for c in choices))
            write_json(folder / "usage.json", {**usage, "estimated_usd": cost, "seconds": time.monotonic()-started, "provider_model": raw.get("model"), "output_truncated": truncated})
            totals = state["cli_usage"] if cli else state["usage"]
            for key in ("input_tokens", "output_tokens"):
                totals[key] += usage[key]
            if not cli:
                state["usage"]["estimated_usd"] += cost
            controller._save(root, state)
            if not cli and (usage["output_tokens"] > reservation["max_output_tokens"] or cost > reservation["estimated_usd_reserved"]):
                state["billing_anomaly"] = True
                controller._save(root, state)
                raise ValueError("Provider usage exceeded the reservation estimate; stop and investigate")
            if raw.get("cli_decode_error"):
                raise ValueError("CLI output violated the structured-action or native-tool policy; usage retained")
            calls, _ = conversation.receive(raw)
        except Exception as exc:
            write_json(folder / "failure.json", {"exception": type(exc).__name__, "note": "No retry. Reservation remains consumed; provider billing may be unknown."})
            raise
        if not calls:
            session.reason = "model_stopped_without_selection"
            return
        replies = []
        for call in calls:
            if not session.active:
                break
            reply = _tool(session, call["name"], call["params"])
            replies.append((call["id"], reply))
        if not session.active:
            return
        conversation.results(replies)
    session.reason = "api_call_limit"


def _trial(root, state, config, model, trial, case):
    path = root / trial["path"]
    session = editor_session.Session(path, "benchmark", _metadata(model), config["budgets"]["tool_requests"])
    failure = None
    try:
        if model["provider"] == "control":
            _control(session, model["model"], case)
        else:
            _api(root, state, config, model, session)
    except Exception as exc:
        failure = type(exc).__name__
        session.reason = "execution_error"
    finally:
        session.finish()
    editor_session.audit(path, "benchmark")
    phase = controller.status(path)["phase"]
    if phase == "frozen":
        controller.finalize(path)  # Outside the ended editor session, even for hosted editors.
    elif phase == "editing":
        controller.close(path, "Benchmark ended without selection: " + session.reason)
    controller.audit(path)
    return {"status": "execution_error" if failure else "completed" if (path / "final/result.json").exists() else "no_selection",
            "exception": failure, "end_reason": session.reason}


def run(output, allow_paid_api=False, allow_cli=False):
    root = Path(output)
    with controller._lock(root):
        plan, state = _load(root, runtime=True)
        if state["phase"] not in ("prepared", "controls_complete", "editors_pending"):
            raise ValueError("Campaign already ran or was interrupted; preserve it and create a separately declared campaign")
        if allow_paid_api:
            for model in plan["config"]["models"]:
                if model["provider"] in benchmark_cli.PROVIDERS:
                    continue
                if not os.environ.get(providers.KEYS[model["provider"]]):
                    raise ValueError("Missing " + providers.KEYS[model["provider"]])
        if allow_cli:
            versions = {m["provider"]: benchmark_cli.preflight(m["provider"]) for m in plan["config"]["models"] if m["provider"] in benchmark_cli.PROVIDERS}
            if state.get("cli_versions", versions) != versions:
                raise ValueError("CLI versions changed; preserve the campaign and create a new one")
            state["cli_versions"] = versions
        models = {m["id"]: m for m in plan["config"]["models"]}
        models.update({"control-" + name: {"id": "control-" + name, "provider": "control", "model": name} for name in CONTROLS})
        cases = {c["id"]: c for c in read_json(root / "private/cases.json")}
        state["phase"] = "running"
        controller._save(root, state)
        try:
            for trial in plan["trials"]:
                enabled = trial["provider"] == "control" or (allow_cli if trial["provider"] in benchmark_cli.PROVIDERS else allow_paid_api)
                if trial["id"] in state["outcomes"] or not enabled:
                    continue
                state["outcomes"][trial["id"]] = {"status": "started"}
                controller._save(root, state)
                try:
                    outcome = _trial(root, state, plan["config"], models[trial["editor_id"]], trial, cases[trial["case_id"]])
                except Exception as exc:
                    outcome = {"status": "invalid_or_failed", "exception": type(exc).__name__}
                outcome["artifacts"] = _artifacts(root / trial["path"])
                state["outcomes"][trial["id"]] = outcome
                controller._save(root, state)
            state["phase"] = ("complete" if all(t["id"] in state["outcomes"] for t in plan["trials"]) else "editors_pending") if (allow_paid_api or allow_cli) else "controls_complete"
            controller._save(root, state)
        except BaseException:
            state["phase"] = "interrupted"
            controller._save(root, state)
            raise
    return summarize(root)


def summarize(output):
    root = Path(output)
    plan, state = _load(root)
    rows = []
    for trial in plan["trials"]:
        path = root / trial["path"]
        outcome = state["outcomes"].get(trial["id"], {"status": "pending"})
        for name, expected in outcome.get("artifacts", {}).items():
            if digest(path / name) != expected:
                raise ValueError("Completed trial artifact changed; no scorecard can be trusted")
        result = read_json(path / "final/result.json") if outcome["status"] == "completed" else None
        status = controller.status(path)
        row = {k: trial[k] for k in ("id", "editor_id", "provider", "case_id", "family")}
        row["transport"] = trial.get("transport", "cli-json-actions-v1" if trial["provider"] in benchmark_cli.PROVIDERS else "scripted" if trial["provider"] == "control" else "native-api-tools")
        row.update(status=outcome["status"], success=bool(result and result["success_rule_met"]),
            end_reason=outcome.get("end_reason"), exception=outcome.get("exception"),
            candidate_accuracy=result["comparison"]["candidate_accuracy"] if result else None,
            repair_accuracy=result["repair_group_accuracy"] if result else None,
            fixed=len(result["comparison"]["fixed_ids"]) if result else None, regressed=len(result["comparison"]["regressed_ids"]) if result else None,
            proposals=status["proposals_used"], feedback_calls=status["feedback_used"])
        logged_usage = [read_json(p) for p in (path / "provider-calls").glob("*/usage.json")]
        reserved = [read_json(p) for p in (path / "provider-calls").glob("*/reservation.json")]
        row["api_calls"] = sum(r.get("transport") != "cli" for r in reserved)
        row["cli_calls"] = sum(r.get("transport") == "cli" for r in reserved)
        row["known_usage"] = {key: sum(u[key] for u in logged_usage) for key in ("input_tokens", "output_tokens", "estimated_usd", "seconds")}
        if trial["provider"] in benchmark_cli.PROVIDERS:
            row["known_usage"]["estimated_usd"] = None
        row["output_truncations"] = sum(u["output_truncated"] for u in logged_usage)
        if (path / "sessions/benchmark/summary.json").exists():
            row["tool_requests"] = read_json(path / "sessions/benchmark/summary.json")["requests"]
            journal = [json.loads(line) for line in (path / "sessions/benchmark/events.jsonl").read_text(encoding="utf-8").splitlines()]
            row["tool_errors"] = sum(e["kind"] == "response" and "error" in e["response"] for e in journal)
        rows.append(row)
    def aggregate(editor_id):
        group = [r for r in rows if r["editor_id"] == editor_id]
        measured = [r for r in group if r["candidate_accuracy"] is not None]
        finished = sum(r["status"] not in ("pending", "started") for r in group)
        cli = group[0]["provider"] in benchmark_cli.PROVIDERS
        return {"editor_id": editor_id, "transport": group[0]["transport"], "planned": len(group), "finished": finished,
                "ranking_eligible": all(r["status"] in ("completed", "no_selection") for r in group),
                "successes": sum(r["success"] for r in group), "success_rate_over_planned": sum(r["success"] for r in group)/len(group) if finished == len(group) else None,
                "mean_accuracy_completed_only": sum(r["candidate_accuracy"] for r in measured)/len(measured) if measured else None,
                "mean_repair_accuracy_completed_only": sum(r["repair_accuracy"] for r in measured)/len(measured) if measured else None,
                "regressions_completed_only": sum(r["regressed"] for r in measured),
                "regression_rate_completed_only": sum(r["regressed"] for r in measured)/(128*len(measured)) if measured else None,
                "per_family": {family: {"planned": sum(r["family"] == family for r in group),
                    "successes": sum(r["family"] == family and r["success"] for r in group)} for family in plan["config"]["families"]},
                "feedback_calls": sum(r["feedback_calls"] for r in group), "tool_requests": sum(r.get("tool_requests", 0) for r in group),
                "tool_errors": sum(r.get("tool_errors", 0) for r in group),
                "api_calls": sum(r["api_calls"] for r in group),
                "cli_calls": sum(r["cli_calls"] for r in group),
                "output_truncations": sum(r["output_truncations"] for r in group),
                "known_usage": {key: (None if key == "estimated_usd" and cli else sum(r["known_usage"][key] for r in group)) for key in ("input_tokens", "output_tokens", "estimated_usd", "seconds")},
                "statuses": {s: sum(r["status"] == s for r in group) for s in sorted({r["status"] for r in group})}}
    external = [aggregate(m["id"]) for m in plan["config"]["models"]]
    paired = []
    for i, a in enumerate(external):
        for b in external[i+1:]:
            comparable = a["ranking_eligible"] and b["ranking_eligible"] and a["transport"] == b["transport"]
            pairs = [(next(r for r in rows if r["editor_id"] == a["editor_id"] and r["case_id"] == case),
                      next(r for r in rows if r["editor_id"] == b["editor_id"] and r["case_id"] == case)) for case in plan["cases"]]
            paired.append({"a": a["editor_id"], "b": b["editor_id"], "comparable": comparable,
                "only_a_succeeded": sum(x["success"] and not y["success"] for x, y in pairs) if comparable else None,
                "only_b_succeeded": sum(y["success"] and not x["success"] for x, y in pairs) if comparable else None,
                "both_succeeded": sum(x["success"] and y["success"] for x, y in pairs) if comparable else None,
                "neither_succeeded": sum(not x["success"] and not y["success"] for x, y in pairs) if comparable else None})
    return {"schema": "weight-space-benchmark-scorecard-v1", "phase": state["phase"], "models": external,
            "paired_comparisons": paired,
            "controls": [aggregate("control-" + name) for name in CONTROLS], "trials": rows,
            "external_comparison_complete": all(g["finished"] == g["planned"] for g in external),
            **{k: state[k] for k in ("api_calls_reserved", "output_tokens_reserved", "estimated_usd_reserved", "usage")},
            "cli_calls_reserved": state.get("cli_calls_reserved", 0), "cli_usage": state.get("cli_usage", {"input_tokens": 0, "output_tokens": 0}),
            "limits": "Synthetic tasks only. Pending/error/abstention trials remain in the planned denominator. Completed-only accuracy is secondary. Controls are not API model results. Dollar estimates are not invoices or provider billing guarantees."}


def audit(output):
    root = Path(output)
    plan, state = _load(root)
    checked = 0
    for trial in plan["trials"]:
        path = root / trial["path"]
        controller.audit(path)
        outcome = state["outcomes"].get(trial["id"])
        if outcome and "artifacts" in outcome:
            for name, expected in outcome["artifacts"].items():
                if digest(path / name) != expected:
                    raise ValueError("Completed trial artifact changed")
            if (path / "sessions/benchmark/summary.json").exists():
                editor_session.audit(path, "benchmark")
        checked += 1
    return {"verified": True, "trials_checked": checked, "phase": state["phase"], "external_comparison_complete": summarize(root)["external_comparison_complete"],
            "integrity": "Unsigned local records; provider identity, billing and server-side revision are not independently authenticated."}


def report(output):
    root = Path(output)
    audit(root)
    summary = summarize(root)
    from .reports import page
    body = '<p class="eyebrow">WEIGHT SPACE LAB · SYNTHETIC EDITING SUITE</p><h1>Weight-editing scorecard</h1>'
    body += '<p>Matched task instances and held-out draws; one frozen candidate per trial. Scripted controls are separate from API editors.</p>'
    for label, groups in (("Model editors", summary["models"]), ("Scripted controls", summary["controls"])):
        body += '<h2>' + label + '</h2><table><tr><th>Editor</th><th>Finished / planned</th><th>Success / planned</th><th>Accuracy (completed only)</th><th>Tool errors / requests</th><th>Feedback calls</th></tr>'
        for group in groups:
            accuracy = '—' if group["mean_accuracy_completed_only"] is None else f'{group["mean_accuracy_completed_only"]:.1%}'
            success = f'{group["successes"]}/{group["planned"]}' if group["finished"] else 'Pending'
            body += f'<tr><td>{escape(group["editor_id"])}<br><small>{escape(group["transport"])}</small></td><td>{group["finished"]}/{group["planned"]}</td><td>{success}</td><td>{accuracy}</td><td>{group["tool_errors"]}/{group["tool_requests"]}</td><td>{group["feedback_calls"]}</td></tr>'
        body += '</table>'
    body += '<p>Pending or transport-error groups are not eligible for ranking. Rates over planned trials are withheld until all trial slots finish. No significance claim is made from this small synthetic suite.</p>'
    body += '<h2>Paired success counts</h2><pre>' + escape(json.dumps(summary["paired_comparisons"], indent=2)) + '</pre>'
    body += '<h2>Every planned trial</h2><pre>' + escape(json.dumps(summary["trials"], indent=2)) + '</pre>'
    body += '<h2>Usage and interpretation</h2><pre>' + escape(json.dumps({k: summary[k] for k in ("usage", "api_calls_reserved", "cli_calls_reserved", "cli_usage", "output_tokens_reserved", "estimated_usd_reserved", "limits")}, indent=2)) + '</pre>'
    (root / "scorecard.html").write_text(page("Weight Space Lab — editing benchmark", body), encoding="utf-8")
    # Regenerable derived output, unlike immutable experiment records.
    (root / "scorecard.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding="utf-8")
    return {"report": str(root / "scorecard.html"), "external_comparison_complete": summary["external_comparison_complete"]}
