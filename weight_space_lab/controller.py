"""Bounded protocol controller for a small original, independent MLP experiment.

This is a logical experiment gate, not an OS sandbox. An external editor must be
isolated from the private controller directory by the research operator.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil

from .evidence import digest, new_directory, read_json, versions, write_json
from .legacy import ROOT, helper


class ControllerChangedError(RuntimeError):
    """An editor's expected event prefix changed before its action acquired the lock."""


def _require_anchor(state, expected):
    if expected is not None:
        actual = {"events": len(state["events"]), "head": state["events"][-1]["sha256"] if state["events"] else "0" * 64}
        if expected != actual:
            raise ControllerChangedError("Controller changed outside the recorded editor request")


def _save(run, state):
    temp = run / "state.json.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(run / "state.json")


def _event(run, state, action, outcome, **details):
    entry = {"index": len(state["events"]), "action": action, "outcome": outcome,
             "utc": datetime.now(timezone.utc).isoformat(),
             "previous": state["events"][-1]["sha256"] if state["events"] else "0" * 64, **details}
    entry["sha256"] = hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    state["events"].append(entry)
    _save(run, state)


@contextmanager
def _lock(run):
    lock = run / ".controller.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Experiment is locked. Another action may be running; inspect the process/lock before manual recovery.") from None
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def _check(run, state, require_runtime=True):
    if state.get("schema") != "weight-space-controller-v1":
        raise ValueError("Unknown experiment controller schema")
    if require_runtime and state["versions"] != versions(["numpy", "safetensors"]):
        raise ValueError("Evaluator environment changed; use the recorded versions or a new experiment")
    for name, expected in state["input_sha256"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Input hash mismatch: {name}; preserve the run and start a new experiment")
    for name, expected in state["code_sha256"].items():
        if digest(run / "private/code" / name) != expected:
            raise ValueError(f"Recorded source snapshot hash mismatch: {name}")
        if require_runtime and digest(ROOT / name) != expected:
            raise ValueError(f"Controller/evaluator code hash changed: {name}; use the recorded code or a new experiment")
    previous = "0" * 64
    for index, event in enumerate(state["events"]):
        entry = {k: v for k, v in event.items() if k != "sha256"}
        expected = hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if event["index"] != index or event["previous"] != previous or event["sha256"] != expected:
            raise ValueError("Experiment event hash chain is inconsistent")
        previous = expected
    if state["proposals_used"] > state["budgets"]["proposals"] or state["feedback_used"] > state["budgets"]["feedback_calls"]:
        raise ValueError("Recorded counters exceed the protocol budgets")
    if state["budgets"] != read_json(run / "protocol.json")["budgets"]:
        raise ValueError("Controller budgets differ from the frozen protocol")
    submitted = sum(e["action"] == "submit" and e["outcome"] == "started" for e in state["events"])
    feedbacks = sum(e["action"] == "feedback" and e["outcome"] == "started" for e in state["events"])
    if (state["proposals_used"] != submitted or state["feedback_used"] != feedbacks
            or state["feedback_example_exposures"] != feedbacks * 32):
        raise ValueError("Budget counters disagree with the event history")
    phase = "editing"
    for event in state["events"]:
        if event["action"] == "freeze" and event["outcome"] == "started":
            if phase != "editing":
                raise ValueError("Repeated or invalid freeze start in event history")
            phase = "freezing"
        if event["action"] == "freeze" and event["outcome"] == "complete":
            if phase != "freezing":
                raise ValueError("Repeated or invalid freeze transition in event history")
            phase = "frozen"
        if event["action"] == "freeze" and event["outcome"] == "failed":
            if phase != "freezing":
                raise ValueError("Freeze failure has no matching start event")
            phase = "freeze_failed"
        if event["action"] == "final" and event["outcome"] == "started":
            if phase != "frozen":
                raise ValueError("Final evaluation started without a unique freeze")
            phase = "final_running"
        if event["action"] == "final" and event["outcome"] in ("complete", "failed"):
            if phase != "final_running":
                raise ValueError("Final completion has no matching start event")
            phase = "final_complete" if event["outcome"] == "complete" else "final_failed"
        if event["action"] == "close" and event["outcome"] == "complete":
            if phase != "closing":
                raise ValueError("A no-selection close occurred outside editing")
            phase = "closed_without_final"
        if event["action"] == "close" and event["outcome"] == "started":
            if phase != "editing":
                raise ValueError("A no-selection close started outside editing")
            phase = "closing"
        if event["action"] == "close" and event["outcome"] == "failed":
            if phase != "closing":
                raise ValueError("A no-selection close failure has no matching start")
            phase = "close_failed"
    if state["phase"] != phase:
        raise ValueError("Controller phase disagrees with event history")
    for candidate_id, candidate in state["candidates"].items():
        if candidate["status"] == "accepted" and candidate["checkpoint"] != f"proposals/{candidate_id}/candidate/model.npz":
            raise ValueError("Candidate checkpoint path is inconsistent with its proposal ID")
        completed = sum(e["action"] == "feedback" and e["outcome"] == "complete" and e["candidate_id"] == candidate_id for e in state["events"])
        if candidate["feedback_calls"] != completed:
            raise ValueError("Candidate feedback count disagrees with event history")
        for name, expected in candidate["artifacts"].items():
            if digest(run / name) != expected:
                raise ValueError(f"Candidate artifact hash mismatch: {name}")
    for name, expected in state.get("frozen", {}).get("artifacts", {}).items():
        if digest(run / name) != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
    for name, expected in state.get("result_artifacts", {}).items():
        if digest(run / name) != expected:
            raise ValueError(f"Result artifact hash mismatch: {name}")


def _load(run):
    state = read_json(run / "state.json")
    _check(run, state)
    if state.get("pending_action"):
        raise ValueError("A previous action did not finish. Its budget was reserved; preserve the interrupted run and start a new one.")
    return state


def _deny(run, state, action, message):
    _event(run, state, action, "denied", error=message)
    raise ValueError(message)


def _hashes(run, directory):
    return {p.relative_to(run).as_posix(): digest(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def _dataset(seed, per_group, prefix):
    import numpy as np
    rng = np.random.default_rng(seed)
    groups = (("repair", .02, .28, 1), ("protected", -.28, -.02, 0),
              ("positive-anchor", .7, 1.4, 1), ("negative-anchor", -1.4, -.7, 0))
    return {"split": "practice" if prefix == "practice" else "heldout", "examples": [
        {"id": f"{prefix}-{name}-{i}", "x": [float(x)], "y": label}
        for name, low, high, label in groups for i, x in enumerate(rng.uniform(low, high, per_group))]}


def create(output, feedback_budget=3, proposal_budget=5, actor_type="manual-validation", editor_id="unassigned"):
    import numpy as np
    if not 1 <= feedback_budget <= 20 or not 1 <= proposal_budget <= 50:
        raise ValueError("Feedback calls must be 1..20 and proposals 1..50")
    if actor_type not in ("manual-validation", "external-agent") or not editor_id.strip():
        raise ValueError("Specify manual-validation or external-agent and an editor identifier")
    if actor_type == "external-agent" and editor_id == "unassigned":
        raise ValueError("An external-agent run requires an explicit --editor-id; identity remains caller-declared")
    run = new_directory(output)
    for name in ("editor", "private", "proposals", "feedback"):
        (run / name).mkdir()
    source = run / "editor/original.npz"
    np.savez(source, weight=np.array([[-1.], [1.]], dtype=np.float32), bias=np.array([0., -.6], dtype=np.float32))
    architecture = {"layers": [{"weight": "weight", "bias": "bias", "activation": "linear"}]}
    write_json(run / "editor/architecture.json", architecture)
    practice = _dataset(20261104, 8, "practice")
    heldout_seed = secrets.randbits(64)
    write_json(run / "private/practice.json", practice)
    write_json(run / "private/heldout.json", _dataset(heldout_seed, 64, "final"))
    write_json(run / "private/split-seeds.json", {"practice": 20261104, "heldout": heldout_seed})
    write_json(run / "editor/practice-inputs.json", {"examples": [{"id": r["id"], "x": r["x"]} for r in practice["examples"]]})
    protocol = {"experiment": "bounded-threshold-repair-v1", "origin": "Independent toolkit experiment; not an upstream reproduction",
        "question": "Can an external editor repair a shifted threshold under a one-entry bias-edit budget while preserving protected-region predictions?",
        "editable": {"tensor": "bias", "flat_indices": [1], "max_changed_entries": 1},
        "budgets": {"proposals": proposal_budget, "feedback_calls": feedback_budget,
                    "practice_examples_per_call": 32, "maximum_feedback_example_exposures": feedback_budget * 32},
        "practice_seed": 20261104, "heldout_seed": "private/split-seeds.json; not part of editor workspace",
        "heldout_sha256_commitment": digest(run / "private/heldout.json"), "heldout_examples": 256,
        "success_rule": {"repair_group_accuracy_at_least": .9, "protected_group_regressions_at_most": 0,
                         "overall_regressions_at_most": 0},
        "positive_control": "A bias value of zero is analytically feasible. Use only in manual validation, not as evidence of agent discovery.",
        "feedback": "Baseline/candidate practice accuracy, fixed/regressed IDs, and labeled per-example predictions. No final-set feedback before freeze.",
        "final_evaluation": "One evaluation of the frozen candidate and untouched baseline on identical held-out examples/settings.",
        "isolation": "Logical controller gate only. Private data are not exposed by the API, but filesystem/compute isolation must be supplied externally for an agent study.",
        "actor_type": actor_type, "editor_id": editor_id, "editor_identity_verification": "caller-declared"}
    write_json(run / "protocol.json", protocol)
    lab = helper("weight_lab")
    write_json(run / "private/baseline-practice.json", lab.evaluate_mlp(source, run / "editor/architecture.json", run / "private/practice.json"))
    (run / "editor/README.md").write_text(
        "# Editor workspace\n\nInspect original.npz and architecture.json. Submit a reasoned patch through the controller. "
        "Only bias flat index 1 may change, at most one stored entry. Feedback calls consume the declared budget, including repeats. "
        "The practice inputs have no labels until metered feedback. Do not access private/, run out-of-band inference, train, or read the positive control during an agent trial. "
        "Operator-supplied isolation is required; these written rules are not a security sandbox.\n", encoding="utf-8")
    inputs = [run / "protocol.json", source, run / "editor/architecture.json", run / "editor/practice-inputs.json",
              run / "private/practice.json", run / "private/heldout.json", run / "private/split-seeds.json", run / "private/baseline-practice.json"]
    code = [ROOT / "lab.py", *[ROOT / "weight_space_lab" / name for name in
            ("controller.py", "editor_session.py", "evidence.py", "legacy.py", "reports.py")], ROOT / "skills/weight-space-lab/scripts/weight_lab.py"]
    state = {"schema": "weight-space-controller-v1", "phase": "editing", "budgets": protocol["budgets"],
             "proposals_used": 0, "feedback_used": 0, "feedback_example_exposures": 0,
             "input_sha256": {p.relative_to(run).as_posix(): digest(p) for p in inputs},
             "code_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in code},
             "versions": versions(["numpy", "safetensors"]), "candidates": {}, "events": [], "pending_action": None}
    for path in code:
        target = run / "private/code" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        state["input_sha256"][target.relative_to(run).as_posix()] = digest(target)
    _event(run, state, "create", "complete", actor_type=actor_type, editor_id=editor_id)
    return status(run)


def status(output):
    run = Path(output)
    state = read_json(run / "state.json")
    return {k: state[k] for k in ("phase", "budgets", "proposals_used", "feedback_used", "feedback_example_exposures", "pending_action")}


def submit(output, patch_file, *, _expected_anchor=None):
    run = Path(output)
    with _lock(run):
        state = _load(run)
        _require_anchor(state, _expected_anchor)
        if state["phase"] != "editing":
            _deny(run, state, "submit", "Proposals are allowed only during editing; the frozen candidate cannot change")
        if state["proposals_used"] >= state["budgets"]["proposals"]:
            _deny(run, state, "submit", "The proposal budget is exhausted")
        state["proposals_used"] += 1
        candidate_id = f"c{state['proposals_used']:03d}"
        folder = run / "proposals" / candidate_id
        state["pending_action"] = {"action": "submit", "candidate_id": candidate_id}
        _event(run, state, "submit", "started", candidate_id=candidate_id)
        try:
            folder.mkdir()
            shutil.copyfile(patch_file, folder / "proposed.json")
            patch = read_json(folder / "proposed.json")
            for op in patch["operations"]:
                if op.get("tensor") != "bias" or op.get("indices") != [1] or op.get("op") not in ("set", "add", "scale"):
                    raise ValueError("This protocol permits set/add/scale on bias flat index 1 only")
            helper("weight_lab").patch_checkpoint(run / "editor/original.npz", folder / "proposed.json", folder / "candidate", max_fraction=.25)
            candidate = {"status": "accepted", "checkpoint": f"proposals/{candidate_id}/candidate/model.npz", "feedback_calls": 0}
            state["candidates"][candidate_id] = candidate
            candidate["artifacts"] = _hashes(run, folder)
            state["pending_action"] = None
            _event(run, state, "submit", "accepted", candidate_id=candidate_id, checkpoint_sha256=digest(run / candidate["checkpoint"]))
            return {"candidate_id": candidate_id, "checkpoint_sha256": digest(run / candidate["checkpoint"]), "proposals_used": state["proposals_used"]}
        except BaseException as exc:
            failure = folder / "failure.json" if folder.is_dir() else run / "proposals" / f"{candidate_id}-failure.json"
            write_json(failure, {"error": str(exc), "budget_consumed": True})
            artifacts = {**_hashes(run, folder), failure.relative_to(run).as_posix(): digest(failure)}
            state["candidates"][candidate_id] = {"status": "rejected", "feedback_calls": 0, "artifacts": artifacts}
            state["pending_action"] = None
            _event(run, state, "submit", "rejected", candidate_id=candidate_id, error=str(exc))
            raise


def feedback(output, candidate_id, *, _expected_anchor=None):
    run = Path(output)
    with _lock(run):
        state = _load(run)
        _require_anchor(state, _expected_anchor)
        if state["phase"] != "editing":
            _deny(run, state, "feedback", "Feedback is allowed only during editing; no selection feedback is available after freezing")
        if state["feedback_used"] >= state["budgets"]["feedback_calls"]:
            _deny(run, state, "feedback", "The feedback budget is exhausted")
        candidate = state["candidates"].get(candidate_id)
        if not candidate or candidate["status"] != "accepted":
            raise ValueError("Feedback requires an accepted candidate ID")
        state["feedback_used"] += 1
        state["feedback_example_exposures"] += 32
        folder = run / "feedback" / f"f{state['feedback_used']:03d}"
        state["pending_action"] = {"action": "feedback", "candidate_id": candidate_id}
        _event(run, state, "feedback", "started", candidate_id=candidate_id, call=state["feedback_used"])
        try:
            folder.mkdir()
            lab = helper("weight_lab")
            report = lab.evaluate_mlp(run / candidate["checkpoint"], run / "editor/architecture.json", run / "private/practice.json")
            write_json(folder / "candidate.json", report)
            comparison = lab.compare_reports(run / "private/baseline-practice.json", folder / "candidate.json")
            write_json(folder / "comparison.json", comparison)
            candidate["feedback_calls"] += 1
            state.setdefault("result_artifacts", {}).update(_hashes(run, folder))
            state["pending_action"] = None
            _event(run, state, "feedback", "complete", candidate_id=candidate_id,
                   record=f"feedback/{folder.name}/comparison.json", record_sha256=digest(folder / "comparison.json"))
            return {**comparison, "feedback_calls_used": state["feedback_used"], "feedback_example_exposures": state["feedback_example_exposures"],
                    "examples": report["examples"]}
        except BaseException as exc:
            failure = folder / "failure.json" if folder.is_dir() else run / "feedback" / f"{folder.name}-failure.json"
            write_json(failure, {"error": str(exc), "budget_consumed": True})
            state.setdefault("result_artifacts", {}).update(_hashes(run, folder))
            state["result_artifacts"][failure.relative_to(run).as_posix()] = digest(failure)
            state["pending_action"] = None
            _event(run, state, "feedback", "failed", candidate_id=candidate_id, error=str(exc))
            raise


def freeze(output, candidate_id, *, _expected_anchor=None):
    run = Path(output)
    with _lock(run):
        state = _load(run)
        _require_anchor(state, _expected_anchor)
        if state["phase"] != "editing":
            raise ValueError("Candidate freezing is allowed only during editing")
        candidate = state["candidates"].get(candidate_id)
        if not candidate or candidate["status"] != "accepted" or not candidate["feedback_calls"]:
            raise ValueError("Freeze an accepted candidate that has completed practice feedback")
        frozen = run / "frozen"
        state["phase"] = "freezing"
        state["pending_action"] = {"action": "freeze", "candidate_id": candidate_id}
        _event(run, state, "freeze", "started", candidate_id=candidate_id)
        try:
            frozen.mkdir()
            shutil.copyfile(run / candidate["checkpoint"], frozen / "model.npz")
            write_json(frozen / "selection.json", {"candidate_id": candidate_id, "checkpoint_sha256": digest(frozen / "model.npz"),
                        "proposal_budget_used": state["proposals_used"], "feedback_budget_used": state["feedback_used"],
                        "selection_rule": "Explicit editor selection before final evaluation; no final scores were available through the controller."})
            state["frozen"] = {"candidate_id": candidate_id, "artifacts": _hashes(run, frozen)}
            state["phase"] = "frozen"
            state["pending_action"] = None
            _event(run, state, "freeze", "complete", candidate_id=candidate_id, checkpoint_sha256=digest(frozen / "model.npz"))
            return {"phase": "frozen", "candidate_id": candidate_id, "checkpoint_sha256": digest(frozen / "model.npz")}
        except BaseException as exc:
            write_json(run / "freeze-failure.json", {"error": str(exc), "candidate_id": candidate_id,
                       "note": "Partial freeze artifacts are preserved. This run cannot return to editing; start a new experiment."})
            state.setdefault("result_artifacts", {}).update(_hashes(run, frozen))
            state["result_artifacts"]["freeze-failure.json"] = digest(run / "freeze-failure.json")
            state["phase"] = "freeze_failed"
            state["pending_action"] = None
            _event(run, state, "freeze", "failed", candidate_id=candidate_id, error=str(exc))
            raise


def finalize(output):
    run = Path(output)
    with _lock(run):
        state = _load(run)
        if state["phase"] == "editing":
            _deny(run, state, "final", "Freeze a candidate before accessing final evaluation")
        if state["phase"] != "frozen":
            _deny(run, state, "final", "Final evaluation is one-time; previous results and failures are preserved")
        final = run / "final"
        state["phase"] = "final_running"
        state["pending_action"] = {"action": "final"}
        _event(run, state, "final", "started", candidate_id=state["frozen"]["candidate_id"])
        try:
            final.mkdir()
            lab = helper("weight_lab")
            for arm, checkpoint in (("baseline", run / "editor/original.npz"), ("candidate", run / "frozen/model.npz")):
                write_json(final / f"{arm}.json", lab.evaluate_mlp(checkpoint, run / "editor/architecture.json", run / "private/heldout.json"))
            comparison = lab.compare_reports(final / "baseline.json", final / "candidate.json")
            after = read_json(final / "candidate.json")
            repair = [r for r in after["examples"] if r["id"].startswith("final-repair-")]
            protected_regressions = [x for x in comparison["regressed_ids"] if x.startswith("final-protected-")]
            repair_accuracy = sum(r["correct"] for r in repair) / len(repair)
            result = {"experiment": "bounded-threshold-repair-v1", "comparison": comparison,
                "repair_group_accuracy": repair_accuracy, "protected_group_regressed_ids": protected_regressions,
                "success_rule_met": repair_accuracy >= .9 and not comparison["regressed_ids"],
                "candidate_id": state["frozen"]["candidate_id"], "budgets_used": {"proposals": state["proposals_used"], "feedback_calls": state["feedback_used"], "example_exposures": state["feedback_example_exposures"]},
                "actor": {k: read_json(run / "protocol.json")[k] for k in ("actor_type", "editor_id")},
                "interpretation": "A narrowly scoped result under this protocol. Manual verification is not agent discovery; failures are retained alongside successes."}
            write_json(final / "result.json", result)
            from .reports import controller_report
            controller_report(final, result, state)
            state.setdefault("result_artifacts", {}).update(_hashes(run, final))
            state["phase"] = "final_complete"
            state["pending_action"] = None
            _event(run, state, "final", "complete", result_sha256=digest(final / "result.json"))
            return result
        except BaseException as exc:
            failure = final / "failure.json" if final.is_dir() else run / "final-failure.json"
            write_json(failure, {"error": str(exc), "final_attempt_consumed": True})
            state.setdefault("result_artifacts", {}).update(_hashes(run, final))
            state["result_artifacts"][failure.relative_to(run).as_posix()] = digest(failure)
            state["phase"] = "final_failed"
            state["pending_action"] = None
            _event(run, state, "final", "failed", error=str(exc))
            raise


def audit(output):
    run = Path(output)
    state = read_json(run / "state.json")
    _check(run, state, require_runtime=False)
    return {"verified": True, "final_complete": state["phase"] == "final_complete", **status(run), "events_checked": len(state["events"]),
            "integrity": "Unsigned local hash chain and file checks. Not an OS sandbox or independent attestation."}


def close(output, reason, *, _expected_anchor=None):
    """Record no-selection/early-stop outcomes without exposing final data."""
    run = Path(output)
    if not reason.strip():
        raise ValueError("A nonempty reason is required to close an experiment without selection")
    with _lock(run):
        state = _load(run)
        _require_anchor(state, _expected_anchor)
        if state["phase"] != "editing":
            _deny(run, state, "close", "Only an editing run can close without final evaluation")
        exhausted = (state["proposals_used"] >= state["budgets"]["proposals"] or
                     state["feedback_used"] >= state["budgets"]["feedback_calls"])
        outcome = {"experiment": "bounded-threshold-repair-v1", "status": "no_candidate_selected",
                   "reason": reason, "budget_exhausted": exhausted, "heldout_evaluated": False,
                   "budgets_used": {"proposals": state["proposals_used"], "feedback_calls": state["feedback_used"]},
                   "interpretation": "No final capability measurement exists. This records failure to select or an early stop, not impossibility of a feasible edit."}
        state["phase"] = "closing"
        state["pending_action"] = {"action": "close"}
        _event(run, state, "close", "started", reason=reason)
        try:
            write_json(run / "no-selection.json", outcome)
            state.setdefault("result_artifacts", {})["no-selection.json"] = digest(run / "no-selection.json")
            state["phase"] = "closed_without_final"
            state["pending_action"] = None
            _event(run, state, "close", "complete", reason=reason, budget_exhausted=exhausted)
            return outcome
        except BaseException as exc:
            write_json(run / "close-failure.json", {"error": str(exc), "note": "Preserve this terminal failed closure and start a new experiment."})
            state.setdefault("result_artifacts", {})["close-failure.json"] = digest(run / "close-failure.json")
            state["phase"] = "close_failed"
            state["pending_action"] = None
            _event(run, state, "close", "failed", error=str(exc))
            raise
