"""Recorded JSONL tool bridge. No model client, network listener, or OS sandbox.

The editor cannot select file paths or invoke final evaluation through this API.
Controller quotas remain global across sessions. A research operator must isolate
the editor from the private run and retain external provider-side provenance.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from . import controller
from .evidence import digest, read_json, write_json
from .legacy import ROOT, helper

MAX_LINE_BYTES = 65536
METHODS = ("describe", "inspect", "tensor", "practice_inputs", "propose", "feedback", "freeze", "abstain", "status")
ACTION = {"propose": "submit", "feedback": "feedback", "freeze": "freeze", "abstain": "close"}


def _strict_json(text):
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("JSON nesting exceeds 64 levels")
        elif char in "]}":
            depth -= 1
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON keys are not accepted")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Non-finite JSON is not accepted")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def _session_path(run, session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", session_id):
        raise ValueError("Session ID must be 1..64 letters, digits, underscores or hyphens")
    return Path(run) / "sessions" / session_id


def _anchor(state):
    return {"events": len(state["events"]), "head": state["events"][-1]["sha256"] if state["events"] else "0" * 64}


def _state(run):
    with controller._lock(run):
        return controller._load(run)


def _public_status(state):
    return {k: state[k] for k in ("phase", "budgets", "proposals_used", "feedback_used", "feedback_example_exposures")}


def _metadata(value, protocol):
    keys = {"editor_id", "provider", "model", "model_revision", "settings", "prompt"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Metadata requires only editor_id, provider, model, model_revision, settings, prompt; credentials are not accepted")
    for name in ("editor_id", "provider", "model", "model_revision"):
        if not isinstance(value[name], str) or not 1 <= len(value[name]) <= 256:
            raise ValueError(f"Metadata {name} must be a nonempty string of at most 256 characters")
    if value["editor_id"] != protocol["editor_id"]:
        raise ValueError("Session editor_id must match the run protocol")
    if not isinstance(value["prompt"], str) or not 1 <= len(value["prompt"].encode()) <= 32768:
        raise ValueError("Retain a nonempty editor prompt of at most 32768 UTF-8 bytes")
    settings = value["settings"]
    if not isinstance(settings, dict) or set(settings) - {"temperature", "reasoning_effort", "seed", "tool_policy"}:
        raise ValueError("Settings allow temperature, reasoning_effort, seed and tool_policy only")
    if any(type(v) not in (str, int, float, bool, type(None)) or (isinstance(v, str) and len(v) > 512) for v in settings.values()):
        raise ValueError("Settings must contain bounded scalar values")
    json.dumps(value, allow_nan=False)
    return value


def _error(exc, method):
    text = str(exc)
    if isinstance(exc, controller.ControllerChangedError):
        code = "controller_changed"
    elif "feedback budget" in text:
        code = "feedback_budget_exhausted"
    elif "proposal budget" in text:
        code = "proposal_budget_exhausted"
    elif "only during editing" in text:
        code = "editing_closed"
    elif method == "propose":
        code = "patch_rejected"
    else:
        code = "operation_rejected"
    return {"code": code, "message": "Operation rejected under the declared protocol. Consult describe/status; operator details are retained in the private journal."}


class Session:
    def __init__(self, run, session_id, metadata, max_requests=32):
        self.run = Path(run)
        self.path = _session_path(run, session_id)
        if type(max_requests) is not int or not 1 <= max_requests <= 256:
            raise ValueError("max_requests must be 1..256")
        protocol = read_json(self.run / "protocol.json")
        metadata = _metadata(metadata, protocol)
        if protocol["experiment"] not in ("bounded-threshold-repair-v1", "bounded-mlp-edit-v1"):
            raise ValueError("Unsupported bounded experiment protocol")
        self.lock = self.run / ".editor-session.lock"
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError("An editor session is already active or its lock needs operator investigation") from None
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(str(os.getpid()))
            state = _state(self.run)
            if state["code_sha256"].get("weight_space_lab/editor_session.py") != digest(__file__):
                raise ValueError("The experiment must pin this bridge source at creation; create a fresh run")
            if state["phase"] != "editing":
                raise ValueError("Start an editor session only during editing")
            self.path.parent.mkdir(exist_ok=True)
            provenance = {"editor_id": metadata["editor_id"], "actor_type": protocol["actor_type"],
                          **{k: metadata[k] for k in ("provider", "model", "model_revision", "settings")}}
            registry = self.run / "editor-provenance.json"
            if registry.exists():
                if read_json(registry) != provenance:
                    raise ValueError("Editor provider/model/revision/settings are pinned across sessions; start a new experiment to change them")
            else:
                if list(self.path.parent.glob("*/header.json")):
                    raise ValueError("Existing session provenance is missing; preserve this run and investigate")
                write_json(registry, provenance)
            self.path.mkdir(exist_ok=False)
            (self.path / "prompt.txt").write_text(metadata["prompt"], encoding="utf-8", newline="\n")
            shutil.copyfile(__file__, self.path / "bridge-source.py")
            shutil.copyfile(ROOT / "LICENSE", self.path / "LICENSE")
            self.header = {"schema": "weight-space-editor-session-v1", "session_id": session_id,
                "actor_type": protocol["actor_type"], "identity": {k: v for k, v in metadata.items() if k != "prompt"},
                "identity_verification": "Caller-declared; provider transcript/authentication are external responsibilities.",
                "max_requests": max_requests, "max_line_bytes": MAX_LINE_BYTES,
                "prompt_sha256": digest(self.path / "prompt.txt"), "bridge_sha256": digest(self.path / "bridge-source.py"),
                "protocol_sha256": digest(self.run / "protocol.json"), "controller_start": _anchor(state),
                "editor_provenance_sha256": digest(registry),
                "methods": list(METHODS), "final_evaluation_method_available": False,
                "isolation": "Logical tool gate only; no OS or network isolation is supplied."}
            write_json(self.path / "header.json", self.header)
            self.journal = (self.path / "events.jsonl").open("x", encoding="utf-8", newline="\n")
            self.previous = digest(self.path / "header.json")
            self.events = 0
            self.requests = 0
            self.ids = set()
            self.expected_anchor = _anchor(state)
            self.active = True
            self.reason = "eof"
            self.protocol = protocol
        except BaseException:
            self.lock.unlink()
            raise

    def record(self, kind, **payload):
        row = {"index": self.events, "kind": kind, "utc": datetime.now(timezone.utc).isoformat(),
               "previous": self.previous, **payload}
        row["sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.journal.write(json.dumps(row, allow_nan=False) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.previous = row["sha256"]
        self.events += 1

    def hello(self):
        return {"type": "hello", "schema": "weight-space-editor-tools-v1", "methods": list(METHODS),
                "limits": {"max_requests": self.header["max_requests"], "max_line_bytes": MAX_LINE_BYTES},
                "note": "No final evaluation or arbitrary file access is available through this API. No model is invoked by this bridge."}

    def dispatch(self, method, params, state):
        expected = {"describe": set(), "inspect": set(), "practice_inputs": set(), "status": set(),
                    "propose": {"patch"}, "feedback": {"candidate_id"}, "freeze": {"candidate_id"},
                    "abstain": {"reason"}}
        if method == "tensor":
            names = [r["name"] for r in helper("weight_lab").describe(self.run / "editor/original.npz")["tensors"]]
            if set(params) - {"name", "start", "count"} or params.get("name") not in names:
                raise ValueError("Tensor requests allow existing tensor names and bounded start/count only")
            if type(params.get("start", 0)) is not int or type(params.get("count", 64)) is not int:
                raise ValueError("Tensor bounds must be integers")
        elif set(params) != expected[method]:
            raise ValueError("Unexpected tool parameters")
        if method == "describe":
            return {k: self.protocol[k] for k in ("experiment", "question", "editable", "budgets", "success_rule", "feedback")}
        if method == "status":
            return _public_status(state)
        if method == "inspect":
            return {"checkpoint": helper("weight_lab").describe(self.run / "editor/original.npz"),
                    "architecture": read_json(self.run / "editor/architecture.json")}
        if method == "tensor":
            return helper("weight_lab").tensor_slice(self.run / "editor/original.npz", **params)
        if method == "practice_inputs":
            return read_json(self.run / "editor/practice-inputs.json")
        if method == "propose":
            path = self.path / f"proposal-request-{self.requests:03d}.json"
            write_json(path, params["patch"])
            return controller.submit(self.run, path, _expected_anchor=self.expected_anchor)
        if method in ("feedback", "freeze"):
            if not isinstance(params["candidate_id"], str) or not re.fullmatch(r"c\d{3}", params["candidate_id"]):
                raise ValueError("Use an accepted candidate ID")
            return getattr(controller, method)(self.run, params["candidate_id"], _expected_anchor=self.expected_anchor)
        if not isinstance(params["reason"], str) or not 1 <= len(params["reason"]) <= 4096:
            raise ValueError("An abstention requires a bounded nonempty reason")
        return controller.close(self.run, params["reason"], _expected_anchor=self.expected_anchor)

    def process(self, wire):
        before = _anchor(_state(self.run))
        self.requests += 1
        self.record("request", number=self.requests, wire_b64=base64.b64encode(wire).decode(),
                    wire_sha256=hashlib.sha256(wire).hexdigest(), controller_before=before,
                    wire_capture="bounded_prefix" if len(wire) > MAX_LINE_BYTES else "complete_line")
        request_id = None
        method = None
        if before != self.expected_anchor:
            self.active = False; self.reason = "controller_changed"
            response = {"id": None, "error": {"code": "controller_changed", "message": "The operator must investigate an out-of-session controller change."}}
        elif len(wire) > MAX_LINE_BYTES:
            self.active = False; self.reason = "oversized_request"
            response = {"id": None, "error": {"code": "request_too_large", "message": "Input exceeded the declared byte limit; the session is ending without draining unbounded input."}}
        else:
            try:
                request = _strict_json(wire.decode("utf-8"))
                if not isinstance(request, dict) or set(request) != {"id", "method", "params"}:
                    raise ValueError("Expected id, method and params")
                request_id = request["id"]
                if (type(request_id) not in (str, int) or (isinstance(request_id, str) and not 1 <= len(request_id) <= 64)
                        or (type(request_id) is int and not 0 <= request_id <= 2**31-1)):
                    request_id = None
                    raise ValueError("Invalid request ID")
                if request_id in self.ids:
                    raise ValueError("Request ID was already used")
                self.ids.add(request_id)
                method = request["method"]
                if not isinstance(method, str):
                    method = None
                    raise ValueError("Method must be a string")
                if method not in METHODS:
                    response = {"id": request_id, "error": {"code": "method_not_allowed", "message": "Only the advertised editor tools are available."}}
                elif not isinstance(request["params"], dict):
                    raise ValueError("Params must be an object")
                else:
                    try:
                        state = _state(self.run)
                        controller._require_anchor(state, before)
                        result = self.dispatch(method, request["params"], state)
                        response = {"id": request_id, "result": result}
                        if method in ("freeze", "abstain"):
                            self.active = False; self.reason = "frozen" if method == "freeze" else "abstained"
                    except Exception as exc:
                        self.record("operator_error", number=self.requests, exception=type(exc).__name__, detail=str(exc))
                        response = {"id": request_id, "error": _error(exc, method)}
                        if isinstance(exc, controller.ControllerChangedError):
                            self.active = False; self.reason = "controller_changed"
            except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
                response = {"id": request_id, "error": {"code": "invalid_request", "message": "Malformed, duplicate or unsupported request parameters."}}
        after = _anchor(_state(self.run))
        self.record("response", number=self.requests, method=method, response=response,
                    controller_before=before, controller_after=after)
        self.expected_anchor = after
        if self.active and self.requests >= self.header["max_requests"]:
            self.active = False; self.reason = "request_limit"
        return response

    def finish(self):
        try:
            state = _state(self.run)
            self.record("end", reason=self.reason, requests=self.requests, controller_end=_anchor(state))
            self.journal.close()
            summary = {"schema": "weight-space-editor-session-summary-v1", "reason": self.reason,
                       "requests": self.requests, "events": self.events, "journal_head": self.previous,
                       "controller_end": _anchor(state), "controller_phase": state["phase"],
                       "artifacts": {p.name: digest(p) for p in sorted(self.path.iterdir()) if p.is_file()}}
            write_json(self.path / "summary.json", summary)
            return {"type": "session_end", "reason": self.reason, "requests": self.requests, "controller_phase": state["phase"]}
        finally:
            if not self.journal.closed:
                self.journal.close()
            self.lock.unlink()


def serve(run, session_id, metadata, input_stream, output_stream, max_requests=32):
    session = Session(run, session_id, metadata, max_requests)
    def emit(value):
        output_stream.write(json.dumps(value, allow_nan=False) + "\n")
        output_stream.flush()
    try:
        emit(session.hello())
        while session.active:
            wire = input_stream.readline(MAX_LINE_BYTES + 1)
            if not wire:
                break
            emit(session.process(wire))
    except BaseException:
        session.reason = "operator_error"
        raise
    finally:
        emit(session.finish())
    return 0


def _audit_transition(row, pending, changes, path):
    """Bind successful/failed mutations to the recorded request, not just an action name."""
    method = row["method"]
    response = row["response"]
    if not changes:
        if method in ACTION and "result" in response:
            raise ValueError("Successful mutation has no controller events")
        return
    try:
        request = _strict_json(base64.b64decode(pending["wire_b64"], validate=True).decode("utf-8"))
        params = request["params"]
        if request["method"] != method or request["id"] != response["id"] or not isinstance(params, dict):
            raise ValueError("Controller mutation differs from the wire request")
    except (KeyError, TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("Controller mutation has no valid wire request") from exc
    outcomes = [e["outcome"] for e in changes]
    if "result" in response:
        expected = ["started", "accepted" if method == "propose" else "complete"]
        if outcomes != expected:
            raise ValueError("Successful tool response disagrees with controller outcomes")
    elif (response["error"]["code"] == "controller_changed" or outcomes not in
          (["denied"], ["started", "rejected"] if method == "propose" else ["started", "failed"])):
        raise ValueError("Failed tool response disagrees with controller outcomes")
    if outcomes == ["denied"]:
        return
    if method in ("feedback", "freeze"):
        if any(e.get("candidate_id") != params.get("candidate_id") for e in changes):
            raise ValueError("Controller candidate differs from the wire request")
    elif method == "abstain":
        if any(e.get("reason", params.get("reason")) != params.get("reason") for e in changes):
            raise ValueError("Controller closure differs from the wire reason")
    elif method == "propose":
        candidate_id = changes[0]["candidate_id"]
        if any(e.get("candidate_id") != candidate_id for e in changes):
            raise ValueError("Controller proposal identity changed")
        copied = path / f"proposal-request-{pending['number']:03d}.json"
        if read_json(copied) != params.get("patch"):
            raise ValueError("Submitted patch differs from the recorded wire request")
        if "result" in response and response["result"].get("candidate_id") != candidate_id:
            raise ValueError("Returned proposal differs from the controller candidate")


def audit(run, session_id):
    run = Path(run)
    path = _session_path(run, session_id)
    header = read_json(path / "header.json")
    summary = read_json(path / "summary.json")
    for name, expected in summary["artifacts"].items():
        if Path(name).name != name or digest(path / name) != expected:
            raise ValueError(f"Session artifact hash mismatch: {name}")
    if (header["protocol_sha256"] != digest(run / "protocol.json") or header["prompt_sha256"] != digest(path / "prompt.txt")
            or header["bridge_sha256"] != digest(path / "bridge-source.py")
            or header["editor_provenance_sha256"] != digest(run / "editor-provenance.json")):
        raise ValueError("Session prompt, source or protocol identity differs")
    controller.audit(run)
    state = read_json(run / "state.json")
    def check_anchor(anchor):
        n = anchor["events"]
        if type(n) is not int or not 0 <= n <= len(state["events"]):
            raise ValueError("Session controller event count is invalid")
        expected = state["events"][n-1]["sha256"] if n else "0" * 64
        if anchor["head"] != expected:
            raise ValueError("Session anchor differs from the controller journal")
    previous = digest(path / "header.json")
    expected_anchor = header["controller_start"]
    check_anchor(expected_anchor)
    requests = 0; pending = None; events = []
    for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines():
        row = _strict_json(line)
        payload = {k: v for k, v in row.items() if k != "sha256"}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if row["index"] != len(events) or row["previous"] != previous or row["sha256"] != expected:
            raise ValueError("Session event hash chain is inconsistent")
        if row["kind"] == "request":
            if pending is not None or row["controller_before"] != expected_anchor:
                raise ValueError("Controller changed outside the recorded session sequence")
            wire = base64.b64decode(row["wire_b64"], validate=True)
            if hashlib.sha256(wire).hexdigest() != row["wire_sha256"]:
                raise ValueError("Recorded wire input hash differs")
            requests += 1
            if row["number"] != requests:
                raise ValueError("Session request numbers are inconsistent")
            pending = row
        elif row["kind"] == "response":
            if pending is None or row["number"] != requests or row["controller_before"] != pending["controller_before"]:
                raise ValueError("Session response has no matching request")
            check_anchor(row["controller_after"])
            start, end = row["controller_before"]["events"], row["controller_after"]["events"]
            changes = state["events"][start:end]
            if end < start or len(changes) > 2 or any(e["action"] != ACTION.get(row["method"]) for e in changes):
                raise ValueError("Controller actions do not match the recorded editor tool")
            _audit_transition(row, pending, changes, path)
            expected_anchor = row["controller_after"]
            pending = None
        previous = expected; events.append(row)
    if (pending is not None or not events or events[-1]["kind"] != "end"
            or events[-1]["controller_end"] != expected_anchor or summary["controller_end"] != expected_anchor
            or summary["events"] != len(events) or summary["requests"] != requests
            or summary["journal_head"] != previous or requests > header["max_requests"]):
        raise ValueError("Session is incomplete or its summary disagrees with the journal")
    return {"verified": True, "session_id": session_id, "editor_id": header["identity"]["editor_id"],
            "actor_type": header["actor_type"], "requests": requests, "end_reason": summary["reason"],
            "controller_phase_now": state["phase"], "final_evaluation_method_available": False,
            "limits": "Unsigned local records; caller-declared identity and external isolation/provider provenance are not verified."}
