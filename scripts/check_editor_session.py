"""Prewritten stdio transport validation. No external agent or model is invoked."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weight_space_lab import controller, editor_session
from weight_space_lab.evidence import digest, write_json


def check(output):
    run = Path(output).resolve()
    controller.create(run, feedback_budget=1, proposal_budget=1,
                      actor_type="manual-validation", editor_id="prewritten-stdio-control")
    metadata = run / "scripted-metadata.json"
    write_json(metadata, {"editor_id": "prewritten-stdio-control", "provider": "none",
        "model": "prewritten-positive-control", "model_revision": "v1",
        "settings": {"tool_policy": "scripted-transport-validation-only"},
        "prompt": "Inspect the original and submit the prewritten bias[1]=0 control, request practice feedback, then freeze. This validates transport only; it is not agent discovery."})
    child = subprocess.Popen([sys.executable, str(ROOT / "lab.py"), "editor", "serve", str(run),
        "scripted", "--metadata", str(metadata), "--max-requests", "8"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        hello = json.loads(child.stdout.readline())
        if hello["type"] != "hello" or "finalize" in hello["methods"]:
            raise ValueError("Unexpected editor interface")
        def request(number, method, params=None):
            child.stdin.write(json.dumps({"id": number, "method": method, "params": params or {}}) + "\n")
            child.stdin.flush()
            response = json.loads(child.stdout.readline())
            if response.get("id") != number or "error" in response:
                raise ValueError(f"Scripted request failed: {response}")
            return response["result"]
        request(1, "describe")
        original = request(2, "inspect")
        patch = {"base_sha256": original["checkpoint"]["sha256"],
            "reason": "Prewritten positive control; transport validation, not agent discovery.",
            "operations": [{"op": "set", "tensor": "bias", "indices": [1], "values": [0]}]}
        candidate = request(3, "propose", {"patch": patch})["candidate_id"]
        request(4, "feedback", {"candidate_id": candidate})
        request(5, "freeze", {"candidate_id": candidate})
        end = json.loads(child.stdout.readline())
        if end["reason"] != "frozen" or (run / "final").exists():
            raise ValueError("Editor must stop at freezing, before final evaluation")
        child.stdin.close()
        child.wait(timeout=30)
        if child.returncode:
            raise RuntimeError(child.stderr.read())
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()
    editor_session.audit(run, "scripted")
    controller.finalize(run)  # Separate operator action after the editor has exited.
    controller.audit(run)
    editor_session.audit(run, "scripted")
    write_json(run / "transport-validation.json", {"script_sha256": digest(__file__),
        "session_verified": True, "editor_exited_before_final": True,
        "claim": "Prewritten manual transport validation; no external-agent trial or discovery."})
    return run / "final/report.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(check(parser.parse_args().output))


if __name__ == "__main__":
    main()
