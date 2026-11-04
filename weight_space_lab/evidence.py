"""Relative artifact ledgers and adapter-specific evidence validation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

from .legacy import helper


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def new_directory(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new output directory; previous results were preserved.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    return output


def versions(names):
    result = {"python": platform.python_version()}
    for name in names:
        result[name] = importlib.metadata.version(name)
    return result


def seal(output, adapter, scope, summary, code_paths, packages, expected_code=None):
    from .legacy import ROOT
    out = Path(output)
    code_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in code_paths}
    if expected_code is not None and code_hashes != expected_code:
        raise ValueError("Workflow source changed during the experiment; partial evidence was preserved")
    for path in code_paths:
        target = out / "code" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if digest(target) != code_hashes[path.relative_to(ROOT).as_posix()]:
            raise ValueError("Source snapshot changed while being copied")
    shutil.copyfile(ROOT / "LICENSE", out / "code/LICENSE")
    shutil.copyfile(ROOT / "NOTICE.md", out / "code/NOTICE.md")
    (out / "code/licenses").mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "licenses/upstream-research.txt", out / "code/licenses/upstream-research.txt")
    manifest = {"schema": "weight-space-experiment-v1", "status": "complete", "adapter": adapter,
                "scope": scope, "summary": summary,
                "versions": versions(packages),
                "code_sha256": code_hashes,
                "artifacts": {p.relative_to(out).as_posix(): digest(p)
                              for p in sorted(out.rglob("*")) if p.is_file() and p.name != "experiment.json"},
                "integrity": "Unsigned local hash ledger; not an independent attestation."}
    write_json(out / "experiment.json", manifest)
    return manifest


def verify_experiment(output):
    out = Path(output).resolve()
    if (out / "failure.json").exists():
        raise ValueError("This experiment contains failure.json and is not complete; inspect the failure before interpreting results.")
    if not (out / "experiment.json").is_file():
        raise ValueError("No completed experiment.json. Inspect failure.json, or run an experiment in a new output directory.")
    manifest = read_json(out / "experiment.json")
    if manifest.get("schema") != "weight-space-experiment-v1" or manifest.get("status") != "complete":
        raise ValueError("Unsupported or incomplete experiment ledger")
    if not manifest.get("artifacts"):
        raise ValueError("The experiment has no recorded artifacts")
    for name, expected in manifest["artifacts"].items():
        path = (out / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(out) or path == out:
            raise ValueError(f"Artifact path escapes the experiment: {name}")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Artifact integrity mismatch: {name}. Preserve the directory and investigate; do not reuse its metrics.")
    if manifest["adapter"] == "sign-demo":
        summary = validate_demo(out)
        if summary != manifest["summary"]:
            raise ValueError("Experiment summary disagrees with recomputed demo evidence")
    elif manifest["adapter"] == "minatar-breakout":
        engine = helper("reproduce_breakout")
        report = read_json(out / "evaluation.json")
        engine.validate_report(report)
        if report["checkpoint_sha256"] != digest(out / "reference/published.npz"):
            raise ValueError("Breakout evaluation belongs to a different checkpoint")
        engine.verify_reference(out / "reference")
        preview = read_json(out / "replay.json")
        if preview["checkpoint_sha256"] != report["checkpoint_sha256"] or preview["episode"] != report["results"][0]:
            raise ValueError("Replay identity differs from evaluated episode zero")
        if (preview["seed"] != report["settings"]["seed"] or preview["action_seed"] != preview["seed"] + 1_000_000
                or not 1 <= len(preview["frames"]) <= 1000):
            raise ValueError("Replay seeds or frame count differ from the evaluated protocol")
        actual_episode, actual_frames = engine.run_episode(
            (0, preview["seed"], report["settings"]["max_steps"]),
            engine.load_weights(out / "reference/published.npz"), len(preview["frames"]))
        if preview["episode"] != actual_episode or preview["frames"] != actual_frames:
            raise ValueError("Replay frames do not reproduce evaluated episode zero")
        if manifest["scope"] not in ("smoke", "full") or (manifest["scope"] == "full" and
                (report["episodes"] != 1000 or report["settings"]["max_steps"] != 10000)):
            raise ValueError("The experiment scope does not match its episode/step counts")
        expected = {"episodes": report["episodes"], "mean_score": report["mean"], "standard_error": report["se"],
                    "truncated_episodes": sum(r["truncated"] for r in report["results"]),
                    "checkpoint_sha256": report["checkpoint_sha256"]}
        if expected != manifest["summary"]:
            raise ValueError("Breakout summary disagrees with episode records")
    elif manifest["adapter"] == "evidence-export":
        if read_json(out / "evidence.json")["summary"] != manifest["summary"]:
            raise ValueError("Export summary disagrees with its ledger")
    else:
        raise ValueError(f"Unsupported evidence adapter: {manifest['adapter']}")
    return {"verified": True, "adapter": manifest["adapter"], "scope": manifest["scope"],
            "artifacts_checked": len(manifest["artifacts"]), "summary": manifest["summary"]}


def validate_demo(out):
    import numpy as np
    lab = helper("weight_lab")
    receipt = read_json(out / "candidate/receipt.json")
    source, _ = lab.load_checkpoint(out / "original.npz")
    candidate, _ = lab.load_checkpoint(out / "candidate/model.npz")
    if receipt["base_sha256"] != digest(out / "original.npz") or receipt["candidate_sha256"] != digest(out / "candidate/model.npz"):
        raise ValueError("Receipt checkpoint hashes disagree with the preserved files")
    if source.keys() != candidate.keys() or any(source[k].shape != candidate[k].shape or source[k].dtype != candidate[k].dtype for k in source):
        raise ValueError("Candidate tensor shape/dtype inventory differs from source")
    changes = {k: int(np.count_nonzero(source[k] != candidate[k])) for k in source}
    total = sum(t.size for t in source.values())
    if (receipt["changed"] != sum(changes.values()) or receipt["total"] != total
            or receipt["fraction"] != sum(changes.values()) / total
            or receipt["changed_tensors"] != {k: n for k, n in changes.items() if n}):
        raise ValueError("Receipt changed-entry counts disagree with actual tensors")
    proposed = read_json(out / "edit.json")
    applied = read_json(out / "candidate/patch.json")
    if proposed != applied or applied["base_sha256"] != receipt["base_sha256"] or applied["reason"] != receipt["reason"]:
        raise ValueError("Proposed patch, applied patch, and receipt disagree")
    reconstructed = {k: t.copy() for k, t in source.items()}
    for operation in applied["operations"]:
        lab.edit_tensor(reconstructed[operation["tensor"]], operation)
    if any(not np.array_equal(reconstructed[k], candidate[k]) for k in source):
        raise ValueError("Candidate does not implement the recorded patch")
    for checkpoint, report_name in ((out / "original.npz", "baseline.json"), (out / "candidate/model.npz", "candidate.json")):
        fresh = lab.evaluate_mlp(checkpoint, out / "architecture.json", out / "heldout.json")
        if fresh != read_json(out / report_name):
            raise ValueError(f"Evaluation disagrees with its checkpoint/input files: {report_name}")
    comparison = lab.compare_reports(out / "baseline.json", out / "candidate.json")
    if comparison != read_json(out / "comparison.json"):
        raise ValueError("Comparison disagrees with per-example predictions")
    return {**comparison, "changed_entries": receipt["changed"], "total_entries": receipt["total"],
            "changed_fraction": receipt["fraction"]}
