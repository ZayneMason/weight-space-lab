"""Thin workflows over the existing explicit task adapters."""
from pathlib import Path
import shutil

from .evidence import digest, new_directory, read_json, seal, validate_demo, verify_experiment, write_json
from .legacy import ROOT, helper
from .reports import breakout_report, demo_report


def code_paths(adapter):
    return [ROOT / "lab.py", *sorted((ROOT / "weight_space_lab").glob("*.py")),
            ROOT / "skills/weight-space-lab/scripts" / (adapter + ".py")]


def failure(out, stage, exc):
    write_json(out / "failure.json", {"status": "failed", "stage": stage, "error": str(exc),
               "exception": type(exc).__name__, "note": "Partial artifacts are preserved. Inspect them and use a new output directory to retry."})


def run_demo(output):
    out = new_directory(output)
    stage = "demo"
    try:
        code = code_paths("weight_lab")
        initial_code = {p.relative_to(ROOT).as_posix(): digest(p) for p in code}
        lab = helper("weight_lab")
        lab.demo_files(out)
        stage = "validate evidence"
        summary = validate_demo(out)
        write_json(out / "inspection.json", lab.describe(out / "original.npz"))
        stage = "render report"
        demo_report(out, summary)
        manifest = seal(out, "sign-demo", "prewritten demonstration; six predefined examples; no discovery claim",
                        summary, code, ["numpy", "safetensors"], initial_code)
        verify_experiment(out)
        return {"report": str(out / "report.html"), **manifest}
    except BaseException as exc:
        failure(out, stage, exc)
        raise


def run_breakout(output, mode="smoke", episodes=None, seed=0, workers=2, max_steps=None,
                 reference=None, preview_frames=200):
    if mode not in ("smoke", "full"):
        raise ValueError("Choose Breakout mode smoke or full")
    count = episodes if episodes is not None else (8 if mode == "smoke" else 1000)
    ceiling = max_steps if max_steps is not None else (1000 if mode == "smoke" else 10000)
    if mode == "full" and (count != 1000 or ceiling != 10000):
        raise ValueError("Full mode requires exactly 1000 episodes and 10000 max steps. Use smoke for bounded checks.")
    if mode == "smoke" and not 2 <= count <= 50:
        raise ValueError("Smoke mode requires 2..50 episodes; use full for the published 1000-episode protocol")
    if not 1 <= ceiling <= 10000 or not 1 <= preview_frames <= 1000 or workers < 1 or not 0 <= seed <= 2**32-count:
        raise ValueError("Invalid seed, workers, preview frame count (1..1000), or step ceiling (1..10000)")
    out = new_directory(output)
    stage = "fetch reference"
    try:
        code = code_paths("reproduce_breakout")
        initial_code = {p.relative_to(ROOT).as_posix(): digest(p) for p in code}
        engine = helper("reproduce_breakout")
        if reference:
            stage = "verify supplied reference"
            engine.verify_reference(reference)
            destination = out / "reference"
            destination.mkdir()
            for name in (*engine.ASSETS, "sources.json"):
                shutil.copyfile(Path(reference) / name, destination / name)
        else:
            engine.fetch(out / "reference")
        stage = "verify reference"
        verification = engine.verify_reference(out / "reference")
        write_json(out / "verification.json", verification)
        stage = "evaluate"
        checkpoint = out / "reference/published.npz"
        engine.evaluate(checkpoint, out / "evaluation.json", episodes=count, seed=seed,
                        workers=workers, max_steps=ceiling)
        report = read_json(out / "evaluation.json")
        engine.validate_report(report)
        stage = "compare"
        write_json(out / "comparison.json", engine.compare(out / "reference", out / "evaluation.json"))
        stage = "replay episode zero"
        episode, frames = engine.run_episode((0, seed, ceiling), engine.load_weights(checkpoint), preview_frames)
        if episode != report["results"][0]:
            raise ValueError("Preview replay differs from evaluated episode zero; no preview claim can be made")
        replay = {"schema": "weight-space-minatar-replay-v1", "game": "MinAtar Breakout",
                  "checkpoint_sha256": digest(checkpoint), "research_revision": engine.REVISION,
                  "seed": seed, "action_seed": seed + 1_000_000, "episode": episode,
                  "selection": "Episode zero, selected in advance; prefix only if frame cap is reached.", "frames": frames}
        write_json(out / "replay.json", replay)
        stage = "render report"
        breakout_report(out, report, replay, mode)
        summary = {"episodes": count, "mean_score": report["mean"], "standard_error": report["se"],
                   "truncated_episodes": sum(r["truncated"] for r in report["results"]),
                   "checkpoint_sha256": report["checkpoint_sha256"]}
        manifest = seal(out, "minatar-breakout", mode, summary, code, ["numpy", "minatar"], initial_code)
        verify_experiment(out)
        return {"report": str(out / "report.html"), **manifest}
    except BaseException as exc:
        failure(out, stage, exc)
        raise


def export_evidence(source, output):
    """Export aggregate/per-example records without weights, prompts or private paths."""
    source = Path(source)
    verification = verify_experiment(source)
    manifest = read_json(source / "experiment.json")
    adapter = manifest["adapter"]
    allowed = {"sign-demo": ["inspection.json", "edit.json", "candidate/receipt.json", "baseline.json", "candidate.json", "comparison.json"],
               "minatar-breakout": ["verification.json", "evaluation.json", "comparison.json", "reference/sources.json"]}
    if adapter not in allowed:
        raise ValueError("Export supports original sign-demo or minatar-breakout experiments, not nested exports")
    out = new_directory(output)
    try:
        payload = {"schema": "weight-space-evidence-v1", "adapter": adapter, "scope": manifest["scope"],
                   "source_ledger_sha256": digest(source / "experiment.json"), "summary": verification["summary"],
                   "code_sha256": manifest["code_sha256"], "versions": manifest["versions"],
                   "artifact_sha256": manifest["artifacts"],
                   "records": {name: read_json(source / name) for name in allowed[adapter]},
                   "omitted": ["checkpoint files", "benchmark prompts", "raw hosted-model transcripts", "workstation paths"],
                   "integrity": "Unsigned metadata export; raw checkpoints are needed to independently repeat inference."}
        write_json(out / "evidence.json", payload)
        if adapter == "minatar-breakout":
            for name in ("LICENSE", "LICENSE-MinAtar"):
                shutil.copyfile(source / "reference" / name, out / name)
        (out / "README.md").write_text(
            "# Experiment evidence\n\nSee `evidence.json` for measurements, protocol, source hashes, and artifact digests. "
            "Checkpoints, prompts and raw hosted-model transcripts are excluded. "
            "This unsigned ledger checks file integrity, not independent authorship. "
            "Breakout reference scores remain attributed to Takehi et al.; retained licenses govern downloaded assets.\n",
            encoding="utf-8")
        return seal(out, "evidence-export", manifest["scope"], verification["summary"],
                    [ROOT / "weight_space_lab/workflows.py"], [])
    except BaseException as exc:
        failure(out, "export", exc)
        raise
