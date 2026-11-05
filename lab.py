#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "safetensors==0.8.0"]
# ///
"""A local toolkit for inspecting, patching, and evaluating neural-network weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GENERIC = {"inspect": "inspect", "tensor": "tensor", "edit": "patch", "patch": "patch",
           "evaluate": "evaluate-mlp", "compare": "compare"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in (*GENERIC, "qwen", "breakout-tools"):
        name, args = argv[0], argv[1:]
        if name == "qwen":
            command = [ROOT / "qwen/reproduce_qwen.py", *(args or ["--help"])]
        elif name == "breakout-tools":
            command = [ROOT / "skills/weight-space-lab/scripts/reproduce_breakout.py", *(args or ["--help"])]
        else:
            command = [ROOT / "skills/weight-space-lab/scripts/weight_lab.py", GENERIC[name], *args]
        return subprocess.run([sys.executable, *map(str, command)], check=False).returncode
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Existing adapters (append --help for exact arguments):
  inspect / tensor     Read a compatible local checkpoint; CPU, no downloads.
  edit / patch         Bounded F16/F32/F64 editing; preserves the source.
  evaluate / compare   Dense MLP evaluation only; not a transformer evaluator.
  breakout-tools       Original Breakout CLI. 'fresh' explicitly consumes model allowance.
  qwen                 Separate BF16 reconstruction/GPU evaluation; large downloads and long runs.

Start: python lab.py demo runs/sign-demo
Costs: demo uses installed CPU dependencies only. Breakout fetches pinned research
assets unless --reference is supplied. No command implicitly starts fresh generation.
See docs/FIRST_EXPERIMENT.md and qwen/README.md for prerequisites and boundaries.""")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Offline CPU example with a prewritten patch and local HTML report")
    demo.add_argument("output", type=Path)
    verify = sub.add_parser("verify", help="Check recorded hashes and recompute supported experiment evidence")
    verify.add_argument("output", type=Path)
    export = sub.add_parser("export-evidence", help="Export verified metadata, without weights or private paths")
    export.add_argument("source", type=Path)
    export.add_argument("output", type=Path)
    breakout = sub.add_parser("breakout", help="Fetch, verify, evaluate and preview MinAtar Breakout; CPU plus research-asset downloads")
    breakout.add_argument("output", type=Path)
    breakout.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    breakout.add_argument("--episodes", type=int)
    breakout.add_argument("--max-steps", type=int)
    breakout.add_argument("--seed", type=int, default=0)
    breakout.add_argument("--workers", type=int, default=2)
    breakout.add_argument("--reference", type=Path, help="Reuse an existing verified reference directory instead of downloading")
    breakout.add_argument("--preview-frames", type=int, default=200)
    benchmark = sub.add_parser("benchmark", help="Paired synthetic editing suite; controls are offline, API runs require explicit opt-in")
    benchmark_actions = benchmark.add_subparsers(dest="action", required=True)
    prepare = benchmark_actions.add_parser("create", help="Declare editors, matched tasks and budgets without calling APIs")
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--config", type=Path, required=True)
    run_benchmark = benchmark_actions.add_parser("run", help="Run scripted controls; paid API editors run only with --allow-paid-api")
    run_benchmark.add_argument("output", type=Path)
    run_benchmark.add_argument("--allow-paid-api", action="store_true", help="Explicitly permit paid Claude/Fireworks requests under the recorded budgets")
    run_benchmark.add_argument("--allow-cli", action="store_true", help="Explicitly permit Claude/Codex CLI rounds using existing sign-ins and account allowance")
    for name in ("summary", "audit", "report"):
        benchmark_action = benchmark_actions.add_parser(name)
        benchmark_action.add_argument("output", type=Path)
    editor = sub.add_parser("editor", help="Recorded JSONL tool bridge; no model invocation or OS sandbox")
    editor_actions = editor.add_subparsers(dest="action", required=True)
    serve = editor_actions.add_parser("serve", help="Operator-side stdio bridge; final evaluation is unavailable to the editor")
    serve.add_argument("run", type=Path)
    serve.add_argument("session_id")
    serve.add_argument("--metadata", type=Path, required=True)
    serve.add_argument("--max-requests", type=int, default=32)
    audit = editor_actions.add_parser("audit", help="Verify a session transcript and its controller anchors")
    audit.add_argument("run", type=Path)
    audit.add_argument("session_id")
    experiment = sub.add_parser("experiment", help="Bounded independent experiment: budgeted feedback, freezing, one-time final evaluation")
    actions = experiment.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create", help="Prepare a narrow threshold-repair study; no agent or model allowance is invoked")
    create.add_argument("output", type=Path)
    create.add_argument("--feedback-budget", type=int, default=3)
    create.add_argument("--proposal-budget", type=int, default=5)
    create.add_argument("--actor-type", choices=["manual-validation", "external-agent"], default="manual-validation")
    create.add_argument("--editor-id", default="unassigned")
    for action in ("status", "submit", "feedback", "freeze", "finalize", "audit", "close"):
        p = actions.add_parser(action)
        p.add_argument("output", type=Path)
        if action == "submit":
            p.add_argument("patch_file", type=Path)
        if action in ("feedback", "freeze"):
            p.add_argument("candidate_id")
        if action == "close":
            p.add_argument("--reason", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "benchmark":
            from weight_space_lab import benchmark
            action = args.pop("action")
            if action == "create":
                from weight_space_lab.evidence import read_json
                args["config"] = read_json(args["config"])
            result = getattr(benchmark, "summarize" if action == "summary" else action)(**args)
            if action == "run":
                benchmark.report(args["output"])
            if isinstance(result, dict):
                result = {k: v for k, v in result.items() if k != "trials"}
        elif command == "editor":
            from weight_space_lab import editor_session
            action = args.pop("action")
            if action == "serve":
                from weight_space_lab.evidence import read_json
                args["metadata"] = read_json(args["metadata"])
                return editor_session.serve(**args, input_stream=sys.stdin.buffer, output_stream=sys.stdout)
            result = editor_session.audit(**args)
        elif command == "experiment":
            from weight_space_lab import controller
            action = args.pop("action")
            result = getattr(controller, action)(**args)
        elif command == "verify":
            from weight_space_lab.evidence import verify_experiment
            result = verify_experiment(args["output"])
        else:
            from weight_space_lab import workflows
            action = {"demo": workflows.run_demo, "breakout": workflows.run_breakout,
                      "export-evidence": workflows.export_evidence}[command]
            result = action(**args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install requirements.txt; Breakout also needs requirements-breakout.txt. Qwen uses its separate environment.", file=sys.stderr)
        return 2
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Partial evidence is preserved; inspect failure.json and use a new output directory.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
