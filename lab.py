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
        if command == "experiment":
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
