"""Validate 12 matched synthetic cases with scripted controls. Never invokes APIs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weight_space_lab import benchmark
from weight_space_lab.evidence import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    config = read_json(ROOT / "experiments/editing-benchmark/pilot.example.json")
    config["task_seeds"] = [11, 23, 47]
    config["limits"]["max_total_api_calls"] *= 3
    config["limits"]["max_total_output_tokens"] *= 3
    benchmark.create(output, config)
    result = benchmark.run(output)  # Default controls-only; never set allow_paid_api here.
    benchmark.audit(output)
    benchmark.report(output)
    if (result["api_calls_reserved"] != 0 or result["external_comparison_complete"]
            or result["controls"][0]["successes"] != 12 or result["controls"][1]["mean_accuracy_completed_only"] != .5):
        raise ValueError("Scripted benchmark calibration did not meet its declared expectations")
    print(json.dumps({"controls": [{"editor": c["editor_id"], "successes": c["successes"], "planned": c["planned"]} for c in result["controls"]],
                      "api_calls": result["api_calls_reserved"], "model_trials": "pending", "report": str(output / "scorecard.html")}, indent=2))


if __name__ == "__main__":
    main()
