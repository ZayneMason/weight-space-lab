"""Manual positive/negative controller controls. No agent or hosted model is invoked."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weight_space_lab import controller
from weight_space_lab.evidence import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run = args.output
    controller.create(run, feedback_budget=2, proposal_budget=2, actor_type="manual-validation", editor_id="prewritten-positive-and-negative-controls")
    candidates = []
    for name, value in (("regressing-control", .6), ("feasible-control", 0.0)):
        path = run / f"{name}.json"
        write_json(path, {"base_sha256": digest(run / "editor/original.npz"),
            "reason": f"Prewritten {name}; infrastructure validation only, not agent discovery.",
            "operations": [{"op": "set", "tensor": "bias", "indices": [1], "values": [value]}]})
        candidate = controller.submit(run, path)["candidate_id"]
        feedback = controller.feedback(run, candidate)
        if name == "regressing-control" and not feedback["regressed_ids"]:
            raise ValueError("Negative control did not expose the expected regressions")
        candidates.append(candidate)
    controller.freeze(run, candidates[1])
    controller.finalize(run)
    controller.audit(run)
    write_json(run / "validation-source.json", {"script_sha256": digest(__file__),
        "claim": "Prewritten manual controller validation. No external-agent research result."})
    print(run / "final/report.html")


if __name__ == "__main__":
    main()
