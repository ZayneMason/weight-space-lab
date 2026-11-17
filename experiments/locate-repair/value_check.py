"""Roadmap #37 side check: is the leftover disagreement of a right-weights pair repair set by the value?

For each seed, fit only the two truly broken entries on the 1,024 practice labels (the same 200 x 200
joint grid the pair repair uses), then report how far each fitted value lands from the teacher's value
(in tensor spreads) and how often the result disagrees with the teacher on a million uniform inputs.

    python experiments/locate-repair/value_check.py --seeds 5 8 12 20 13 15
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("behaviour_scores", HERE / "behaviour_scores.py")
bs = importlib.util.module_from_spec(spec)
sys.modules["behaviour_scores"] = bs
spec.loader.exec_module(bs)
tf, rs, lr = bs.tf, bs.rs, bs.lr


def check(seed, width=32):
    case = rs.resize(tf.corrupt(width, seed), rs.LOW, rs.HIGH)
    px, py, _ = lr.split(case["practice"])
    edits, accuracy = tf.joint(case, px, py, [tuple(f) for f in case["faults"]])
    x, labels = bs.make_probe(case, bs.PROBE)
    entries = []
    for name, index, value in edits:
        teacher, scale = float(case["teacher"][name].flat[index]), case["scale"][name]
        entries.append({"entry": f"{name}[{index}]",
                        "fault_spreads": round((float(case["arrays"][name].flat[index]) - teacher) / scale, 3),
                        "fit_minus_teacher_spreads": round((value - teacher) / scale, 3)})
    return {"seed": seed, "practice_accuracy": round(float(accuracy), 4),
            "disagreement": float((tf.predict(tf.apply(case["arrays"], edits), x) != labels).mean()), "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        rows.append(check(seed))
        print(json.dumps(rows[-1]), flush=True)
    if args.output:
        args.output.write_text(json.dumps({"schema": "weight-space-value-check-v1", "cases": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
