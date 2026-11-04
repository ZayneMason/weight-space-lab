"""Export numeric pilot evidence and content hashes; omit questions, answers and paths."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(source):
    source = Path(source)
    arms = {}
    matched = None
    for arm in ("baseline", "p50c"):
        folder = source / arm
        job = read(folder / "job.json")
        if job["arm"] != arm or job["pilot"] is not True:
            raise ValueError("Only explicitly labeled paired pilot records are accepted")
        identity = {k: job[k] for k in ("problem_ids", "sampling", "protocol", "prepared_sha256")}
        if matched is not None and identity != matched:
            raise ValueError("Pilot arms have incompatible inputs or settings")
        matched = identity
        rows = []
        for index, problem_id in enumerate(job["problem_ids"]):
            path = folder / (hashlib.sha256(problem_id.encode()).hexdigest()[:20] + ".json")
            record = read(path)
            if record["id"] != problem_id or len(record["generations"]) != 4:
                raise ValueError("Incomplete or misidentified pilot problem")
            samples = []
            for sample in record["generations"]:
                if (sample["tokens"] != len(sample["token_ids"])
                        or type(sample["correct"]) is not bool
                        or sample["finish_reason"] not in ("length", "stop")
                        or sample["truncated"] != (sample["finish_reason"] == "length")):
                    raise ValueError("Inconsistent sample count, grade or finish reason")
                samples.append({"sample": sample["sample"], "correct": sample["correct"],
                                "tokens": sample["tokens"], "finish_reason": sample["finish_reason"],
                                "truncated": sample["truncated"],
                                "text_sha256": hashlib.sha256(sample["text"].encode()).hexdigest(),
                                "token_ids_sha256": hashlib.sha256(json.dumps(sample["token_ids"], separators=(",", ":")).encode()).hexdigest()})
            if sorted(s["sample"] for s in samples) != [0, 1, 2, 3]:
                raise ValueError("Sample IDs must cover 0..3 without duplicates")
            rows.append({"problem_index": index, "seed": record["seed"],
                         "problem_id_sha256": hashlib.sha256(problem_id.encode()).hexdigest(),
                         "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(), "samples": samples})
        samples = [s for r in rows for s in r["samples"]]
        if not samples:
            raise ValueError("Pilot has no samples")
        aggregate = {"pilot": True, "arm": arm, "problems": len(rows), "generations": len(samples),
                     "accuracy": sum(s["correct"] for s in samples) / len(samples),
                     "token_limit_share": sum(s["truncated"] for s in samples) / len(samples),
                     "tokens": sum(s["tokens"] for s in samples)}
        if aggregate != read(folder / "summary.json"):
            raise ValueError("Pilot aggregate disagrees with sample records")
        arms[arm] = {"source_job_sha256": digest(folder / "job.json"), "summary": aggregate, "problems": rows}
    protocol = matched["protocol"]
    return {"schema": "weight-space-sanitized-qwen-pilot-v1", "date": "2026-11-04",
            "origin": "Existing local pilot; no new inference performed during export.",
            "sampling": matched["sampling"],
            "revisions": {k: protocol[k] for k in ("research_revision", "model_revision", "dataset_revision")},
            "arms": arms,
            "omitted": ["benchmark questions and answers", "generated text and token IDs", "filesystem paths", "environment variables"],
            "limits": "Unsigned derived records, two problems only. Hashes bind content without redistributing it; grades are not independently rechecked by this export."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new evidence filename; existing evidence is preserved")
    result = summarize(args.source)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
