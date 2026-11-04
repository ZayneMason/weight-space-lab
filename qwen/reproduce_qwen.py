#!/usr/bin/env python3
"""Pinned Qwen p50c reconstruction and resumable paired MATH-500 evaluation.

Use prepare once, then run each frozen arm in a separate process, then compare.
Pilot runs are permanently marked and must use a separate output directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.resolved.json").read_text(encoding="utf-8-sig"))
RESEARCH = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/" + PROTOCOL["research_revision"]


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(path)
    return {"url": url, "sha256": sha(path)}


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def reconstruct(tensors, manifest, edits):
    """Apply exact sparse BF16 replacements; reject wrong bases and changed counts."""
    import numpy as np
    import torch
    total = 0
    evidence = []
    for i, spec in enumerate(manifest["tensors"]):
        name = spec["name"]
        base = tensors[name]
        if base.dtype != torch.bfloat16 or list(base.shape) != spec["shape"]:
            raise ValueError(f"Wrong tensor shape/dtype: {name}")
        if tensor_sha(base) != spec["base_sha256"]:
            raise ValueError(f"Base tensor hash mismatch: {name}")
        if spec["form"] != "sparse":
            raise ValueError("This p50c runner accepts sparse exact replacements only")
        indices, values = edits[f"t{i}_idx"], edits[f"t{i}_val"]
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or values.dtype != np.uint16:
            raise ValueError("Invalid sparse edit arrays")
        if values.shape != indices.shape or len(np.unique(indices)) != len(indices):
            raise ValueError("Duplicate indices or mismatched values")
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= base.numel()):
            raise ValueError("Edit index out of bounds")
        candidate = base.clone()
        bits = candidate.view(torch.uint16).numpy().reshape(-1)
        bits[indices] = values
        changed = int(torch.count_nonzero(base.view(torch.int16) != candidate.view(torch.int16)))
        digest = tensor_sha(candidate)
        if digest != spec["edited_sha256"] or changed != spec["changed_entries"]:
            raise ValueError(f"Edited tensor hash/count mismatch: {name}")
        if not torch.isfinite(candidate.float()).all():
            raise ValueError("Nonfinite edited tensor")
        tensors[name] = candidate
        total += changed
        evidence.append({"name": name, "base_sha256": spec["base_sha256"],
                         "edited_sha256": digest, "changed_entries": changed})
    if total != manifest["changed_entries_total"]:
        raise ValueError("Total changed-entry count mismatch")
    return evidence


def prepare(root):
    import numpy as np
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from datasets import load_dataset
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record = root / "prepared.json"
    if record.exists():
        raise FileExistsError("Preparation exists; preserve it and use another directory")
    assets = {}
    files = ["README.md", "code/mathqa.py", "code/st_numpy.py", "code/apply_edit.py",
             "code/evaluate.py", "runs/p50c/edit/manifest.json", "runs/p50c/edit/edit.npz"]
    config_files = ["config.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"]
    files += ["base_model/Qwen3.5-0.8B-text/" + f for f in config_files]
    for name in files:
        assets[name] = fetch(RESEARCH + "/exp8_math/" + name, root / "reference" / name)
    assets["LICENSE"] = fetch(RESEARCH + "/LICENSE", root / "reference/LICENSE")
    snapshot = Path(snapshot_download(PROTOCOL["model"], revision=PROTOCOL["model_revision"],
                                     allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "LICENSE"]))
    weight_files = sorted(snapshot.glob("*.safetensors"))
    if len(weight_files) != 1:
        raise ValueError("Expected the pinned release's single safetensors shard")
    tensors = load_file(str(weight_files[0]))
    # The text-only config uses the release's model.language_model.* names.
    # Remove vision and auxiliary prediction tensors before inference.
    tensors = {k: v for k, v in tensors.items() if k.startswith("model.language_model.") or k == "lm_head.weight"}
    if not tensors:
        raise ValueError("No text-model tensors in release")
    for arm in ("baseline", "p50c"):
        out = root / arm
        out.mkdir(exist_ok=False)
        for name in ("tokenizer.json", "merges.txt", "vocab.json", "LICENSE"):
            shutil.copyfile(snapshot / name, out / name)
        for name in config_files:
            shutil.copyfile(root / "reference/base_model/Qwen3.5-0.8B-text" / name, out / name)
        if arm == "baseline":
            save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
        else:
            manifest = json.loads((root / "reference/runs/p50c/edit/manifest.json").read_text())
            with np.load(root / "reference/runs/p50c/edit/edit.npz", allow_pickle=False) as edits:
                evidence = reconstruct(tensors, manifest, edits)
            if sum(t["changed_entries"] for t in evidence) != PROTOCOL["expected_changed_entries"]:
                raise ValueError("Protocol changed-entry count differs")
            save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    ds = load_dataset(PROTOCOL["dataset"], revision=PROTOCOL["dataset_revision"], split="test")
    rows = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in ds]
    if len(rows) != 500 or len({r["id"] for r in rows}) != 500:
        raise ValueError("MATH-500 row-count/ID mismatch")
    write_json(root / "math500.json", rows)
    checksums = {str(p.relative_to(root)): sha(p) for arm in ("baseline", "p50c")
                 for p in sorted((root / arm).iterdir()) if p.is_file()}
    write_json(record, {"protocol": PROTOCOL, "assets": assets, "checkpoint_files": checksums,
                        "base_release_sha256": sha(weight_files[0]), "tensor_verification": evidence,
                        "dataset_sha256": sha(root / "math500.json"), "script_sha256": sha(__file__)})
    print(json.dumps({"prepared": str(root), "tensor_verification": evidence}), flush=True)


def last_boxed(text):
    """Match released mathqa.last_boxed, including its boxed-before-fbox precedence."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    start = text.rfind("\\boxed")
    key = "\\boxed"
    if start < 0:
        start, key = text.rfind("\\fbox"), "\\fbox"
    if start < 0:
        return None
    rest = text[start + len(key):]
    value = rest.lstrip()
    if value.startswith("{"):
        depth = 0
        for i, char in enumerate(value):
            depth += (char == "{") - (char == "}")
            if depth == 0:
                return value[1:i].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def grade(text, gold):
    from math_verify import parse, verify
    pred = last_boxed(text)
    if pred is None:
        return {"pred": None, "correct": False}
    g, p = parse("\\boxed{" + gold + "}", parsing_timeout=5), parse("\\boxed{" + pred + "}", parsing_timeout=5)
    try:
        correct = bool(g and p and verify(g, p, timeout_seconds=5))
    except Exception:
        # The released grader also scores failed/timed-out verification as incorrect.
        correct = False
    return {"pred": pred, "correct": correct}


def runtime_versions():
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}


def run(args):
    # Include the isolated environment's build helpers for Triton/FlashInfer.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    root, out = Path(args.root), Path(args.out)
    prepared = json.loads((root / "prepared.json").read_text())
    if prepared["protocol"] != PROTOCOL:
        raise ValueError("Prepared protocol changed")
    for name, expected in prepared["checkpoint_files"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Frozen checkpoint changed: {name}")
    if sha(root / "math500.json") != prepared["dataset_sha256"]:
        raise ValueError("Dataset changed")
    if not args.pilot and (args.limit != 500 or args.max_tokens != 32768):
        raise ValueError("Full evaluation must use 500 problems and 32768 new tokens")
    if not 1 <= args.limit <= 500 or not 1 <= args.max_tokens <= 32768:
        raise ValueError("Invalid pilot bounds")
    rows = json.loads((root / "math500.json").read_text())[:args.limit]
    sampling = {**PROTOCOL["sampling"], "n": 4, "max_tokens": args.max_tokens}
    out.mkdir(parents=True, exist_ok=True)
    job = {"arm": args.arm, "pilot": args.pilot, "problem_ids": [r["id"] for r in rows],
           "sampling": sampling, "protocol": PROTOCOL, "prepared_sha256": sha(root / "prepared.json"),
           "script_sha256": sha(__file__), "versions": runtime_versions(),
           "engine": {"dtype": "bfloat16", "max_model_len": 34816,
                      "gpu_memory_utilization": args.gpu_memory, "enforce_eager": args.eager,
                      "max_num_seqs": 16, "seed": 0,
                      "compilation_config": {"mode": 0, "cudagraph_mode": "NONE" if args.eager else "FULL_DECODE_ONLY",
                                             "cudagraph_capture_sizes": [] if args.eager else [1, 2, 4, 8, 16]}}}
    job["environment"] = {key: value for key, value in os.environ.items()
                          if key in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CC", "OMP_NUM_THREADS", "LIBRARY_PATH", "LD_LIBRARY_PATH")
                          or key.startswith("VLLM_")}
    job["batch_problems"] = 4
    job_path = out / "job.json"
    if job_path.exists() and json.loads(job_path.read_text()) != job:
        raise ValueError("Refusing to mix different jobs in the same output directory")
    write_json(job_path, job)
    shutil.copyfile(__file__, out / "runner-source.py")
    completed = []
    for row in rows:
        path = out / (hashlib.sha256(row["id"].encode()).hexdigest()[:20] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["id"] != row["id"] or len(saved["generations"]) != 4:
                raise ValueError("Invalid completed problem record")
            completed.append(row["id"])
    pending = [r for r in rows if r["id"] not in completed]
    if pending:
        tok = AutoTokenizer.from_pretrained(root / args.arm, trust_remote_code=False)
        llm = LLM(model=str(root / args.arm), **job["engine"])
        # Separate deterministic per-problem seeds make interruption/resume reproducible.
        # This differs from the original release's one whole-shard engine call.
        for offset in range(0, len(pending), job["batch_problems"]):
            batch = pending[offset:offset + job["batch_problems"]]
            seeds = [sampling["seed"] + job["problem_ids"].index(row["id"]) for row in batch]
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": row["problem"] + "\n" + PROTOCOL["prompt_suffix"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True) for row in batch]
            started = time.time()
            responses = llm.generate(prompts, [SamplingParams(**{**sampling, "seed": seed}) for seed in seeds])
            if len(responses) != len(batch):
                raise RuntimeError("Incomplete batch; no results saved")
            for row, prompt, seed, response in zip(batch, prompts, seeds, responses):
                if len(response.outputs) != 4 or any(s.finish_reason not in ("stop", "length") for s in response.outputs):
                    raise RuntimeError("Incomplete generation; no problem result saved")
                generations = [{"sample": s.index, "text": s.text, "token_ids": list(s.token_ids),
                                "tokens": len(s.token_ids), "finish_reason": s.finish_reason,
                                "truncated": s.finish_reason == "length", **grade(s.text, row["answer"])}
                               for s in response.outputs]
                result = {**row, "prompt": prompt, "prompt_token_ids": response.prompt_token_ids,
                          "seed": seed, "generations": generations, "batch_seconds": time.time() - started}
                path = out / (hashlib.sha256(row["id"].encode()).hexdigest()[:20] + ".json")
                write_json(path, result)
                print(json.dumps({"arm": args.arm, "id": row["id"], "correct": sum(g["correct"] for g in generations),
                                  "cut": sum(g["truncated"] for g in generations), "batch_seconds": result["batch_seconds"]}), flush=True)
    records = [json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.name not in ("job.json", "summary.json")]
    generations = [g for r in records for g in r["generations"]]
    summary = {"pilot": args.pilot, "arm": args.arm, "problems": len(records), "generations": len(generations),
               "accuracy": sum(g["correct"] for g in generations) / len(generations),
               "token_limit_share": sum(g["truncated"] for g in generations) / len(generations),
               "tokens": sum(g["tokens"] for g in generations)}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def compare(baseline, edited, out):
    import numpy as np
    dirs = [Path(baseline), Path(edited)]
    jobs = [json.loads((d / "job.json").read_text()) for d in dirs]
    if jobs[0]["arm"] != "baseline" or jobs[1]["arm"] != "p50c":
        raise ValueError("Expected baseline then p50c arms")
    for key in ("pilot", "problem_ids", "sampling", "protocol", "prepared_sha256", "script_sha256", "versions", "engine", "environment", "batch_problems"):
        if jobs[0][key] != jobs[1][key]:
            raise ValueError(f"Unmatched paired jobs: {key}")
    arms = []
    for d, job in zip(dirs, jobs):
        values = []
        for row_id in job["problem_ids"]:
            record = json.loads((d / (hashlib.sha256(row_id.encode()).hexdigest()[:20] + ".json")).read_text())
            if record["id"] != row_id or len(record["generations"]) != 4:
                raise ValueError("Incomplete paired evidence")
            values.append([sum(g[k] for g in record["generations"]) / 4 for k in ("correct", "truncated")])
        arms.append(np.asarray(values))
    matrix = np.concatenate([arms[0], arms[1], arms[1] - arms[0]], axis=1)
    rng = np.random.default_rng(0)
    boot = np.asarray([matrix[rng.integers(0, len(matrix), len(matrix))].mean(axis=0) for _ in range(10000)])
    labels = ["baseline_accuracy", "baseline_truncation", "p50c_accuracy", "p50c_truncation", "accuracy_delta", "truncation_delta"]
    report = {"pilot": jobs[0]["pilot"], "problems": len(matrix), "bootstrap_unit": "problem (all four samples together)",
              "bootstrap_replicates": 10000, "seed": 0,
              "protocol_difference": "Local calls group four problems with per-problem seed=ordinal; released evaluator calls a whole shard with seed=0. Local inference versions and hardware are recorded separately.",
              "metrics": {k: {"mean": float(matrix[:, i].mean()), "ci95": np.quantile(boot[:, i], [.025, .975]).tolist()}
                          for i, k in enumerate(labels)}}
    write_json(out, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("root")
    evaluation = sub.add_parser("run")
    evaluation.add_argument("root")
    evaluation.add_argument("arm", choices=["baseline", "p50c"])
    evaluation.add_argument("out")
    evaluation.add_argument("--pilot", action="store_true")
    evaluation.add_argument("--limit", type=int, default=500)
    evaluation.add_argument("--max-tokens", type=int, default=32768)
    evaluation.add_argument("--gpu-memory", type=float, default=.65)
    evaluation.add_argument("--eager", action="store_true", help="Disable CUDA graph capture for diagnosis")
    comparison = sub.add_parser("compare")
    comparison.add_argument("baseline")
    comparison.add_argument("edited")
    comparison.add_argument("out")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "run":
        run(args)
    else:
        compare(args.baseline, args.edited, args.out)


if __name__ == "__main__":
    main()
