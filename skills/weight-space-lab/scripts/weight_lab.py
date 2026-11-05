#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "safetensors==0.8.0"]
# ///
"""Local, gradient-free checkpoint inspection, patching, and dense-network eval."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_checkpoint(path, max_mib=512):
    path = Path(path)
    if not math.isfinite(max_mib) or max_mib <= 0:
        raise ValueError("max-mib must be positive and finite")
    limit = max_mib * 1024 * 1024
    if path.stat().st_size > limit:
        raise ValueError("Checkpoint exceeds max-mib")
    metadata = None
    if path.suffix == ".npz":
        with zipfile.ZipFile(path) as archive:
            if sum(f.file_size for f in archive.infolist()) > limit:
                raise ValueError("Expanded NPZ exceeds max-mib")
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("Duplicate NPZ members")
        with np.load(path, allow_pickle=False) as archive:
            tensors = {name: archive[name] for name in archive.files}
    elif path.suffix == ".safetensors":
        with safe_open(path, framework="numpy") as archive:
            metadata = archive.metadata()
        tensors = load_file(path)
    else:
        raise ValueError("Use a single .npz or .safetensors checkpoint")
    if not tensors or sum(t.nbytes for t in tensors.values()) > limit:
        raise ValueError("Empty checkpoint or tensor bytes exceed max-mib")
    for name, tensor in tensors.items():
        if tensor.dtype.kind not in "fiu" or tensor.dtype.itemsize > 8:
            raise ValueError(f"Unsupported dtype for {name}: {tensor.dtype}")
        if not np.isfinite(tensor).all():
            raise ValueError(f"Non-finite source tensor: {name}")
    return tensors, metadata


def snapshot(path, max_mib=512):
    before = digest(path)
    tensors, metadata = load_checkpoint(path, max_mib)
    if digest(path) != before:
        raise ValueError("Checkpoint changed while being read")
    return tensors, metadata, before


def describe(path, max_mib=512):
    tensors, metadata, sha = snapshot(path, max_mib)
    rows = []
    for name, tensor in sorted(tensors.items()):
        values = tensor.astype(np.float64)
        row = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
               "elements": tensor.size}
        if tensor.size:
            row.update(min=float(values.min()), max=float(values.max()),
                       mean=float(values.mean()), zero_fraction=float((values == 0).mean()))
        rows.append(row)
    return {"sha256": sha, "parameters": sum(t.size for t in tensors.values()),
            "metadata": metadata, "tensors": rows}


def tensor_slice(path, name, start=0, count=64, max_mib=512):
    if start < 0 or not 1 <= count <= 4096:
        raise ValueError("start must be nonnegative; count must be 1..4096")
    tensors, _, sha = snapshot(path, max_mib)
    tensor = tensors[name]
    if start >= tensor.size:
        raise ValueError("start is outside tensor")
    return {"sha256": sha, "name": name, "shape": list(tensor.shape),
            "start": start, "values": tensor.reshape(-1)[start:start + count].tolist()}


def indices(value, size):
    if not isinstance(value, list) or not value:
        raise ValueError("indices must be a nonempty list")
    if any(type(i) is not int or not 0 <= i < size for i in value):
        raise ValueError("Index outside tensor or not an integer")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate indices")
    return value


def exact_keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Expected fields: {', '.join(expected)}")


def finite_array(value):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Values must be finite")
    return array


def edit_tensor(tensor, operation):
    if tensor.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Editing supports F16/F32/F64 only")
    kind = operation["op"]
    if kind in ("set", "add", "scale"):
        exact_keys(operation, ["op", "tensor", "indices", "values"])
        ix = indices(operation["indices"], tensor.size)
        values = finite_array(operation["values"])
        if values.shape != (len(ix),):
            raise ValueError("One value is required per flat index")
        flat = tensor.reshape(-1)
        old = flat[ix].astype(np.float64)
        result = values if kind == "set" else old + values if kind == "add" else old * values
        with np.errstate(over="ignore", invalid="ignore"):
            flat[ix] = result
    elif kind == "remove_direction":
        exact_keys(operation, ["op", "tensor", "columns", "direction", "strength"])
        if tensor.ndim != 2:
            raise ValueError("remove_direction requires a 2D [output, input] matrix")
        cols = indices(operation["columns"], tensor.shape[1])
        direction = finite_array(operation["direction"])
        if direction.shape != (tensor.shape[0],):
            raise ValueError("direction length must equal matrix output dimension")
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("direction must have a finite nonzero norm")
        strength = operation["strength"]
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        unit = direction / norm
        block = tensor[:, cols].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            tensor[:, cols] = block - strength * np.outer(unit, unit @ block)
    else:
        raise ValueError(f"Unknown operation: {kind}")
    if not np.isfinite(tensor).all():
        raise ValueError("Edit produced non-finite weights")


def patch_checkpoint(source, patch_file, output, max_fraction=0.01, max_mib=512, *, allow_noop=False):
    if not math.isfinite(max_fraction) or not 0 < max_fraction <= 1:
        raise ValueError("max-fraction must be in (0, 1]")
    spec = read_json(patch_file)
    exact_keys(spec, ["base_sha256", "reason", "operations"])
    if not isinstance(spec["reason"], str) or not spec["reason"].strip():
        raise ValueError("A nonempty edit reason is required")
    if not isinstance(spec["operations"], list) or not spec["operations"]:
        raise ValueError("At least one operation is required")
    original, metadata, sha = snapshot(source, max_mib)
    if spec["base_sha256"] != sha:
        raise ValueError("Base SHA-256 does not match checkpoint")
    edited = {name: tensor.copy(order="C") for name, tensor in original.items()}
    for operation in spec["operations"]:
        edit_tensor(edited[operation["tensor"]], operation)
    changes = {name: int(np.count_nonzero(edited[name] != tensor))
               for name, tensor in original.items()}
    changed = sum(changes.values())
    total = sum(t.size for t in original.values())
    if not total or (not changed and not allow_noop):
        raise ValueError("Patch changes no stored weights")
    fraction = changed / total
    if fraction > max_fraction:
        raise ValueError(f"Changed fraction {fraction:.6f} exceeds {max_fraction}")
    if digest(source) != sha:
        raise ValueError("Source changed during patch preparation")
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    try:
        target = output / ("model" + Path(source).suffix)
        if target.suffix == ".npz":
            np.savez(target, **edited)
        else:
            save_file(edited, target, metadata=metadata)
        receipt = {"base_sha256": sha, "candidate_sha256": digest(target),
                   "changed": changed, "total": total, "fraction": fraction,
                   "changed_tensors": {k: v for k, v in changes.items() if v},
                   "reason": spec["reason"], "checkpoint": target.name,
                   "rollback": "Use the untouched source checkpoint."}
        write_json(output / "patch.json", spec)
        write_json(output / "receipt.json", receipt)
    except BaseException:
        shutil.rmtree(output)
        raise
    return receipt


def evaluate_mlp(checkpoint, architecture, dataset, max_mib=512):
    tensors, _, sha = snapshot(checkpoint, max_mib)
    arch = read_json(architecture)
    data = read_json(dataset)
    exact_keys(arch, ["layers"])
    exact_keys(data, ["split", "examples"])
    if data["split"] not in ("practice", "heldout") or not data["examples"]:
        raise ValueError("Dataset requires practice/heldout split and examples")
    if not isinstance(arch["layers"], list) or not arch["layers"]:
        raise ValueError("At least one dense layer is required")
    ids, inputs, labels = [], [], []
    for row in data["examples"]:
        exact_keys(row, ["id", "x", "y"])
        if not isinstance(row["id"], str) or not row["id"] or type(row["y"]) is not int:
            raise ValueError("Each example needs a string id and integer label")
        ids.append(row["id"])
        inputs.append(row["x"])
        labels.append(row["y"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate example ids")
    x = finite_array(inputs)
    if x.ndim != 2:
        raise ValueError("Inputs must be a batch of 1D feature vectors")
    for layer in arch["layers"]:
        exact_keys(layer, ["weight", "bias", "activation"])
        w = tensors[layer["weight"]]
        b = tensors[layer["bias"]]
        if w.ndim != 2 or x.shape[1] != w.shape[1] or b.shape != (w.shape[0],):
            raise ValueError("Dense weights must be [output, input], biases [output]")
        x = x @ w.astype(np.float64).T + b.astype(np.float64)
        activation = layer["activation"]
        if activation == "relu":
            x = np.maximum(x, 0)
        elif activation == "tanh":
            x = np.tanh(x)
        elif activation != "linear":
            raise ValueError("Supported activations: linear, relu, tanh")
        if not np.isfinite(x).all():
            raise ValueError("Evaluation produced non-finite activations")
    if x.shape[1] < 2 or any(not 0 <= y < x.shape[1] for y in labels):
        raise ValueError("Need >=2 output logits and labels within class range")
    predicted = x.argmax(axis=1)
    rows = [{"id": i, "label": y, "prediction": int(p), "correct": bool(y == p)}
            for i, y, p in zip(ids, labels, predicted)]
    return {"schema": "weight-space-mlp-eval-v1", "checkpoint_sha256": sha,
            "architecture_sha256": digest(architecture), "dataset_sha256": digest(dataset),
            "split": data["split"], "settings": {"dtype": "float64", "decision": "argmax"},
            "samples": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "examples": rows}


def compare_reports(baseline_file, candidate_file):
    a, b = read_json(baseline_file), read_json(candidate_file)
    for field in ("schema", "architecture_sha256", "dataset_sha256", "split", "settings", "samples"):
        if a[field] != b[field]:
            raise ValueError(f"Incompatible evaluation field: {field}. Evaluate both checkpoints with the same architecture, dataset, split, and settings; no artifacts were changed.")
    if a["schema"] != "weight-space-mlp-eval-v1":
        raise ValueError("Unsupported evaluation schema")
    fixed, regressions, correct = [], [], [0, 0]
    if not a["examples"] or len(a["examples"]) != len(b["examples"]):
        raise ValueError("Evaluation example counts differ or are empty")
    seen = set()
    for old, new in zip(a["examples"], b["examples"]):
        if old["id"] != new["id"] or old["label"] != new["label"] or old["id"] in seen:
            raise ValueError("Evaluation ids/labels differ or repeat")
        seen.add(old["id"])
        prev, now = old["label"] == old["prediction"], new["label"] == new["prediction"]
        correct[0] += prev
        correct[1] += now
        if now and not prev:
            fixed.append(old["id"])
        if prev and not now:
            regressions.append(old["id"])
    n = len(seen)
    if a["samples"] != n:
        raise ValueError("Sample count disagrees with examples")
    return {"baseline_sha256": a["checkpoint_sha256"], "candidate_sha256": b["checkpoint_sha256"],
            "split": a["split"], "samples": n, "baseline_accuracy": correct[0] / n,
            "candidate_accuracy": correct[1] / n, "accuracy_delta": (correct[1] - correct[0]) / n,
            "fixed_ids": fixed, "regressed_ids": regressions,
            "interpretation": "Observed on this dataset only; not evidence of general capability gain."}


def demo(output):
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    return demo_files(output)


def demo_files(output):
    """Populate a newly owned empty directory; used by the shared workflow."""
    output = Path(output)
    source = output / "original.npz"
    np.savez(source, weight=np.array([[1.], [-1.]], dtype=np.float32), bias=np.zeros(2, dtype=np.float32))
    architecture, dataset, patch = (output / f for f in ("architecture.json", "heldout.json", "edit.json"))
    write_json(architecture, {"layers": [{"weight": "weight", "bias": "bias", "activation": "linear"}]})
    write_json(dataset, {"split": "heldout", "examples": [
        {"id": f"sign-{i}", "x": [x], "y": int(x > 0)}
        for i, x in enumerate([-4.7, -2.1, -0.3, 0.2, 1.6, 5.9])]})
    write_json(patch, {"base_sha256": digest(source),
                       "reason": "Hand-authored demo: reverse inverted sign classifier logits.",
                       "operations": [{"op": "scale", "tensor": "weight", "indices": [0, 1], "values": [-1, -1]}]})
    before = evaluate_mlp(source, architecture, dataset)
    receipt = patch_checkpoint(source, patch, output / "candidate", max_fraction=0.5)
    after = evaluate_mlp(output / "candidate/model.npz", architecture, dataset)
    write_json(output / "baseline.json", before)
    write_json(output / "candidate.json", after)
    comparison = compare_reports(output / "baseline.json", output / "candidate.json")
    write_json(output / "comparison.json", comparison)
    return {"demo": "Deterministic plumbing test; patch was authored in advance, not discovered by an agent.",
            "receipt": receipt, "comparison": comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "tensor", "patch", "evaluate-mlp"):
        p = commands.add_parser(command)
        p.add_argument("checkpoint", type=Path)
        p.add_argument("--max-mib", type=float, default=512)
        if command == "tensor":
            p.add_argument("name")
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--count", type=int, default=64)
        elif command == "patch":
            p.add_argument("patch_file", type=Path)
            p.add_argument("output", type=Path)
            p.add_argument("--max-fraction", type=float, default=0.01)
        elif command == "evaluate-mlp":
            p.add_argument("architecture", type=Path)
            p.add_argument("dataset", type=Path)
            p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("compare")
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p = commands.add_parser("demo")
    p.add_argument("output", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if command == "inspect":
            result = describe(args.pop("checkpoint"), **args)
        elif command == "tensor":
            result = tensor_slice(args.pop("checkpoint"), **args)
        elif command == "patch":
            result = patch_checkpoint(args.pop("checkpoint"), **args)
        elif command == "evaluate-mlp":
            output = args.pop("output")
            result = evaluate_mlp(**args)
            write_json(output, result)
        elif command == "compare":
            result = compare_reports(args["baseline"], args["candidate"])
        else:
            result = demo(**args)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
