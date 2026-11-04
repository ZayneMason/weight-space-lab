"""Independently compare the two complete exported checkpoint files."""
import json
from pathlib import Path
import sys

from safetensors import safe_open
import torch

root = Path(sys.argv[1])
manifest = json.loads((root / "reference/runs/p50c/edit/manifest.json").read_text())
expected = {t["name"]: t["changed_entries"] for t in manifest["tensors"]}
changed = {}
with safe_open(root / "baseline/model.safetensors", framework="pt") as base:
    with safe_open(root / "p50c/model.safetensors", framework="pt") as edited:
        if set(base.keys()) != set(edited.keys()):
            raise ValueError("Checkpoint tensor inventories differ")
        for name in base.keys():
            a, b = base.get_tensor(name), edited.get_tensor(name)
            if a.dtype != b.dtype or a.shape != b.shape:
                raise ValueError(f"Shape/dtype changed: {name}")
            if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                if a.dtype != torch.bfloat16:
                    raise ValueError(f"Unexpected changed dtype: {name}")
                changed[name] = int(torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)))
        if changed != expected:
            raise ValueError(f"Unexpected exported changes: {changed}")
        result = {"tensor_count": len(base.keys()), "changed": changed,
                  "unchanged_tensors_bitwise_equal": len(base.keys()) - len(changed),
                  "total_changed_entries": sum(changed.values())}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
