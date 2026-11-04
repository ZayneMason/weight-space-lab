#!/usr/bin/env python3
"""Read-only remote GPU inventory. No installs, downloads, or model execution."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command(argv, timeout=30):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def inspect(workdir):
    target = Path(workdir).expanduser().resolve()
    existing = target
    while not existing.exists():
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    versions = {}
    for name in ("torch", "vllm", "transformers", "safetensors", "numpy",
                 "huggingface-hub", "datasets", "math-verify",
                 "latex2sympy2_extended", "antlr4-python3-runtime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "purpose": "Inventory only; this does not certify inference compatibility.",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "python_executable": sys.executable,
        "working_directory": str(target), "directory_exists": target.is_dir(),
        "disk_free_bytes": usage.free, "packages": versions,
        "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                        "--format=csv,noheader,nounits"]),
        "torch_cuda": command([sys.executable, "-c",
            "import json, torch; print(json.dumps({'torch': torch.__version__, "
            "'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), "
            "'device_count': torch.cuda.device_count()}))"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.workdir), indent=2))
