"""Load the existing model-specific helpers without changing their public CLIs."""
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills/weight-space-lab/scripts"


def helper(name):
    if name not in ("weight_lab", "reproduce_breakout"):
        raise ValueError(f"Unknown adapter: {name}")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)
