"""Compatibility import for code/model.py."""

import importlib.util
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent / "code"
code_path = str(CODE_DIR)
if code_path not in sys.path:
    sys.path.insert(0, code_path)

spec = importlib.util.spec_from_file_location("_sat_model", CODE_DIR / "model.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

ShowAttendTell = module.ShowAttendTell

__all__ = ["ShowAttendTell"]
