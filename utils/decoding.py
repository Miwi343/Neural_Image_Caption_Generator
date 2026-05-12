"""Compatibility import for code/utils/decoding.py."""

import importlib.util
from pathlib import Path

CODE_UTILS = Path(__file__).resolve().parents[1] / "code" / "utils"

spec = importlib.util.spec_from_file_location("_sat_decoding", CODE_UTILS / "decoding.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

for name in dir(module):
    if not name.startswith("_"):
        globals()[name] = getattr(module, name)
