"""Compatibility import for the implementation under code/."""

import importlib.util
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent / "code"

spec = importlib.util.spec_from_file_location("_sat_config", CODE_DIR / "config.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

for name in dir(module):
    if name.isupper():
        globals()[name] = getattr(module, name)
