"""Compatibility wrapper for the requested code/ layout."""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("_sat_config", os.path.join(ROOT, "config.py"))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

for name in dir(module):
    if name.isupper():
        globals()[name] = getattr(module, name)
