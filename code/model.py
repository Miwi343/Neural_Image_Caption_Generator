"""Compatibility wrapper for the requested code/ layout."""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("_sat_model", os.path.join(ROOT, "model.py"))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

ShowAttendTell = module.ShowAttendTell

__all__ = ["ShowAttendTell"]
