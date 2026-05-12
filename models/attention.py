"""Compatibility import for code/models/attention.py."""

import importlib.util
from pathlib import Path

CODE_MODELS = Path(__file__).resolve().parents[1] / "code" / "models"

spec = importlib.util.spec_from_file_location("_sat_attention", CODE_MODELS / "attention.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

Attention = module.Attention

__all__ = ["Attention"]
