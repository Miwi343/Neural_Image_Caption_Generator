"""Compatibility import for code/models/encoder.py."""

import importlib.util
from pathlib import Path

CODE_MODELS = Path(__file__).resolve().parents[1] / "code" / "models"

spec = importlib.util.spec_from_file_location("_sat_encoder", CODE_MODELS / "encoder.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

Encoder = module.Encoder

__all__ = ["Encoder"]
