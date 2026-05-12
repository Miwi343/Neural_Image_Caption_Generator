"""Compatibility import for code/models/decoder.py."""

import importlib.util
from pathlib import Path

CODE_MODELS = Path(__file__).resolve().parents[1] / "code" / "models"

spec = importlib.util.spec_from_file_location("_sat_decoder", CODE_MODELS / "decoder.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

Decoder = module.Decoder
PaperDropout = module.PaperDropout

__all__ = ["Decoder", "PaperDropout"]
