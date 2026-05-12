"""Run or import the training code from code/train.py."""

import importlib.util
import runpy
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent / "code"


def _use_code_dir() -> None:
    code_path = str(CODE_DIR)
    if code_path not in sys.path:
        sys.path.insert(0, code_path)


if __name__ == "__main__":
    _use_code_dir()
    runpy.run_path(str(CODE_DIR / "train.py"), run_name="__main__")
else:
    _use_code_dir()
    spec = importlib.util.spec_from_file_location("_sat_train", CODE_DIR / "train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    for name in dir(module):
        if not name.startswith("_"):
            globals()[name] = getattr(module, name)
