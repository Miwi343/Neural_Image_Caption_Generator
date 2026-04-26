"""Run the root training script from the requested code/ layout."""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

runpy.run_path(os.path.join(ROOT, "train.py"), run_name="__main__")
