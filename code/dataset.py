"""Compatibility wrapper for the requested code/ layout."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.dataset import *  # noqa: F401,F403
