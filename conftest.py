"""
conftest.py — pytest root configuration.

Inserts the project root into ``sys.path`` so test modules can import
project packages directly (e.g. ``from sentiment_analyzer import ...``)
without requiring an installable package or PYTHONPATH setup.

This file is picked up automatically by pytest when it is placed in the
project root directory alongside the source modules.
"""

import sys
from pathlib import Path

# Ensure the project root is always importable, regardless of the working
# directory from which pytest is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))
