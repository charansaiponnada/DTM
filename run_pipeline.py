#!/usr/bin/env python3
"""
run_pipeline.py
───────────────
Thin CLI entry point. Real logic lives in src/cli.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main  # noqa: E402, F401 — re-export for backward compat

if __name__ == "__main__":
    main()
