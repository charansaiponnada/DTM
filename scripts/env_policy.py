from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_running_in_workspace_venv() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    expected_venv = (workspace_root / ".venv").resolve()

    if not expected_venv.exists():
        raise RuntimeError(
            f"Expected virtual environment folder not found: {expected_venv}. "
            "Create it with: python -m venv .venv"
        )

    current_prefix = Path(sys.prefix).resolve()
    in_any_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("VIRTUAL_ENV"))
    in_workspace_venv = str(current_prefix).lower().startswith(str(expected_venv).lower())

    if not in_any_venv or not in_workspace_venv:
        raise RuntimeError(
            "This project must be executed using the workspace virtual environment only. "
            "Use: .venv\\Scripts\\python.exe <script.py ...>"
        )
