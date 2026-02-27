from __future__ import annotations

import os


def ensure_running_in_conda_env() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    in_conda_env = bool(conda_prefix) and bool(conda_name)

    if not in_conda_env:
        raise RuntimeError(
            "This project must be executed using Conda environment only. "
            "Use: conda activate dataset-dtm"
        )
