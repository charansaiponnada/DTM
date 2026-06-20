"""
src/features.py
────────────────
Shared terrain feature computation utilities.
Centralises formulas used by multiple pipeline modules to avoid DRY violations.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np


def compute_curvature_evans(
    dem: np.ndarray,
    cell_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plan and Profile curvature using the Evans (1979) second-order
    polynomial surface fitting method.

    Parameters
    ----------
    dem       : 2D elevation array (float64)
    cell_size : grid resolution in map units (same as dem units)

    Returns
    -------
    plan_curv    : plan curvature (1/m)  — negative = concave convergence
    profile_curv : profile curvature (1/m) — negative = concave, decelerating flow
    """
    z = dem.astype(np.float64)
    cs = cell_size

    D = (np.roll(z, -1, axis=0) + np.roll(z, 1, axis=0) - 2 * z) / (2 * cs ** 2)
    E = (np.roll(z, -1, axis=1) + np.roll(z, 1, axis=1) - 2 * z) / (2 * cs ** 2)
    F = (
        -np.roll(np.roll(z, -1, 0), -1, 1)
        + np.roll(np.roll(z, -1, 0), 1, 1)
        + np.roll(np.roll(z, 1, 0), -1, 1)
        - np.roll(np.roll(z, 1, 0), 1, 1)
    ) / (4 * cs ** 2)
    G = (np.roll(z, 1, axis=1) - np.roll(z, -1, axis=1)) / (2 * cs)
    H = (np.roll(z, 1, axis=0) - np.roll(z, -1, axis=0)) / (2 * cs)

    p_sq = G ** 2 + H ** 2 + 1e-10
    numerator = -2 * (D * G ** 2 + E * H ** 2 + F * G * H)

    plan_curv = (numerator / p_sq).astype(np.float32)
    profile_curv = (numerator / (p_sq * np.sqrt(p_sq))).astype(np.float32)

    return plan_curv, profile_curv


def compute_aspect(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Aspect in degrees (0 = North, clockwise)."""
    dy, dx = np.gradient(dem.astype(np.float64), cell_size)
    return np.degrees(np.arctan2(-dx, dy)) % 360


def compute_tpi(dem: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Topographic Position Index: elevation minus mean elevation in window.
    Negative TPI indicates valleys / hollows (water convergence prone).
    """
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(dem.astype(np.float64), size=window)
    return (dem - local_mean).astype(np.float32)


FEATURE_NAMES_WL = [
    "elevation_normalized", "slope_deg", "aspect_deg",
    "twi", "tpi", "log_flow_accumulation",
    "plan_curvature", "profile_curvature",
    "depression_depth_m", "distance_to_stream_m",
]
