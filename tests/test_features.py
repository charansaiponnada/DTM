import numpy as np
from src.features import (
    compute_curvature_evans,
    compute_aspect,
    compute_tpi,
    FEATURE_NAMES_WL,
)


def test_feature_names_wl_structure():
    assert len(FEATURE_NAMES_WL) == 10
    for name in (
        "elevation_normalized", "slope_deg", "aspect_deg",
        "twi", "tpi", "log_flow_accumulation",
        "plan_curvature", "profile_curvature",
        "depression_depth_m", "distance_to_stream_m",
    ):
        assert name in FEATURE_NAMES_WL


def test_compute_curvature_evans_flat():
    z = np.ones((5, 5), dtype=float)
    plan, prof = compute_curvature_evans(z, cell_size=1.0)
    assert plan.shape == z.shape
    assert np.allclose(plan, 0.0, atol=1e-10)
    assert np.allclose(prof, 0.0, atol=1e-10)


def test_compute_curvature_evans_parabolic():
    x = np.linspace(-2, 2, 5)
    y = np.linspace(-2, 2, 5)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2
    plan, prof = compute_curvature_evans(z, cell_size=1.0)
    assert plan.shape == z.shape
    assert not np.allclose(plan, 0.0)
    assert not np.allclose(prof, 0.0)


def test_compute_aspect():
    z = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
    ], dtype=float)
    aspect = compute_aspect(z, cell_size=1.0)
    assert aspect.shape == z.shape
    assert np.all(aspect >= 0) and np.all(aspect < 360)


def test_compute_tpi():
    z = np.arange(25, dtype=float).reshape(5, 5)
    tpi = compute_tpi(z, window=3)
    assert tpi.shape == z.shape
