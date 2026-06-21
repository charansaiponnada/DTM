"""Tests for ground_classifier geometric feature extraction."""
import numpy as np
from src.preprocessing.ground_classifier import compute_geometric_features


def test_compute_geometric_features_shape():
    """Should return 12 features per point."""
    np.random.seed(42)
    xyz = np.random.rand(100, 3).astype(np.float64)
    features = compute_geometric_features(xyz, k=8, radius_density=0.5)
    assert features.shape == (100, 12), f"Expected (100, 12), got {features.shape}"


def test_compute_geometric_features_no_nan():
    """All 12 features should be finite (no NaN/Inf)."""
    np.random.seed(1)
    xyz = np.random.rand(50, 3).astype(np.float64)
    features = compute_geometric_features(xyz)
    assert np.all(np.isfinite(features)), "Non-finite values in features"


def test_compute_geometric_features_planar_points():
    """Points on a plane: planarity should be high."""
    np.random.seed(0)
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)  # perfectly flat
    xyz = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
    features = compute_geometric_features(xyz, k=4, radius_density=1.0)
    planarity = features[:, 4]
    assert np.median(planarity) > 0.5, f"Low planarity on flat surface: {np.median(planarity):.3f}"


def test_compute_geometric_features_linear_points():
    """Points on a line: linearity should be high."""
    np.random.seed(0)
    t = np.linspace(0, 10, 30)
    xyz = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)]).astype(np.float64)
    features = compute_geometric_features(xyz, k=4, radius_density=1.0)
    linearity = features[:, 5]
    assert np.median(linearity) > 0.5, f"Low linearity on line: {np.median(linearity):.3f}"


def test_compute_geometric_features_small_pointset():
    """Should handle tiny point sets without crashing."""
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    features = compute_geometric_features(xyz, k=2, radius_density=0.5)
    assert features.shape == (3, 12)
    assert np.all(np.isfinite(features))
