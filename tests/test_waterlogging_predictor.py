"""Tests for waterlogging_predictor feature building and utilities."""
import numpy as np
import rasterio
from pathlib import Path
from src.hydrology.waterlogging_predictor import (
    read_terrain_rasters,
    compute_depression_depth,
    build_feature_stack,
)


def test_read_terrain_rasters_returns_seven_values(tmp_path):
    """Should return dem, twi, log_acc, slope, valid_mask, transform, cell_size."""
    # Create minimal synthetic rasters
    shape = (10, 10)
    transform = rasterio.transform.from_bounds(0, 0, 5, 5, 10, 10)
    crs = "EPSG:32643"

    def _write(path, data):
        with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1],
                           count=1, dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(data.astype(np.float32), 1)

    dem = np.random.rand(*shape).astype(np.float32) * 50 + 100
    twi = np.random.rand(*shape).astype(np.float32) * 10
    acc = np.random.rand(*shape).astype(np.float32) * 1000
    slope = np.random.rand(*shape).astype(np.float32) * 30

    dtm_p = tmp_path / "dem.tif"
    twi_p = tmp_path / "twi.tif"
    acc_p = tmp_path / "acc.tif"
    slp_p = tmp_path / "slp.tif"
    _write(dtm_p, dem)
    _write(twi_p, twi)
    _write(acc_p, acc)
    _write(slp_p, slope)

    result = read_terrain_rasters(dtm_p, twi_p, acc_p, slp_p)
    assert len(result) == 7, f"Expected 7 values, got {len(result)}"
    r_dem, r_twi, r_log_acc, r_slope, r_valid, r_transform, r_cs = result
    assert r_dem.shape == shape
    assert r_twi.shape == shape
    assert r_log_acc.shape == shape
    assert r_slope.shape == shape
    assert r_valid.shape == shape
    assert r_valid.dtype == bool
    assert np.allclose(r_cs, 0.5)


def test_compute_depression_depth_flat(tmp_path):
    """On flat terrain, depression depth should be zero."""
    shape = (10, 10)
    transform = rasterio.transform.from_bounds(0, 0, 5, 5, 10, 10)
    dem = np.ones(shape, dtype=np.float32) * 100
    dtm_p = tmp_path / "dem.tif"
    with rasterio.open(dtm_p, "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="float32", crs="EPSG:32643", transform=transform) as dst:
        dst.write(dem, 1)
    valid = np.ones(shape, dtype=bool)
    dep = compute_depression_depth(dtm_p, tmp_path, valid, dem)
    assert dep.shape == shape
    assert np.allclose(dep, 0.0, atol=1e-4), "Expected zero depression on flat terrain"


def test_compute_depression_depth_depression(tmp_path):
    """A depression should yield positive depression depth."""
    shape = (10, 10)
    transform = rasterio.transform.from_bounds(0, 0, 5, 5, 10, 10)
    dem = np.ones(shape, dtype=np.float32) * 100
    dem[5, 5] = 95  # depression
    dtm_p = tmp_path / "dem.tif"
    with rasterio.open(dtm_p, "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="float32", crs="EPSG:32643", transform=transform) as dst:
        dst.write(dem, 1)
    valid = np.ones(shape, dtype=bool)
    dep = compute_depression_depth(dtm_p, tmp_path, valid, dem)
    assert dep.shape == shape
    assert dep[5, 5] > 0, f"Expected positive depression depth, got {dep[5, 5]}"


def test_build_feature_stack_shape(tmp_path):
    """build_feature_stack should return stack, mask, transform."""
    shape = (10, 10)
    transform = rasterio.transform.from_bounds(0, 0, 5, 5, 10, 10)
    def _write(path, data):
        with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1],
                           count=1, dtype="float32", crs="EPSG:32643", transform=transform) as dst:
            dst.write(data.astype(np.float32), 1)
    dem = np.random.rand(*shape).astype(np.float32) * 50 + 100
    twi = np.random.rand(*shape).astype(np.float32) * 10
    acc = np.random.rand(*shape).astype(np.float32) * 1000
    slope = np.random.rand(*shape).astype(np.float32) * 30
    dtm_p = tmp_path / "dem.tif"; twi_p = tmp_path / "twi.tif"
    acc_p = tmp_path / "acc.tif"; slp_p = tmp_path / "slp.tif"
    _write(dtm_p, dem); _write(twi_p, twi); _write(acc_p, acc); _write(slp_p, slope)

    stack, mask, xform = build_feature_stack(dtm_p, twi_p, acc_p, slp_p)
    assert stack.shape == (10, 10, 10), f"Got {stack.shape}"
    assert mask.shape == (10, 10)
    assert mask.dtype == bool


def test_build_feature_stack_norm_stats(tmp_path):
    """Passing elev_norm_stats should produce same result as default."""
    shape = (10, 10)
    transform = rasterio.transform.from_bounds(0, 0, 5, 5, 10, 10)
    def _write(path, data):
        with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1],
                           count=1, dtype="float32", crs="EPSG:32643", transform=transform) as dst:
            dst.write(data.astype(np.float32), 1)
    dem = np.random.rand(*shape).astype(np.float32) * 50 + 100
    twi = np.random.rand(*shape).astype(np.float32) * 10
    acc = np.random.rand(*shape).astype(np.float32) * 1000
    slope = np.random.rand(*shape).astype(np.float32) * 30
    dtm_p = tmp_path / "dem.tif"; twi_p = tmp_path / "twi.tif"
    acc_p = tmp_path / "acc.tif"; slp_p = tmp_path / "slp.tif"
    _write(dtm_p, dem); _write(twi_p, twi); _write(acc_p, acc); _write(slp_p, slope)

    # Compute reference stats from the DEM
    stack_default, mask, xform = build_feature_stack(dtm_p, twi_p, acc_p, slp_p)
    e_mean = float(dem[dem > -9999].mean())
    e_std = float(dem[dem > -9999].std())
    stack_with_stats, _, _ = build_feature_stack(dtm_p, twi_p, acc_p, slp_p, elev_norm_stats=(e_mean, e_std))
    # F0 = elevation_normalized should be identical
    assert np.allclose(stack_default[:, :, 0], stack_with_stats[:, :, 0], atol=1e-4)
