"""Tests for DTM generation and terrain analysis."""
import numpy as np
from src.dtm.dtm_generator import get_dtm_stats


def test_get_dtm_stats_nonexistent():
    """Should return error dict for missing file."""
    stats = get_dtm_stats("nonexistent.tif")
    assert "error" in stats


def test_get_dtm_stats_synthetic(tmp_path):
    """Should compute coverage and elevation stats."""
    import rasterio
    shape = (20, 20)
    transform = rasterio.transform.from_bounds(0, 0, 10, 10, 20, 20)
    dem = np.random.rand(*shape).astype(np.float32) * 50 + 100
    dem[0:3, :] = -9999  # nodata strip
    path = tmp_path / "dtm.tif"
    with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="float32", crs="EPSG:32643",
                       transform=transform, nodata=-9999.0) as dst:
        dst.write(dem, 1)
    stats = get_dtm_stats(path)
    assert "min_elevation_m" in stats
    assert "max_elevation_m" in stats
    assert "mean_elevation_m" in stats
    assert "std_elevation_m" in stats
    assert "nodata_pct" in stats
    assert stats["nodata_pct"] > 0  # nodata strip
