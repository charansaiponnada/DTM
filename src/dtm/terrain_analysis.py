"""
src/dtm/terrain_analysis.py
─────────────────────────────
Compute terrain derivative rasters from a DTM COG:
  - Slope (degrees)
  - Aspect (degrees, N=0 clockwise)
  - Plan curvature
  - Profile curvature
  - Topographic Position Index (TPI)
  - Roughness Index
  - Hill-shade (for visualization)

All outputs are written as COG.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from scipy.ndimage import uniform_filter
from loguru import logger

from src.dtm.dtm_generator import write_geotiff, convert_to_cog, NODATA


def compute_all_derivatives(
    dtm_path: str | Path,
    output_dir: str | Path,
    crs: str = "EPSG:32643",
    azimuth: float = 315.0,    # hill-shade sun direction (NW)
    altitude: float = 45.0,    # hill-shade sun elevation
) -> Dict[str, Path]:
    """
    Compute and export all terrain derivative rasters.

    Parameters
    ----------
    dtm_path   : path to input DTM COG
    output_dir : directory for output COGs
    crs        : coordinate reference system string
    azimuth    : sun azimuth for hillshade (degrees)
    altitude   : sun altitude for hillshade (degrees)

    Returns
    -------
    Dict mapping layer name → output COG path
    """
    dtm_path   = Path(dtm_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dtm_path) as src:
        dem       = src.read(1).astype(np.float32)
        transform = src.transform
        cell_size = float(src.res[0])
        nodata    = src.nodata or NODATA
        crs_obj   = src.crs or CRS.from_epsg(int(crs.split(":")[-1]))

    valid = (dem != nodata) & np.isfinite(dem)
    dem   = np.where(valid, dem, 0.0)

    paths: Dict[str, Path] = {}

    # ── Gradient ─────────────────────────────────────────────────────────
    dy, dx = np.gradient(dem.astype(np.float64), cell_size)

    # ── Slope ─────────────────────────────────────────────────────────────
    slope_rad = np.arctan(np.hypot(dx, dy))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_deg[~valid] = NODATA
    paths["slope"] = _save_cog(slope_deg, transform, crs_obj, output_dir / "slope.tif", "slope_degrees")

    # ── Aspect ────────────────────────────────────────────────────────────
    aspect_deg = (np.degrees(np.arctan2(-dx, dy)) % 360).astype(np.float32)
    aspect_deg[~valid] = NODATA
    paths["aspect"] = _save_cog(aspect_deg, transform, crs_obj, output_dir / "aspect.tif", "aspect_degrees")

    # ── Curvature (Evans method) ──────────────────────────────────────────
    cs = cell_size
    z  = dem.astype(np.float64)
    D  = (np.roll(z,-1,0) + np.roll(z,1,0) - 2*z) / (2*cs**2)
    E  = (np.roll(z,-1,1) + np.roll(z,1,1) - 2*z) / (2*cs**2)
    F  = (-np.roll(np.roll(z,-1,0),-1,1) + np.roll(np.roll(z,-1,0),1,1)
          + np.roll(np.roll(z,1,0),-1,1) - np.roll(np.roll(z,1,0),1,1)) / (4*cs**2)
    G  = (np.roll(z, 1,1) - np.roll(z,-1,1)) / (2*cs)
    H  = (np.roll(z, 1,0) - np.roll(z,-1,0)) / (2*cs)
    p  = G**2 + H**2 + 1e-10

    plan_curv    = (-2*(D*G**2 + E*H**2 + F*G*H) / p).astype(np.float32)
    profile_curv = (-2*(D*G**2 + E*H**2 + F*G*H) / (p * np.sqrt(p))).astype(np.float32)

    plan_curv[~valid]    = NODATA
    profile_curv[~valid] = NODATA
    paths["plan_curvature"]    = _save_cog(plan_curv, transform, crs_obj, output_dir/"plan_curvature.tif", "plan_curvature")
    paths["profile_curvature"] = _save_cog(profile_curv, transform, crs_obj, output_dir/"profile_curvature.tif", "profile_curvature")

    # ── TPI (Topographic Position Index) ─────────────────────────────────
    for window in [15, 51]:
        local_mean = uniform_filter(dem, size=window)
        tpi = (dem - local_mean).astype(np.float32)
        tpi[~valid] = NODATA
        key = f"tpi_{window}"
        paths[key] = _save_cog(tpi, transform, crs_obj, output_dir / f"tpi_{window}.tif", f"tpi_w{window}")

    # ── Roughness ─────────────────────────────────────────────────────────
    local_range = (
        uniform_filter(dem, size=5) - uniform_filter(-dem, size=5)
    ).astype(np.float32)
    local_range[~valid] = NODATA
    paths["roughness"] = _save_cog(local_range, transform, crs_obj, output_dir / "roughness.tif", "roughness_m")

    # ── Hillshade (for visualization) ─────────────────────────────────────
    azimuth_r  = np.radians(360.0 - azimuth + 90)
    altitude_r = np.radians(altitude)
    hillshade  = (
        np.sin(altitude_r) * np.cos(slope_rad) +
        np.cos(altitude_r) * np.sin(slope_rad) * np.cos(azimuth_r - np.arctan2(-dx, dy))
    )
    hillshade = (np.clip(hillshade, 0, 1) * 255).astype(np.float32)
    hillshade[~valid] = NODATA
    paths["hillshade"] = _save_cog(hillshade, transform, crs_obj, output_dir / "hillshade.tif", "hillshade")

    logger.success(f"Terrain derivatives computed: {list(paths.keys())}")
    return paths


def _save_cog(
    arr: np.ndarray,
    transform,
    crs_obj,
    cog_path: Path,
    band_name: str,
) -> Path:
    tmp = cog_path.parent / f"_{cog_path.stem}_tmp.tif"
    write_geotiff(arr, transform, tmp, crs=crs_obj, band_name=band_name, nodata=NODATA)
    return convert_to_cog(tmp, cog_path=cog_path)
