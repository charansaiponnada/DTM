"""
src/evaluation/dtm_metrics.py
──────────────────────────────
Evaluate DTM vertical accuracy against a reference elevation model.

Reference options (in order of preference):
  1. User-supplied higher-resolution or independently validated raster.
  2. SRTM 1-arc-second (30 m) resampled – free global reference via rasterio.
     Download with: `aws s3 cp s3://elevation-tiles-prod/...` or USGS EarthExplorer.
  3. Flat-plane check: stddev of residuals over confirmed flat areas.

Metrics reported
----------------
  RMSE, MAE, ME (mean error / bias), std_error, NMAD (normalized MAD),
  LE90 (linear error at 90th percentile – ASPRS Accuracy Standard),
  n_check_points
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
except ImportError:
    raise ImportError("rasterio is required: pip install rasterio")


def download_srtm_for_extent(
    dtm_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """
    Download SRTM 30m (1 arc-second) data for the bounding box of a DTM
    using OpenTopography API. Falls back gracefully if no internet.

    Parameters
    ----------
    dtm_path    : path to DTM raster
    output_path : output path for SRTM GeoTIFF

    Returns
    -------
    Path to downloaded SRTM file, or None if download failed
    """
    import rasterio
    from rasterio.warp import transform_bounds
    import urllib.request
    import json

    dtm_path = Path(dtm_path)
    if output_path is None:
        output_path = dtm_path.parent / "_srtm_ref.tif"
    output_path = Path(output_path)

    if output_path.exists():
        logger.info(f"SRTM reference already cached: {output_path}")
        return output_path

    with rasterio.open(dtm_path) as src:
        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds
        )

    url = (
        f"https://portal.opentopography.org/API/globaldem?dem=SRTM&"
        f"south={south}&north={north}&west={west}&east={east}&"
        f"output=GTiff&API_KEY=demo"
    )

    logger.info(f"Downloading SRTM 30m for extent ({west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}) …")
    try:
        urllib.request.urlretrieve(url, str(output_path))
        logger.success(f"SRTM downloaded → {output_path}")
        return output_path
    except Exception as exc:
        logger.warning(f"SRTM download failed: {exc}. Will use internal flat-plane check instead.")
        return None


def evaluate_dtm_accuracy(
    dtm_path: str | Path,
    reference_path: Optional[str | Path] = None,
    flat_area_threshold_slope_deg: float = 1.0,
    max_check_points: int = 100_000,
    auto_download_srtm: bool = True,
) -> Dict:
    """
    Compute vertical accuracy metrics for a DTM raster.

    Parameters
    ----------
    dtm_path                      : path to the generated DTM COG
    reference_path                : optional external reference raster.
                                    Must overlap spatially; will be reprojected
                                    and resampled to DTM grid automatically.
    flat_area_threshold_slope_deg : for internal flat-plane check when no
                                    reference is given (degrees).
    max_check_points              : maximum pixel samples used.

    Returns
    -------
    dict  with rmse, mae, mean_error, std_error, nmad, le90,
              n_check_points, reference_type
    """
    dtm_path = Path(dtm_path)
    logger.info(f"Evaluating DTM accuracy: {dtm_path.name}")

    with rasterio.open(dtm_path) as src:
        dtm      = src.read(1).astype(np.float32)
        trans    = src.transform
        nodata   = src.nodata if src.nodata is not None else -9999.0
        crs      = src.crs
        res      = float(src.res[0])

    valid_mask = (dtm != nodata) & np.isfinite(dtm)

    # ── Reference raster path ─────────────────────────────────────────
    if reference_path is not None:
        ref_path = Path(reference_path)
        logger.info(f"  Using reference raster: {ref_path.name}")
        ref_arr  = _load_and_align_reference(ref_path, dtm, trans, crs, nodata)
        ref_valid = (ref_arr != nodata) & np.isfinite(ref_arr)
        mask      = valid_mask & ref_valid
        residuals = (dtm - ref_arr)[mask]
        ref_type  = "external"

    elif auto_download_srtm:
        # ── Try SRTM auto-download ──────────────────────────────────
        srtm_path = download_srtm_for_extent(dtm_path)
        if srtm_path and srtm_path.exists():
            logger.info(f"  Auto-downloaded SRTM reference: {srtm_path.name}")
            ref_arr  = _load_and_align_reference(srtm_path, dtm, trans, crs, nodata)
            ref_valid = (ref_arr != nodata) & np.isfinite(ref_arr)
            mask      = valid_mask & ref_valid
            residuals = (dtm - ref_arr)[mask]
            ref_type  = "srtm_30m"
        else:
            # ── Internal flat-area consistency check ────────────────
            logger.warning(
                "SRTM download failed – performing internal flat-area check."
            )
            residuals, ref_type = _internal_flat_check(dtm, valid_mask, res, flat_area_threshold_slope_deg)

    else:
        # ── Internal flat-area consistency check ─────────────────────
        logger.warning(
            "No external reference provided – performing internal flat-area check. "
            "LE90 and RMSE are against a local plane, not absolute truth."
        )
        residuals, ref_type = _internal_flat_check(dtm, valid_mask, res, flat_area_threshold_slope_deg)

    # ── Subsample if needed ──────────────────────────────────────────
    if len(residuals) > max_check_points:
        idx       = np.random.choice(len(residuals), max_check_points, replace=False)
        residuals = residuals[idx]

    n = len(residuals)
    if n == 0:
        logger.error("No valid check points found – unable to compute DTM metrics.")
        return {"error": "no valid check points", "n_check_points": 0}

    rmse       = float(np.sqrt(np.mean(residuals**2)))
    mae        = float(np.mean(np.abs(residuals)))
    mean_err   = float(np.mean(residuals))
    std_err    = float(np.std(residuals))
    nmad       = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    le90       = float(np.percentile(np.abs(residuals), 90))

    metrics = {
        "rmse_m":          round(rmse,     4),
        "mae_m":           round(mae,      4),
        "mean_error_m":    round(mean_err, 4),
        "std_error_m":     round(std_err,  4),
        "nmad_m":          round(nmad,     4),
        "le90_m":          round(le90,     4),
        "n_check_points":  n,
        "reference_type":  ref_type,
        "dtm_resolution_m": round(res, 3),
    }

    logger.success(
        f"DTM accuracy  RMSE={rmse:.4f}m  MAE={mae:.4f}m  "
        f"ME={mean_err:.4f}m  LE90={le90:.4f}m"
    )
    return metrics


def _internal_flat_check(
    dtm: np.ndarray, valid_mask: np.ndarray, res: float, flat_threshold_deg: float = 1.0
):
    """Compute internal flat-plane check residuals."""
    from scipy.ndimage import uniform_filter

    local_mean = uniform_filter(np.where(valid_mask, dtm, np.nan), size=3)
    slope_proxy = np.abs(dtm - local_mean)
    flat_m = flat_threshold_deg * np.pi / 180 * res

    flat_mask = valid_mask & (slope_proxy < flat_m)
    if flat_mask.sum() < 100:
        flat_mask = valid_mask

    local_plane = uniform_filter(np.where(flat_mask, dtm, 0.0), size=15)
    residuals = (dtm - local_plane)[flat_mask]
    return residuals, "internal_flat_plane"


def _load_and_align_reference(
    ref_path: Path,
    dtm_arr: np.ndarray,
    dtm_transform,
    dtm_crs,
    nodata: float,
) -> np.ndarray:
    """Reproject and resample a reference raster to the DTM grid."""
    H, W = dtm_arr.shape
    ref_arr = np.full((H, W), nodata, dtype=np.float32)

    with rasterio.open(ref_path) as ref:
        reproject(
            source      = rasterio.band(ref, 1),
            destination = ref_arr,
            src_transform = ref.transform,
            src_crs       = ref.crs,
            dst_transform = dtm_transform,
            dst_crs       = dtm_crs,
            resampling    = Resampling.bilinear,
            src_nodata    = ref.nodata,
            dst_nodata    = nodata,
        )
    return ref_arr
