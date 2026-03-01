"""
src/dtm/dtm_generator.py
────────────────────────
Converts classified ground points into a Digital Terrain Model (DTM)
raster using Inverse-Distance Weighting (IDW) or Kriging interpolation,
then exports as a Cloud-Optimized GeoTIFF (COG) per OGC standards.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from loguru import logger
from tqdm import tqdm


NODATA = -9999.0


# ══════════════════════════════════════════════════════════════════════════
#  Grid Builder
# ══════════════════════════════════════════════════════════════════════════

def build_grid(
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
    padding: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, rasterio.transform.Affine, int, int]:
    """
    Build a regular XY grid covering the point cloud extent.

    Returns
    -------
    gx        : (H, W) X-coordinates of grid cell centres
    gy        : (H, W) Y-coordinates of grid cell centres
    transform : Rasterio affine transform
    nrows     : grid height (rows)
    ncols     : grid width  (cols)
    """
    x_min, x_max = x.min() - padding, x.max() + padding
    y_min, y_max = y.min() - padding, y.max() + padding

    ncols = int(np.ceil((x_max - x_min) / resolution))
    nrows = int(np.ceil((y_max - y_min) / resolution))

    transform = from_bounds(x_min, y_min, x_max, y_max, ncols, nrows)

    # Cell centres
    cx = x_min + resolution * (np.arange(ncols) + 0.5)
    cy = y_max - resolution * (np.arange(nrows) + 0.5)   # north-up
    gx, gy = np.meshgrid(cx, cy)

    logger.info(f"Grid: {nrows} × {ncols} cells @ {resolution} m resolution")
    return gx, gy, transform, nrows, ncols


# ══════════════════════════════════════════════════════════════════════════
#  IDW Interpolation
# ══════════════════════════════════════════════════════════════════════════

def idw_interpolate(
    src_xyz: np.ndarray,         # (N, 3) ground points
    gx: np.ndarray,              # (H, W) grid X
    gy: np.ndarray,              # (H, W) grid Y
    power: float = 2.0,
    radius: float = 5.0,
    min_points: int = 3,
    batch_size: int = 100_000,
    k_neighbors: int = 16,       # max K nearest neighbors (for speed)
) -> np.ndarray:
    """
    Inverse-Distance Weighting interpolation using vectorised cKDTree queries.

    Uses K-nearest-neighbour lookup (k_neighbors) then filters by radius –
    fully vectorised per batch, no Python loops over individual cells.

    Parameters
    ----------
    src_xyz      : ground point coordinates (X, Y, Z)
    gx / gy      : query grid arrays
    power        : IDW exponent (2 is standard)
    radius       : search radius in metres (discard neighbours beyond this)
    min_points   : minimum neighbours required to interpolate a cell
    batch_size   : number of grid cells processed per batch (memory control)
    k_neighbors  : max neighbours to fetch per query (caps memory)

    Returns
    -------
    z_grid : (H, W) float32 elevation array; NODATA where no neighbours
    """
    H, W     = gx.shape
    z_grid   = np.full((H, W), NODATA, dtype=np.float32)
    src_xy   = src_xyz[:, :2]
    src_z    = src_xyz[:, 2]

    # Build KD-tree once (cKDTree on float64 for accuracy)
    tree = cKDTree(src_xy.astype(np.float64))
    query_pts = np.column_stack([gx.ravel(), gy.ravel()])  # (H*W, 2)

    # Cap k to available points
    k = min(k_neighbors, len(src_xy))
    n_batches = int(np.ceil(len(query_pts) / batch_size))
    logger.info(f"IDW interpolating {H}×{W} grid in {n_batches} batches (k={k}) …")

    for i in tqdm(range(n_batches), desc="IDW"):
        start = i * batch_size
        end   = min(start + batch_size, len(query_pts))
        batch = query_pts[start:end].astype(np.float64)   # (B, 2)

        # --- Vectorised K-NN query ------------------------------------------
        dists, idxs = tree.query(batch, k=k, workers=-1)   # (B, k) each
        # dists shape: (B,)  if k=1, else (B, k)
        if k == 1:
            dists = dists[:, np.newaxis]
            idxs  = idxs[:, np.newaxis]

        # --- Apply radius mask ----------------------------------------------
        in_radius = dists <= radius                         # (B, k) bool
        valid_count = in_radius.sum(axis=1)                 # (B,)
        valid_cells = valid_count >= min_points             # only cells with enough nbrs

        if not valid_cells.any():
            continue

        # --- IDW weights (vectorised) ---------------------------------------
        safe_dists = np.where(in_radius, np.maximum(dists, 1e-6), np.inf)
        weights    = np.where(in_radius, 1.0 / (safe_dists ** power), 0.0)  # (B, k)
        w_sum      = weights.sum(axis=1)                    # (B,)

        # Gather neighbour Z values: shape (B, k)
        z_nbr = src_z[idxs]                                 # (B, k) – valid for all k

        # Weighted sum
        z_interp = (weights * z_nbr).sum(axis=1) / np.maximum(w_sum, 1e-12)  # (B,)

        # --- Write back to grid ---------------------------------------------
        pidxs = start + np.where(valid_cells)[0]           # flat grid indices of valid cells
        rows, cols = np.unravel_index(pidxs, (H, W))
        z_grid[rows, cols] = z_interp[valid_cells].astype(np.float32)

    valid_pct = 100 * (z_grid != NODATA).sum() / z_grid.size
    logger.success(f"IDW complete: {valid_pct:.1f}% cells filled")
    return z_grid


# ══════════════════════════════════════════════════════════════════════════
#  COG Writer
# ══════════════════════════════════════════════════════════════════════════

def write_geotiff(
    array: np.ndarray,
    transform: rasterio.transform.Affine,
    output_path: str | Path,
    crs: str | CRS = "EPSG:32643",
    nodata: float = NODATA,
    band_name: str = "elevation_m",
) -> Path:
    """Write a 2-D array to a standard GeoTIFF (intermediate step)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(crs, str):
        crs = CRS.from_epsg(int(crs.split(":")[-1]))

    with rasterio.open(
        output_path,
        mode="w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(array, 1)
        dst.update_tags(1, BAND_NAME=band_name, UNITS="metres")

    logger.info(f"GeoTIFF written → {output_path}")
    return output_path


def convert_to_cog(
    tiff_path: str | Path,
    cog_path: Optional[str | Path] = None,
    overview_levels: Tuple[int, ...] = (2, 4, 8, 16),
    resampling: str = "average",
) -> Path:
    """
    Convert a standard GeoTIFF to a Cloud-Optimized GeoTIFF (COG).
    COG is the OGC-recommended raster format for this hackathon.

    Parameters
    ----------
    tiff_path       : input GeoTIFF
    cog_path        : output COG path (defaults to _cog suffix)
    overview_levels : pyramid overview levels for fast access
    resampling      : resampling algorithm for overview generation
    """
    tiff_path = Path(tiff_path)
    if cog_path is None:
        cog_path = tiff_path.parent / (tiff_path.stem + "_cog.tif")
    cog_path = Path(cog_path)

    output_profile = cog_profiles.get("deflate")
    output_profile.update(
        blockxsize=512,
        blockysize=512,
    )

    logger.info(f"Converting to COG …")
    cog_translate(
        str(tiff_path),          # source  (positional)
        str(cog_path),           # dst_path (positional)
        output_profile,          # dst_kwargs (positional)
        overview_level=4,        # 0-4 → overviews at 2^1..2^4 = [2,4,8,16]
        overview_resampling="average",
        in_memory=False,
        quiet=True,
    )
    tiff_path.unlink(missing_ok=True)  # clean up intermediate
    logger.success(f"COG written → {cog_path.name}")
    return cog_path


# ══════════════════════════════════════════════════════════════════════════
#  High-Level DTM Builder
# ══════════════════════════════════════════════════════════════════════════

def generate_dtm(
    classified_las_path: str | Path,
    output_path: str | Path,
    resolution: float = 0.5,
    idw_power: float = 2.0,
    idw_radius: float = 5.0,
    smooth_sigma: float = 1.0,
    crs: str = "EPSG:32643",
) -> Path:
    """
    End-to-end DTM generation from a classified LAS file.

    Steps:
      1. Load classified LAS
      2. Extract ground points (class 2)
      3. Build interpolation grid
      4. IDW interpolation
      5. Gaussian smoothing (removes interpolation noise)
      6. Export as COG

    Parameters
    ----------
    classified_las_path : LAS with classification field set
    output_path         : destination COG .tif path
    resolution          : DTM cell size in metres
    idw_power           : IDW exponent
    idw_radius          : IDW search radius in metres
    smooth_sigma        : Gaussian σ (cells) – 0 to disable
    crs                 : EPSG string

    Returns
    -------
    Path to the output COG file
    """
    las_path = Path(classified_las_path)
    logger.info(f"Generating DTM from {las_path.name} …")

    # ── 1. Load & filter ground points ──────────────────────────────────
    las = laspy.read(las_path)
    classification = np.array(las.classification)
    ground_mask    = classification == 2

    n_ground = ground_mask.sum()
    logger.info(f"Ground points: {n_ground:,} / {len(classification):,} total")

    if n_ground < 100:
        raise ValueError(
            f"Only {n_ground} ground points found – check classification. "
            "Ensure SMRF / CSF was run before DTM generation."
        )

    src_xyz = np.column_stack([
        np.array(las.x)[ground_mask],
        np.array(las.y)[ground_mask],
        np.array(las.z)[ground_mask],
    ]).astype(np.float64)

    # ── 2. Build grid ────────────────────────────────────────────────────
    gx, gy, transform, nrows, ncols = build_grid(
        src_xyz[:, 0], src_xyz[:, 1], resolution
    )

    # ── 3. IDW Interpolation ─────────────────────────────────────────────
    z_grid = idw_interpolate(src_xyz, gx, gy, power=idw_power, radius=idw_radius)

    # ── 4. Smooth ────────────────────────────────────────────────────────
    if smooth_sigma and smooth_sigma > 0:
        valid_mask   = z_grid != NODATA
        filled       = z_grid.copy()
        filled[~valid_mask] = 0.0
        smoothed     = gaussian_filter(filled, sigma=smooth_sigma)
        weight_map   = gaussian_filter(valid_mask.astype(np.float32), sigma=smooth_sigma)
        weight_map   = np.where(weight_map > 0, weight_map, 1.0)
        z_grid       = np.where(valid_mask, smoothed / weight_map, NODATA).astype(np.float32)

    # ── 5. Write COG ─────────────────────────────────────────────────────
    tmp_tif = Path(output_path).parent / "_dtm_tmp.tif"
    write_geotiff(z_grid, transform, tmp_tif, crs=crs, band_name="DTM_elevation_m")
    cog_path = convert_to_cog(tmp_tif, cog_path=output_path)

    logger.success(f"DTM generation complete → {cog_path}")
    return cog_path


def get_dtm_stats(dtm_path: str | Path) -> dict:
    """Return basic DTM statistics for reporting."""
    with rasterio.open(dtm_path) as src:
        data = src.read(1, masked=True)
        return {
            "min_elevation_m":  float(data.min()),
            "max_elevation_m":  float(data.max()),
            "mean_elevation_m": float(data.mean()),
            "std_elevation_m":  float(data.std()),
            "relief_m":         float(data.max() - data.min()),
            "nodata_pct":       float(data.mask.mean() * 100),
            "resolution_m":     src.res[0],
            "crs":              str(src.crs),
            "shape":            src.shape,
        }
