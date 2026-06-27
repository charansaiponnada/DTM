"""
app/geo_utils.py
────────────────
Geospatial helpers for the Streamlit dashboard:
  • Raster TIF → base64-encoded PNG for folium ImageOverlay
  • CRS conversion UTM → WGS84
  • GeoDataFrame loaders (drainage channels, waterlogging hotspots)
"""
from __future__ import annotations
import io, base64
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
from pyproj import Transformer


# ── Colormaps used for each raster layer ──────────────────────────────────────
CMAPS = {
    "dtm":             "terrain",
    "hillshade":       "gray",
    "slope":           "YlOrRd",
    "twi":             "Blues",
    "waterlogging":    "RdYlGn_r",   # red = high risk
    "flow_accumulation":"plasma",
}


def _read_raster_band(tif_path: Path, max_px: int = 512):
    """Read first band of a GeoTIFF, downsampled to ≤ max_px on the longest side."""
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        scale = min(max_px / max(h, w), 1.0)
        out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
        data = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype(np.float32)
        nodata = src.nodata
        bounds_wgs84 = raster_bounds_wgs84(src)
    data = np.where(data == nodata, np.nan, data) if nodata is not None else data
    return data, bounds_wgs84


def raster_bounds_wgs84(src) -> list[list[float]]:
    """Return [[lat_min, lon_min], [lat_max, lon_max]] from an open rasterio dataset."""
    left, bottom, right, top = transform_bounds(
        src.crs, "EPSG:4326", src.bounds.left, src.bounds.bottom,
        src.bounds.right, src.bounds.top
    )
    return [[bottom, left], [top, right]]


def raster_to_overlay(tif_path: Path, cmap_key: str = "dtm",
                       opacity: float = 0.75, max_px: int = 512,
                       vmin=None, vmax=None):
    """
    Convert a GeoTIFF to a base64 PNG suitable for folium.raster_layers.ImageOverlay.

    Returns (png_url, bounds_wgs84, center_latlon).
    """
    data, bounds = _read_raster_band(tif_path, max_px)

    finite = data[np.isfinite(data)]
    lo = float(np.percentile(finite, 2))  if vmin is None else vmin
    hi = float(np.percentile(finite, 98)) if vmax is None else vmax

    cmap_name = CMAPS.get(cmap_key, "viridis")
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=lo, vmax=hi, clip=True)

    rgba = cmap(norm(data))          # (H, W, 4) float
    rgba[..., 3] = np.where(np.isfinite(data), opacity, 0.0)  # transparent nodata

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    png_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    lat_c = (bounds[0][0] + bounds[1][0]) / 2
    lon_c = (bounds[0][1] + bounds[1][1]) / 2
    return png_url, bounds, [lat_c, lon_c]


def drop_nodata_channels(gdf_utm: gpd.GeoDataFrame, dtm_path: Path,
                         min_valid_frac: float = 0.85) -> gpd.GeoDataFrame:
    """Drop flow-routing artifacts that cross nodata terrain (straight diagonal
    streaks routed through the rectangular DEM padding). Samples points along
    each line in the raster's native CRS; keeps lines almost entirely on valid
    DTM. Input gdf must be in the raster CRS (UTM)."""
    with rasterio.open(dtm_path) as src:
        nd = src.nodata
        keep = np.zeros(len(gdf_utm), dtype=bool)
        for i, geom in enumerate(gdf_utm.geometry.values):
            pts  = [geom.interpolate(t, normalized=True).coords[0]
                    for t in np.linspace(0, 1, 7)]
            vals = np.array([v[0] for v in src.sample(pts)], dtype=float)
            ok   = np.isfinite(vals) & (vals != nd)
            keep[i] = ok.mean() >= min_valid_frac
    return gdf_utm[keep].copy()


def load_drainage_channels(gpkg_path: Path,
                           dtm_path: Path | None = None) -> gpd.GeoDataFrame:
    """Load drainage_channels layer, optionally drop nodata-routing artifacts,
    reproject to WGS84, simplify geometry."""
    gdf = gpd.read_file(gpkg_path, layer="drainage_channels")
    if dtm_path is not None:
        gdf = drop_nodata_channels(gdf, dtm_path)
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf["geometry"].simplify(0.000005, preserve_topology=True)
    # Format display columns
    gdf["cost_inr_k"]    = (gdf["cost_inr"] / 1000).round(1)
    gdf["velocity_ms"]   = gdf["velocity_ms"].round(3)
    gdf["length_m"]      = gdf["length_m"].round(1)
    gdf["slope_pct"]     = (gdf["slope_mm"] / 10).round(2)
    return gdf


def load_waterlogging_hotspots(gpkg_path: Path,
                                risk_filter: list[str] | None = None
                                ) -> gpd.GeoDataFrame:
    """Load waterlogging_hotspots layer, optionally filter by risk_level."""
    gdf = gpd.read_file(gpkg_path, layer="waterlogging_hotspots")
    if risk_filter:
        gdf = gdf[gdf["risk_level"].isin(risk_filter)]
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf["geometry"].simplify(0.000003, preserve_topology=True)
    gdf["area_ha"]  = (gdf["area_m2"] / 10000).round(3)
    gdf["prob_pct"] = (gdf["probability"] * 100).round(1)
    return gdf


def load_catchment_boundaries(gpkg_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path, layer="catchment_boundaries")
    return gdf.to_crs("EPSG:4326")


def high_risk_zones(wl_tif: Path, thresh: float = 0.65,
                    max_px: int = 900, min_area_m2: float = 80.0) -> gpd.GeoDataFrame:
    """Vectorise contiguous high-risk areas straight from the waterlogging raster
    (fast — array threshold + rasterio.features.shapes) instead of dissolving
    thousands of pixel hotspots. Returns merged zone polygons in WGS84."""
    from rasterio.features import shapes as rio_shapes
    from rasterio.enums import Resampling
    from shapely.geometry import shape as shp_shape

    with rasterio.open(wl_tif) as src:
        h, w = src.height, src.width
        scale = min(max_px / max(h, w), 1.0)
        oh, ow = max(1, int(h*scale)), max(1, int(w*scale))
        prob = src.read(1, out_shape=(oh, ow), resampling=Resampling.bilinear).astype(np.float32)
        transform = src.transform * src.transform.scale(w/ow, h/oh)
        nd = src.nodata
        crs = src.crs
    mask = (prob >= thresh) & np.isfinite(prob)
    if nd is not None:
        mask &= (prob != nd)
    if not mask.any():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    geoms = [shp_shape(g) for g, v in rio_shapes(mask.astype(np.uint8), mask=mask,
                                                 transform=transform) if v == 1]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf = gdf[gdf.geometry.area >= min_area_m2]           # drop pixel specks (m², UTM)
    gdf["geometry"] = gdf.geometry.buffer(0)              # fix any self-touch
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.simplify(0.00002, preserve_topology=True)
    return gdf.reset_index(drop=True)


def channel_color(channel_type: str) -> str:
    return "#7B4F28" if str(channel_type).lower() == "earthen" else "#4A4A8A"


RISK_COLORS = {"HIGH": "#D32F2F", "MEDIUM": "#FF8F00", "LOW": "#FBC02D"}
