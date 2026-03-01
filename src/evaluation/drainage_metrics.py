"""
src/evaluation/drainage_metrics.py
────────────────────────────────────
Evaluate the designed drainage network quality.

Metrics reported
----------------
  channel_count            : total number of designed channel segments
  total_length_m           : cumulative channel length (metres)
  total_cost_inr           : estimated construction cost (INR)
  stream_coverage_ratio    : fraction of extracted stream length covered by designed channels
  hydraulic_adequacy_ratio : fraction of segments where designed capacity >= required discharge
  avg_velocity_ms          : mean flow velocity across all segments
  capacity_exceeded_count  : segments where velocity > 3 m/s (erosion risk)
  order_distribution       : channel count by Strahler stream order
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger

try:
    import geopandas as gpd
except ImportError:
    raise ImportError("geopandas is required: pip install geopandas")


def evaluate_drainage_design(
    drainage_gpkg_path: str | Path,
    streams_layer: str = "drainage_channels",
    channels_layer: str = "drainage_channels",
    design_discharge_col: str = "design_discharge_m3s",
    capacity_col: str = "capacity_m3s",
    velocity_col: str = "velocity_ms",
    max_velocity_ms: float = 3.0,
) -> Dict:
    """
    Evaluate drainage network design quality from a GeoPackage.

    Parameters
    ----------
    drainage_gpkg_path   : path to GeoPackage containing drainage layers
    streams_layer        : layer name for extracted stream network
    channels_layer       : layer name for designed channel segments
    design_discharge_col : attribute column for required discharge (m³/s)
    capacity_col         : attribute column for channel capacity (m³/s)
    velocity_col         : attribute column for flow velocity (m/s)
    max_velocity_ms      : maximum safe velocity (erosion threshold)

    Returns
    -------
    dict  with channel_count, total_length_m, total_cost_inr,
              stream_coverage_ratio, hydraulic_adequacy_ratio,
              avg_velocity_ms, capacity_exceeded_count, order_distribution
    """
    gpkg_path = Path(drainage_gpkg_path)
    logger.info(f"Evaluating drainage design: {gpkg_path.name}")

    if not gpkg_path.exists():
        return {"error": f"GPKG not found: {gpkg_path}", "channel_count": 0}

    # ── Load layers ───────────────────────────────────────────────────
    import fiona
    available_layers = fiona.listlayers(str(gpkg_path))

    channels_gdf = _safe_read_layer(gpkg_path, channels_layer, available_layers)
    streams_gdf  = _safe_read_layer(gpkg_path, streams_layer,  available_layers)

    if channels_gdf is None or len(channels_gdf) == 0:
        logger.warning("No channel segments found in GPKG.")
        return {"error": "no channel data", "channel_count": 0}

    # ── Basic channel statistics ───────────────────────────────────────
    channels_gdf = channels_gdf.to_crs(epsg=32643) if channels_gdf.crs else channels_gdf
    channel_lengths = channels_gdf.geometry.length.fillna(0.0)
    total_length_m  = float(channel_lengths.sum())
    channel_count   = len(channels_gdf)

    # ── Cost ──────────────────────────────────────────────────────────
    if "cost_inr" in channels_gdf.columns:
        total_cost_inr = float(channels_gdf["cost_inr"].sum())
    elif "total_cost" in channels_gdf.columns:
        total_cost_inr = float(channels_gdf["total_cost"].sum())
    else:
        # Estimate from length: ₹800/m earthen default
        total_cost_inr = total_length_m * 800.0

    # ── Stream coverage ratio ─────────────────────────────────────────
    if streams_gdf is not None and len(streams_gdf) > 0:
        streams_gdf    = streams_gdf.to_crs(epsg=32643) if streams_gdf.crs else streams_gdf
        total_stream_m = float(streams_gdf.geometry.length.sum())
        coverage_ratio = min(total_length_m / (total_stream_m + 1e-6), 1.0)
    else:
        total_stream_m = None
        coverage_ratio = None

    # ── Hydraulic adequacy ────────────────────────────────────────────
    if capacity_col in channels_gdf.columns and design_discharge_col in channels_gdf.columns:
        cap  = channels_gdf[capacity_col].fillna(0.0)
        req  = channels_gdf[design_discharge_col].fillna(0.0)
        adequate = (cap >= req).sum()
        hydraulic_adequacy_ratio = float(adequate / channel_count)
    else:
        hydraulic_adequacy_ratio = None

    # ── Velocity check ────────────────────────────────────────────────
    if velocity_col in channels_gdf.columns:
        vel = channels_gdf[velocity_col].fillna(0.0)
        avg_velocity_ms        = float(vel.mean())
        capacity_exceeded_count = int((vel > max_velocity_ms).sum())
    else:
        avg_velocity_ms        = None
        capacity_exceeded_count = None

    # ── Order distribution ────────────────────────────────────────────
    if "order" in channels_gdf.columns:
        order_counts = channels_gdf["order"].value_counts().sort_index().to_dict()
        order_distribution = {int(k): int(v) for k, v in order_counts.items()}
    else:
        order_distribution = {}

    metrics = {
        "channel_count":             channel_count,
        "total_length_m":            round(total_length_m,   1),
        "total_cost_inr":            round(total_cost_inr,   0),
        "total_cost_inr_lakhs":      round(total_cost_inr / 1e5, 2),
        "stream_coverage_ratio":     round(coverage_ratio, 4) if coverage_ratio is not None else None,
        "total_stream_length_m":     round(total_stream_m, 1) if total_stream_m is not None else None,
        "hydraulic_adequacy_ratio":  round(hydraulic_adequacy_ratio, 4) if hydraulic_adequacy_ratio is not None else None,
        "avg_velocity_ms":           round(avg_velocity_ms, 3) if avg_velocity_ms is not None else None,
        "capacity_exceeded_count":   capacity_exceeded_count,
        "max_safe_velocity_ms":      max_velocity_ms,
        "order_distribution":        order_distribution,
    }

    logger.success(
        f"Drainage design  channels={channel_count}  "
        f"length={total_length_m:.0f}m  "
        f"cost=₹{total_cost_inr/1e5:.1f}L  "
        + (f"coverage={coverage_ratio:.1%}" if coverage_ratio else "")
    )
    return metrics


def _safe_read_layer(
    gpkg_path: Path,
    layer_name: str,
    available_layers: list,
) -> Optional["gpd.GeoDataFrame"]:
    """Read a GPKG layer, returning None if not present."""
    if layer_name not in available_layers:
        # Try partial match
        matches = [l for l in available_layers if layer_name in l]
        if not matches:
            logger.warning(f"Layer '{layer_name}' not found in {gpkg_path.name}")
            return None
        layer_name = matches[0]

    try:
        return gpd.read_file(str(gpkg_path), layer=layer_name)
    except Exception as exc:
        logger.warning(f"Could not read layer '{layer_name}': {exc}")
        return None
