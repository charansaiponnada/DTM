"""
scripts/fix_drainage.py
───────────────────────
Recompute drainage hydraulic design with REAL inputs instead of proxies.

Original design() used catchment = length*50*order and slope = 0.005/order.
This script replaces those with physically-derived values sampled from rasters
the pipeline already produced:

  • Catchment area  = upslope contributing cells from flow_accumulation.tif
                      (stored as log1p → cells = exp(facc) - 1) × cell_area
  • Channel slope   = true longitudinal bed slope from the DTM endpoints (Δz/L)

then re-applies the same (correct) Rational-Method + Manning's sizing, rewrites
the GeoPackage drainage_channels layer, and updates the drainage block in
data/output/_reports/honest_metrics.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hydrology.drainage_network import rational_discharge, design_trapezoidal_channel, DrainageDesignParameters

OUT = Path("data/output")
PARAMS = DrainageDesignParameters()


def _sample(arr, transform, xy, nodata):
    """Sample raster value at a map coordinate; None if outside/nodata."""
    try:
        col, row = ~transform * (xy[0], xy[1])
        r, c = int(row), int(col)
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            v = arr[r, c]
            if np.isfinite(v) and (nodata is None or v != nodata):
                return float(v)
    except Exception:
        pass
    return None


def _lines(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    return []


def recompute_village(village: str) -> dict | None:
    d = OUT / village
    gpkg = d / "drainage_network.gpkg"
    facc_p, dtm_p = d / "flow_accumulation.tif", d / "dtm.tif"
    if not (gpkg.exists() and facc_p.exists() and dtm_p.exists()):
        print(f"  [skip] {village}: missing inputs"); return None

    with rasterio.open(facc_p) as s:
        facc = s.read(1).astype(float); facc_t = s.transform
        facc_nd = s.nodata; cell = float(s.res[0]); cell_area = cell * cell
    with rasterio.open(dtm_p) as s:
        dem = s.read(1).astype(float); dem_t = s.transform; dem_nd = s.nodata

    ch = gpd.read_file(gpkg, layer="drainage_channels")
    rows, costs, vels, caps_ok, types, Qs, catchments = [], [], [], 0, [], [], []

    for _, row in ch.iterrows():
        segs = _lines(row.geometry)
        if not segs:
            continue
        geom = max(segs, key=lambda g: g.length)
        coords = list(geom.coords)
        length = geom.length
        if length <= 0:
            continue

        # real catchment: max upslope cells along the channel
        facc_vals = [v for v in (_sample(facc, facc_t, xy, facc_nd) for xy in coords) if v is not None]
        if facc_vals:
            cells = np.expm1(max(facc_vals))            # log1p → cells
            catch_m2 = float(max(cells, 1.0) * cell_area)
        else:
            catch_m2 = length * 50.0                     # fallback

        # real bed slope from DTM endpoints
        z0 = _sample(dem, dem_t, coords[0], dem_nd)
        z1 = _sample(dem, dem_t, coords[-1], dem_nd)
        slope = (abs(z1 - z0) / length) if (z0 is not None and z1 is not None) else 0.001
        slope = float(min(max(slope, 0.0005), 0.10))

        Q = rational_discharge(catch_m2, PARAMS.rainfall_intensity_mmhr, PARAMS.runoff_coefficient)
        b, dep, top_w, V, Qcap, cost_pm, ch_type = design_trapezoidal_channel(Q, slope, PARAMS)
        cost = length * cost_pm

        costs.append(cost); vels.append(V); types.append(ch_type); Qs.append(Q); catchments.append(catch_m2)
        caps_ok += int(Qcap >= Q)
        rows.append({
            "segment_id": int(row.get("segment_id", len(rows))),
            "length_m": round(length, 2),
            "catchment_m2": round(catch_m2, 1),
            "slope_mm": round(slope * 1000, 3),
            "Q_design_m3s": round(Q, 4),
            "channel_type": ch_type,
            "bottom_width_m": round(b, 2),
            "depth_m": round(dep, 2),
            "top_width_m": round(top_w, 2),
            "velocity_ms": round(V, 3),
            "capacity_m3s": round(Qcap, 4),
            "cost_inr": round(cost, 0),
            "geometry": geom,
        })

    if not rows:
        print(f"  [skip] {village}: no channels"); return None

    total_len = sum(r["length_m"] for r in rows)
    total_cost = float(sum(costs))
    n = len(rows)
    summary = {
        "channel_count": n,
        "total_length_m": round(total_len, 1),
        "total_cost_inr": round(total_cost, 0),
        "total_cost_inr_lakhs": round(total_cost / 1e5, 2),
        "cost_per_km_inr": round(total_cost / (total_len / 1000), 0) if total_len else 0,
        "avg_velocity_ms": round(float(np.mean(vels)), 3),
        "max_velocity_ms": round(float(np.max(vels)), 3),
        "n_earthen": sum(1 for t in types if t == "earthen"),
        "n_concrete": sum(1 for t in types if t == "concrete"),
        "capacity_exceeded_count": int(n - caps_ok),
        "max_design_flow_m3s": round(float(np.max(Qs)), 4),
        "max_catchment_m2": round(float(np.max(catchments)), 0),
        "design_method": "rational_method_real_catchment + mannings_trapezoidal",
        "inputs": "catchment from flow_accumulation (log1p cells); slope from DTM longitudinal gradient",
    }

    # rewrite the gpkg layer with corrected per-segment attributes
    gdf = gpd.GeoDataFrame(rows, crs=ch.crs)
    gdf.to_file(str(gpkg), layer="drainage_channels", driver="GPKG")

    print(f"  {village:16s} {n} ch | {total_len/1000:.1f} km | Rs {summary['total_cost_inr_lakhs']:.0f}L "
          f"| concrete {summary['n_concrete']} | capfail {summary['capacity_exceeded_count']} "
          f"| avgV {summary['avg_velocity_ms']} | maxCatch {summary['max_catchment_m2']/1e4:.2f}ha")
    return summary


def main():
    villages = sys.argv[1:] or [v.name for v in OUT.iterdir()
                                if (v / "drainage_network.gpkg").exists() and not v.name.startswith("_")]
    rep_path = OUT / "_reports" / "honest_metrics.json"
    rep = json.loads(rep_path.read_text())
    print("Recomputing drainage with real catchment + slope ...")
    for v in sorted(villages):
        s = recompute_village(v)
        if s and v in rep["villages"]:
            rep["villages"][v]["drainage"] = s
    rep_path.write_text(json.dumps(rep, indent=2))
    print(f"[OK] updated {rep_path}")


if __name__ == "__main__":
    main()
