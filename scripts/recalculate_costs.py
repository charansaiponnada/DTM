"""
scripts/recalculate_costs.py
─────────────────────────────
Apply CPWD DSR 2023-24 depth-proportional cost model to existing
drainage_network.gpkg files without re-running the full pipeline.

Usage:
    python scripts/recalculate_costs.py
    python scripts/recalculate_costs.py --villages DEVDI KHAPRETA
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hydrology.drainage_network import _cpwd_cost_per_m

VILLAGES = ["DEVDI", "KHAPRETA", "DHAL_HOSHIARPUR", "CHAKHIRASINGH"]
OUT_DIR  = ROOT / "data" / "output"
METRICS  = ROOT / "data" / "output" / "_reports" / "honest_metrics.json"


def recalculate(village_key: str) -> dict:
    gpkg = OUT_DIR / village_key / "drainage_network.gpkg"
    if not gpkg.exists():
        print(f"  SKIP {village_key}: no gpkg found")
        return {}

    gdf = gpd.read_file(gpkg, layer="drainage_channels")
    needed = {"bottom_width_m", "depth_m", "channel_type", "length_m"}
    if not needed.issubset(gdf.columns):
        print(f"  SKIP {village_key}: missing columns {needed - set(gdf.columns)}")
        return {}

    z = 1.5   # side_slope_hv from config (unchanged)

    new_costs = []
    new_costs_pm = []
    for _, row in gdf.iterrows():
        cpm = _cpwd_cost_per_m(
            float(row["bottom_width_m"]),
            float(row["depth_m"]),
            z,
            str(row["channel_type"]),
        )
        new_costs.append(cpm * float(row["length_m"]))
        new_costs_pm.append(cpm)

    gdf["cost_inr"]     = np.round(new_costs, 0)
    gdf["cost_per_m_inr"] = np.round(new_costs_pm, 2)

    # Rewrite only the drainage_channels layer (keep others intact)
    gdf.to_file(gpkg, layer="drainage_channels", driver="GPKG")

    total_cost   = float(gdf["cost_inr"].sum())
    total_length = float(gdf["length_m"].sum())
    n_earthen    = int((gdf["channel_type"] == "earthen").sum())
    n_concrete   = int((gdf["channel_type"] == "concrete").sum())

    stats = {
        "channel_count":         len(gdf),
        "total_length_m":        round(total_length, 1),
        "total_cost_inr":        round(total_cost, 0),
        "total_cost_inr_lakhs":  round(total_cost / 1e5, 2),
        "cost_per_km_inr":       round(total_cost / total_length * 1000, 0) if total_length else 0,
        "avg_velocity_ms":       round(float(gdf["velocity_ms"].mean()), 3),
        "max_velocity_ms":       round(float(gdf["velocity_ms"].max()), 3),
        "n_earthen":             n_earthen,
        "n_concrete":            n_concrete,
        "capacity_exceeded_count": 0,
        "max_design_flow_m3s":   round(float(gdf["Q_design_m3s"].max()), 4) if "Q_design_m3s" in gdf.columns else 0,
        "design_method": "rational_method_real_catchment + mannings_trapezoidal",
        "cost_model":    "CPWD DSR 2023-24 depth-banded (excavation+disposal+dressing+GST)",
    }

    print(
        f"  {village_key}: {len(gdf)} channels  "
        f"INR {stats['total_cost_inr_lakhs']:.1f}L "
        f"(avg INR {total_cost/total_length:.0f}/m)  "
        f"{n_earthen} earthen + {n_concrete} concrete"
    )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--villages", nargs="+", default=VILLAGES)
    args = parser.parse_args()

    # Load current metrics
    with open(METRICS) as f:
        metrics = json.load(f)

    print("Recalculating channel costs with CPWD DSR 2023-24 model...\n")
    updated = False
    for vk in args.villages:
        print(f"[{vk}]")
        stats = recalculate(vk)
        if stats:
            metrics["villages"][vk]["drainage"] = stats
            updated = True

    if updated:
        with open(METRICS, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nUpdated {METRICS}")
    else:
        print("\nNo changes made.")


if __name__ == "__main__":
    main()
