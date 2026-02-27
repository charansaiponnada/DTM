from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from env_policy import ensure_running_in_conda_env


def _read_features(features_dir: Path) -> dict[tuple[str, str, str], float]:
    elevation_index: dict[tuple[str, str, str], float] = {}
    for file_path in sorted(features_dir.glob("*_features.csv")):
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["file"], row["x"], row["y"])
                elevation_index[key] = float(row["elevation"])
    if not elevation_index:
        raise FileNotFoundError(f"No feature files found in {features_dir}")
    return elevation_index


def _read_predictions(predictions_csv: Path, elevation_index: dict[tuple[str, str, str], float]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with predictions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["file"], row["x"], row["y"])
            elevation = elevation_index.get(key)
            if elevation is None:
                continue

            grouped[row["file"]].append(
                {
                    "file": row["file"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "risk": float(row["risk_probability"]),
                    "pred_label": int(row["predicted_label"]),
                    "elevation": elevation,
                }
            )
    if not grouped:
        raise RuntimeError("No prediction rows matched feature coordinates.")
    return grouped


def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _to_feature_line(src: dict, dst: dict, line_id: int) -> dict:
    length = _dist(src, dst)
    drop = src["elevation"] - dst["elevation"]
    return {
        "type": "Feature",
        "properties": {
            "line_id": line_id,
            "village": src["file"],
            "risk_source": round(src["risk"], 6),
            "length_m": round(length, 3),
            "elev_drop": round(drop, 3),
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[src["x"], src["y"]], [dst["x"], dst["y"]]],
        },
    }


def _to_feature_point(row: dict, point_type: str) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "village": row["file"],
            "type": point_type,
            "risk": round(row["risk"], 6),
            "elevation": round(row["elevation"], 3),
        },
        "geometry": {"type": "Point", "coordinates": [row["x"], row["y"]]},
    }


def optimize(
    grouped_rows: dict[str, list[dict]],
    risk_threshold: float,
    outlet_quantile: float,
    max_lines_per_village: int,
    min_elev_drop: float,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    line_features: list[dict] = []
    hotspot_features: list[dict] = []
    outlet_features: list[dict] = []

    summary = {
        "villages": {},
        "totals": {
            "hotspots": 0,
            "outlets": 0,
            "proposed_lines": 0,
            "avg_length_m": 0.0,
            "avg_elev_drop": 0.0,
        },
    }

    lengths = []
    drops = []
    line_id = 1

    for village, rows in grouped_rows.items():
        elevations = np.array([r["elevation"] for r in rows], dtype=np.float32)
        outlet_cutoff = float(np.quantile(elevations, outlet_quantile))

        outlets = [r for r in rows if r["elevation"] <= outlet_cutoff]
        hotspots = [r for r in rows if r["risk"] >= risk_threshold]
        hotspots = sorted(hotspots, key=lambda r: r["risk"], reverse=True)[:max_lines_per_village]

        for o in outlets[: min(200, len(outlets))]:
            outlet_features.append(_to_feature_point(o, "outlet"))
        for h in hotspots:
            hotspot_features.append(_to_feature_point(h, "hotspot"))

        proposed = 0
        for hotspot in hotspots:
            valid_outlets = [o for o in outlets if hotspot["elevation"] - o["elevation"] >= min_elev_drop]
            if not valid_outlets:
                continue
            target = min(valid_outlets, key=lambda o: _dist(hotspot, o))
            feature = _to_feature_line(hotspot, target, line_id)
            line_id += 1
            line_features.append(feature)

            lengths.append(feature["properties"]["length_m"])
            drops.append(feature["properties"]["elev_drop"])
            proposed += 1

        summary["villages"][village] = {
            "hotspots_detected": len(hotspots),
            "outlets_detected": len(outlets),
            "proposed_lines": proposed,
            "risk_threshold": risk_threshold,
        }

    summary["totals"]["hotspots"] = len(hotspot_features)
    summary["totals"]["outlets"] = len(outlet_features)
    summary["totals"]["proposed_lines"] = len(line_features)
    summary["totals"]["avg_length_m"] = float(np.mean(lengths)) if lengths else 0.0
    summary["totals"]["avg_elev_drop"] = float(np.mean(drops)) if drops else 0.0

    return line_features, hotspot_features, outlet_features, summary


def _save_geojson(path: Path, features: list[dict]) -> None:
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ensure_running_in_conda_env()

    parser = argparse.ArgumentParser(description="Create optimized drainage proposal from ML prediction outputs")
    parser.add_argument("--predictions", default="outputs/ml/predictions.csv")
    parser.add_argument("--features-dir", default="outputs/training_data")
    parser.add_argument("--output-dir", default="outputs/optimization")
    parser.add_argument("--risk-threshold", type=float, default=0.75)
    parser.add_argument("--outlet-quantile", type=float, default=0.10)
    parser.add_argument("--max-lines-per-village", type=int, default=200)
    parser.add_argument("--min-elev-drop", type=float, default=0.3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elevation_index = _read_features(Path(args.features_dir))
    grouped_rows = _read_predictions(Path(args.predictions), elevation_index)

    line_features, hotspot_features, outlet_features, summary = optimize(
        grouped_rows=grouped_rows,
        risk_threshold=args.risk_threshold,
        outlet_quantile=args.outlet_quantile,
        max_lines_per_village=args.max_lines_per_village,
        min_elev_drop=args.min_elev_drop,
    )

    lines_path = output_dir / "proposed_drainage_lines.geojson"
    hotspots_path = output_dir / "hotspots.geojson"
    outlets_path = output_dir / "outlets.geojson"
    params_path = output_dir / "design_parameters.json"
    summary_path = output_dir / "optimization_summary.json"

    _save_geojson(lines_path, line_features)
    _save_geojson(hotspots_path, hotspot_features)
    _save_geojson(outlets_path, outlet_features)

    design_parameters = {
        "risk_threshold": args.risk_threshold,
        "outlet_quantile": args.outlet_quantile,
        "max_lines_per_village": args.max_lines_per_village,
        "min_elev_drop": args.min_elev_drop,
        "algorithm": "nearest-feasible-outlet",
    }
    params_path.write_text(json.dumps(design_parameters, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Drainage lines: {len(line_features)} -> {lines_path}")
    print(f"Hotspots: {len(hotspot_features)} -> {hotspots_path}")
    print(f"Outlets: {len(outlet_features)} -> {outlets_path}")
    print(f"Design parameters: {params_path}")
    print(f"Optimization summary: {summary_path}")


if __name__ == "__main__":
    main()