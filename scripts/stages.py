from __future__ import annotations

import csv
import json
from pathlib import Path

import laspy
import numpy as np

from info import analyze_las, find_pointcloud_files


def _sanitize_name(file_path: Path) -> str:
    return file_path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def stage_preflight(input_dir: Path, output_dir: Path, expected_crs: str | None = None) -> list[dict]:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = find_pointcloud_files(input_dir)
    if not files:
        raise ValueError(f"No LAS/LAZ files found in {input_dir}")

    summaries = [analyze_las(path) for path in files]
    for item in summaries:
        item["crs_ok"] = True
        if expected_crs and item["crs"]:
            item["crs_ok"] = expected_crs in item["crs"]

    summary_path = reports_dir / "preflight_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def _load_arrays(file_path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    las = laspy.read(file_path)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    if "classification" in las.point_format.dimension_names:
        classification = np.asarray(las.classification, dtype=np.uint8)
        ground_mask = classification == 2
        if not np.any(ground_mask):
            threshold = np.quantile(z, 0.35)
            ground_mask = z <= threshold
    else:
        threshold = np.quantile(z, 0.35)
        ground_mask = z <= threshold

    total = x.shape[0]
    if max_points > 0 and total > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=max_points, replace=False))
        x, y, z, ground_mask = x[idx], y[idx], z[idx], ground_mask[idx]

    return x, y, z, ground_mask


def stage_prepare(
    input_dir: Path,
    output_dir: Path,
    max_points_per_file: int,
    seed: int,
) -> list[dict]:
    prepared_dir = output_dir / "interim" / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    files = find_pointcloud_files(input_dir)
    prepared = []

    for file_path in files:
        x, y, z, ground_mask = _load_arrays(file_path, max_points_per_file, seed)
        name = _sanitize_name(file_path)
        npz_path = prepared_dir / f"{name}_prepared.npz"
        np.savez_compressed(npz_path, x=x, y=y, z=z, ground=ground_mask)

        prepared.append(
            {
                "source": str(file_path),
                "name": name,
                "prepared_npz": str(npz_path),
                "point_count": int(x.shape[0]),
                "ground_count": int(np.count_nonzero(ground_mask)),
            }
        )

    return prepared


def _build_min_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    x0, y0 = float(np.min(x)), float(np.min(y))
    ix = np.floor((x - x0) / resolution).astype(np.int32)
    iy = np.floor((y - y0) / resolution).astype(np.int32)

    width = int(ix.max()) + 1
    height = int(iy.max()) + 1

    flat_index = iy * width + ix
    cell_count = width * height

    dtm_flat = np.full(cell_count, np.inf, dtype=np.float32)
    np.minimum.at(dtm_flat, flat_index, z.astype(np.float32))

    counts_flat = np.bincount(flat_index, minlength=cell_count).astype(np.int32)
    valid_flat = counts_flat > 0

    dtm_flat[~valid_flat] = np.nan
    dtm = dtm_flat.reshape((height, width))
    counts = counts_flat.reshape((height, width))
    valid = valid_flat.reshape((height, width))

    xs = x0 + (np.arange(width) + 0.5) * resolution
    ys = y0 + (np.arange(height) + 0.5) * resolution
    return dtm, counts, x0, y0, xs, ys


def stage_dtm(prepared: list[dict], output_dir: Path, resolution: float) -> list[dict]:
    dtm_dir = output_dir / "interim" / "dtm"
    dtm_dir.mkdir(parents=True, exist_ok=True)

    dtm_outputs = []
    for item in prepared:
        data = np.load(item["prepared_npz"])
        x = data["x"]
        y = data["y"]
        z = data["z"]
        ground = data["ground"].astype(bool)

        if np.count_nonzero(ground) < 10:
            continue

        dtm, counts, x0, y0, xs, ys = _build_min_grid(x[ground], y[ground], z[ground], resolution)
        out_path = dtm_dir / f"{item['name']}_dtm.npz"
        np.savez_compressed(
            out_path,
            dtm=dtm,
            counts=counts,
            x0=x0,
            y0=y0,
            resolution=resolution,
            xs=xs,
            ys=ys,
        )

        dtm_outputs.append({"name": item["name"], "dtm_npz": str(out_path)})

    return dtm_outputs


def stage_hydrology(dtm_outputs: list[dict], output_dir: Path) -> list[dict]:
    hydro_dir = output_dir / "interim" / "hydrology"
    hydro_dir.mkdir(parents=True, exist_ok=True)

    hydro_outputs = []
    for item in dtm_outputs:
        data = np.load(item["dtm_npz"])
        dtm = data["dtm"].astype(np.float32)

        valid = np.isfinite(dtm)
        if not np.any(valid):
            continue

        fill_value = np.nanmean(dtm[valid])
        filled = np.where(valid, dtm, fill_value)

        gy, gx = np.gradient(filled)
        slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)

        relative_depth = (np.nanmax(filled) - filled).astype(np.float32)
        flow_proxy = (relative_depth - np.nanmin(relative_depth)) / (
            np.nanmax(relative_depth) - np.nanmin(relative_depth) + 1e-6
        )
        flow_proxy = flow_proxy.astype(np.float32)

        out_path = hydro_dir / f"{item['name']}_hydro.npz"
        np.savez_compressed(out_path, slope=slope, relative_depth=relative_depth, flow_proxy=flow_proxy)
        hydro_outputs.append({"name": item["name"], "hydro_npz": str(out_path), "dtm_npz": item["dtm_npz"]})

    return hydro_outputs


def stage_features(hydro_outputs: list[dict], output_dir: Path) -> list[Path]:
    feature_dir = output_dir / "training_data"
    feature_dir.mkdir(parents=True, exist_ok=True)

    feature_files: list[Path] = []
    for item in hydro_outputs:
        dtm_data = np.load(item["dtm_npz"])
        hydro_data = np.load(item["hydro_npz"])

        dtm = dtm_data["dtm"].astype(np.float32)
        slope = hydro_data["slope"].astype(np.float32)
        rel = hydro_data["relative_depth"].astype(np.float32)
        flow = hydro_data["flow_proxy"].astype(np.float32)
        resolution = float(dtm_data["resolution"])
        x0 = float(dtm_data["x0"])
        y0 = float(dtm_data["y0"])

        rows, cols = np.where(np.isfinite(dtm))
        output_path = feature_dir / f"{item['name']}_features.csv"

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file",
                "x",
                "y",
                "elevation",
                "slope",
                "relative_depth",
                "flow_proxy",
                "label",
            ])
            for r, c in zip(rows, cols):
                x = x0 + (c + 0.5) * resolution
                y = y0 + (r + 0.5) * resolution
                writer.writerow(
                    [
                        item["name"],
                        f"{x:.3f}",
                        f"{y:.3f}",
                        f"{dtm[r, c]:.4f}",
                        f"{slope[r, c]:.4f}",
                        f"{rel[r, c]:.4f}",
                        f"{flow[r, c]:.4f}",
                        -1,
                    ]
                )

        feature_files.append(output_path)

    return feature_files