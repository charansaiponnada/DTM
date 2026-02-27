from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

import laspy
import numpy as np
from sklearn.neighbors import NearestNeighbors

from info import analyze_las, find_pointcloud_files


def _sanitize_name(file_path: Path) -> str:
    return file_path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def _append_note(current: str | None, note: str) -> str:
    if not current:
        return note
    return f"{current};{note}"


def _sample_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = x.shape[0]
    if max_points > 0 and total > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=max_points, replace=False))
        x, y, z, ground_mask = x[idx], y[idx], z[idx], ground_mask[idx]
    return x, y, z, ground_mask


def _apply_noise_filter_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sor_k: int,
    sor_std_mult: float,
    ror_radius: float,
    ror_min_neighbors: int,
) -> np.ndarray:
    if x.shape[0] < max(sor_k + 1, 10):
        return np.ones(x.shape[0], dtype=bool)

    coords = np.column_stack((x, y))
    model = NearestNeighbors(n_neighbors=sor_k + 1, algorithm="auto")
    model.fit(coords)

    distances, _ = model.kneighbors(coords)
    mean_neighbor_dist = distances[:, 1:].mean(axis=1)

    dist_threshold = mean_neighbor_dist.mean() + sor_std_mult * mean_neighbor_dist.std()
    sor_mask = mean_neighbor_dist <= dist_threshold

    radius_neighbors = model.radius_neighbors(coords, radius=ror_radius, return_distance=False)
    counts = np.array([len(indices) - 1 for indices in radius_neighbors], dtype=np.int32)
    ror_mask = counts >= ror_min_neighbors

    combined = sor_mask & ror_mask

    min_keep = max(500, int(0.10 * x.shape[0]))
    if int(np.count_nonzero(combined)) >= min_keep:
        return combined
    if int(np.count_nonzero(sor_mask)) >= min_keep:
        return sor_mask
    return np.ones(x.shape[0], dtype=bool)


def _classify_with_pdal(input_file: Path, classified_file: Path, smrf: dict) -> tuple[bool, str | None]:
    pipeline_definition = {
        "pipeline": [
            str(input_file),
            {
                "type": "filters.smrf",
                "ignore": "Classification[7:7]",
                "window": float(smrf.get("window", 16.0)),
                "slope": float(smrf.get("slope", 0.2)),
                "threshold": float(smrf.get("threshold", 0.45)),
                "scalar": float(smrf.get("scalar", 1.2)),
            },
            {
                "type": "writers.las",
                "filename": str(classified_file),
                "minor_version": 4,
                "dataformat_id": 3,
            },
        ]
    }

    try:
        import pdal  # type: ignore
    except Exception:
        pdal = None

    classified_file.parent.mkdir(parents=True, exist_ok=True)

    if pdal is not None:
        try:
            import json as _json

            pipeline = pdal.Pipeline(_json.dumps(pipeline_definition))
            pipeline.execute()
            return classified_file.exists(), None
        except Exception as exc:
            return False, f"pdal-py-exec-failed: {exc}"

    candidates = [
        Path(r"C:\Program Files\QGIS 3.40.15\bin\pdal.exe"),
        Path(r"C:\OSGeo4W\bin\pdal.exe"),
        Path("pdal.exe"),
    ]
    pdal_cli = next((path for path in candidates if path.exists() or str(path).lower() == "pdal.exe"), None)
    if pdal_cli is None:
        return False, "pdal-not-installed"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(pipeline_definition, tmp, indent=2)
        tmp_path = Path(tmp.name)

    try:
        command = [str(pdal_cli), "pipeline", str(tmp_path)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            return False, f"pdal-cli-exec-failed: {stderr}"
        return classified_file.exists(), None
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _classify_ground_mask_sampled_with_pdal(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    output_dir: Path,
    name: str,
    smrf_params: dict,
) -> tuple[np.ndarray | None, str | None, str | None]:
    classified_dir = output_dir / "interim" / "classified"
    classified_dir.mkdir(parents=True, exist_ok=True)

    sampled_las_path = classified_dir / f"{name}_sampled_for_pdal.las"
    classified_las_path = classified_dir / f"{name}_classified.las"

    las = laspy.create(file_version="1.4", point_format=3)
    las.x = x
    las.y = y
    las.z = z
    las.classification = np.zeros(x.shape[0], dtype=np.uint8)
    las.write(sampled_las_path)

    ok, err = _classify_with_pdal(sampled_las_path, classified_las_path, smrf_params)
    if not ok:
        return None, str(classified_las_path), err

    classified = laspy.read(classified_las_path)
    if "classification" not in classified.point_format.dimension_names:
        return None, str(classified_las_path), "pdal-output-missing-classification"

    mask = np.asarray(classified.classification, dtype=np.uint8) == 2
    if not np.any(mask):
        return None, str(classified_las_path), "pdal-output-no-ground-class"

    return mask, str(classified_las_path), None


def _load_arrays_with_method(
    file_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata: dict = {
        "classification_requested": "heuristic",
        "classification_used": "heuristic",
        "classified_file": None,
        "classification_note": None,
    }

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
            metadata["classification_note"] = _append_note(
                metadata["classification_note"], "class-2-missing-used-quantile-fallback"
            )
    else:
        threshold = np.quantile(z, 0.35)
        ground_mask = z <= threshold
        metadata["classification_note"] = _append_note(
            metadata["classification_note"], "classification-dimension-missing-used-quantile-fallback"
        )

    return x, y, z, ground_mask, metadata


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


def stage_prepare(
    input_dir: Path,
    output_dir: Path,
    max_points_per_file: int,
    seed: int,
    classification_method: str,
    sor_k: int,
    sor_std_mult: float,
    ror_radius: float,
    ror_min_neighbors: int,
    smrf_params: dict,
) -> list[dict]:
    prepared_dir = output_dir / "interim" / "prepared"
    reports_dir = output_dir / "reports"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = find_pointcloud_files(input_dir)
    prepared = []

    for file_path in files:
        x, y, z, ground_mask, metadata = _load_arrays_with_method(file_path=file_path)
        metadata["classification_requested"] = classification_method

        x, y, z, ground_mask = _sample_arrays(x, y, z, ground_mask, max_points_per_file, seed)

        noise_mask = _apply_noise_filter_mask(
            x=x,
            y=y,
            z=z,
            sor_k=sor_k,
            sor_std_mult=sor_std_mult,
            ror_radius=ror_radius,
            ror_min_neighbors=ror_min_neighbors,
        )

        x = x[noise_mask]
        y = y[noise_mask]
        z = z[noise_mask]
        ground_mask = ground_mask[noise_mask]

        name = _sanitize_name(file_path)

        if classification_method == "pdal" and x.shape[0] > 0:
            pdal_ground_mask, classified_file, pdal_error = _classify_ground_mask_sampled_with_pdal(
                x=x,
                y=y,
                z=z,
                output_dir=output_dir,
                name=name,
                smrf_params=smrf_params,
            )
            if pdal_ground_mask is not None:
                ground_mask = pdal_ground_mask
                metadata["classification_used"] = "pdal_smrf"
                metadata["classified_file"] = classified_file
                metadata["classification_note"] = _append_note(metadata["classification_note"], "pdal-on-sampled-points")
            else:
                metadata["classification_note"] = _append_note(metadata["classification_note"], pdal_error or "pdal-fallback")

        npz_path = prepared_dir / f"{name}_prepared.npz"
        np.savez_compressed(npz_path, x=x, y=y, z=z, ground=ground_mask)

        row = {
            "source": str(file_path),
            "name": name,
            "prepared_npz": str(npz_path),
            "point_count": int(x.shape[0]),
            "ground_count": int(np.count_nonzero(ground_mask)),
            "noise_filter_kept": int(np.count_nonzero(noise_mask)),
            "noise_filter_removed": int(noise_mask.shape[0] - np.count_nonzero(noise_mask)),
            "classification_requested": metadata["classification_requested"],
            "classification_used": metadata["classification_used"],
            "classification_note": metadata["classification_note"],
            "classified_file": metadata["classified_file"],
        }
        prepared.append(row)

    prepare_summary_path = reports_dir / "prepare_summary.json"
    prepare_summary_path.write_text(json.dumps(prepared, indent=2), encoding="utf-8")

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