from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np

from env_policy import ensure_running_in_workspace_venv


def analyze_las(file_path: Path) -> dict:
    with laspy.open(file_path) as reader:
        header = reader.header
        dimensions = sorted(list(header.point_format.dimension_names))
        crs_obj = header.parse_crs()
        crs = str(crs_obj) if crs_obj is not None else None

        point_count = int(header.point_count)
        min_x, min_y, min_z = [float(v) for v in header.mins]
        max_x, max_y, max_z = [float(v) for v in header.maxs]

        sample_size = min(point_count, 200_000)
        sample = reader.read_points(sample_size) if sample_size > 0 else None

    area = (max_x - min_x) * (max_y - min_y)
    density = float(point_count / area) if area > 0 else None

    classes = []
    has_ground_class = False
    if sample is not None and "classification" in dimensions and hasattr(sample, "classification"):
        classes = [int(item) for item in np.unique(sample.classification)]
        has_ground_class = 2 in classes

    intensity = None
    if sample is not None and "intensity" in dimensions and hasattr(sample, "intensity"):
        min_int = int(sample.intensity.min())
        max_int = int(sample.intensity.max())
        intensity = {
            "min": min_int,
            "max": max_int,
            "all_zero": max_int == 0,
        }

    return {
        "file": str(file_path),
        "dimensions": dimensions,
        "crs": crs,
        "point_count": point_count,
        "bounds": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z,
        },
        "area": area,
        "density": density,
        "classes": classes,
        "has_ground_class": has_ground_class,
        "intensity": intensity,
    }


def find_pointcloud_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in {".las", ".laz"}:
        return [path]
    if not path.exists():
        return []

    files = list(path.glob("*.las")) + list(path.glob("*.laz"))
    return sorted(files)


def print_summary(summary: dict) -> None:
    print("=" * 70)
    print(f"File: {summary['file']}")
    print(f"CRS: {summary['crs']}")
    print(f"Points: {summary['point_count']}")

    bounds = summary["bounds"]
    print(
        "Bounds X/Y/Z: "
        f"[{bounds['min_x']:.2f}, {bounds['max_x']:.2f}] / "
        f"[{bounds['min_y']:.2f}, {bounds['max_y']:.2f}] / "
        f"[{bounds['min_z']:.2f}, {bounds['max_z']:.2f}]"
    )

    density = summary["density"]
    density_text = f"{density:.2f} pts/sq.unit" if density is not None else "N/A"
    print(f"Density: {density_text}")

    if summary["classes"]:
        print(f"Classes: {summary['classes']}")
        print(f"Ground class present: {summary['has_ground_class']}")
    else:
        print("Classes: not available")

    intensity = summary["intensity"]
    if intensity is None:
        print("Intensity: not available")
    else:
        print(
            f"Intensity range: {intensity['min']} to {intensity['max']} "
            f"(all zero: {intensity['all_zero']})"
        )


def main() -> None:
    ensure_running_in_workspace_venv()

    parser = argparse.ArgumentParser(
        description="Inspect LAS/LAZ files and print data-readiness summary."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="Gujrat_Point_Cloud",
        help="Path to a LAS/LAZ file or a folder containing LAS/LAZ files.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional output JSON file for all summaries.",
    )
    args = parser.parse_args()

    target = Path(args.path)
    files = find_pointcloud_files(target)
    if not files:
        raise SystemExit(f"No LAS/LAZ files found at: {target}")

    summaries = [analyze_las(file_path) for file_path in files]
    for item in summaries:
        print_summary(item)

    print("=" * 70)
    print(f"Analyzed files: {len(summaries)}")

    if args.json_path:
        output_path = Path(args.json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"Saved JSON summary: {output_path}")


if __name__ == "__main__":
    main()