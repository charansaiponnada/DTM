from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from stages import stage_dtm, stage_features, stage_hydrology, stage_preflight, stage_prepare

STAGES = ["preflight", "prepare", "dtm", "hydrology", "features"]


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _selected_stages(from_stage: str, to_stage: str) -> list[str]:
    start_idx = STAGES.index(from_stage)
    end_idx = STAGES.index(to_stage)
    if start_idx > end_idx:
        raise ValueError("--from-stage must be before or equal to --to-stage")
    return STAGES[start_idx : end_idx + 1]


def _load_existing_prepared(output_dir: Path) -> list[dict]:
    prepared_dir = output_dir / "interim" / "prepared"
    items = []
    if not prepared_dir.exists():
        return items
    for path in sorted(prepared_dir.glob("*_prepared.npz")):
        name = path.name.replace("_prepared.npz", "")
        items.append({"name": name, "prepared_npz": str(path)})
    return items


def _load_existing_dtm(output_dir: Path) -> list[dict]:
    dtm_dir = output_dir / "interim" / "dtm"
    items = []
    if not dtm_dir.exists():
        return items
    for path in sorted(dtm_dir.glob("*_dtm.npz")):
        name = path.name.replace("_dtm.npz", "")
        items.append({"name": name, "dtm_npz": str(path)})
    return items


def _load_existing_hydrology(output_dir: Path) -> list[dict]:
    hydro_dir = output_dir / "interim" / "hydrology"
    dtm_dir = output_dir / "interim" / "dtm"
    items = []
    if not hydro_dir.exists():
        return items
    for path in sorted(hydro_dir.glob("*_hydro.npz")):
        name = path.name.replace("_hydro.npz", "")
        dtm_path = dtm_dir / f"{name}_dtm.npz"
        if dtm_path.exists():
            items.append({"name": name, "hydro_npz": str(path), "dtm_npz": str(dtm_path)})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean pipeline runner for point cloud preprocessing")
    parser.add_argument("--config", default="pipeline_config.json", help="Path to config JSON")
    parser.add_argument("--input", default=None, help="Optional override for input directory")
    parser.add_argument("--output", default=None, help="Optional override for output directory")
    parser.add_argument("--max-points", type=int, default=None, help="Optional override for max points per file")
    parser.add_argument("--resolution", type=float, default=None, help="Optional override for grid resolution")
    parser.add_argument("--from-stage", choices=STAGES, default="preflight")
    parser.add_argument("--to-stage", choices=STAGES, default="features")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    input_dir = Path(args.input or config.get("input_dir", "Gujrat_Point_Cloud"))
    output_dir = Path(args.output or config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _selected_stages(args.from_stage, args.to_stage)
    print(f"Running stages: {selected}")

    preflight = []
    prepared = []
    dtm_outputs = []
    hydro_outputs = []
    feature_files = []

    if "preflight" in selected:
        print("[stage] preflight")
        preflight = stage_preflight(input_dir, output_dir, expected_crs=config.get("expected_crs"))
        print(f"  files checked: {len(preflight)}")

    if "prepare" in selected:
        print("[stage] prepare")
        prepared = stage_prepare(
            input_dir=input_dir,
            output_dir=output_dir,
            max_points_per_file=int(args.max_points or config.get("max_points_per_file", 500_000)),
            seed=int(config.get("seed", 42)),
        )
        print(f"  files prepared: {len(prepared)}")

    if "dtm" in selected:
        print("[stage] dtm")
        if not prepared:
            prepared = _load_existing_prepared(output_dir)
        if not prepared:
            raise RuntimeError("DTM stage requires prepared artifacts in outputs/interim/prepared.")
        dtm_outputs = stage_dtm(
            prepared=prepared,
            output_dir=output_dir,
            resolution=float(args.resolution or config.get("grid_resolution", 1.0)),
        )
        print(f"  dtm outputs: {len(dtm_outputs)}")

    if "hydrology" in selected:
        print("[stage] hydrology")
        if not dtm_outputs:
            dtm_outputs = _load_existing_dtm(output_dir)
        if not dtm_outputs:
            raise RuntimeError("Hydrology stage requires DTM artifacts in outputs/interim/dtm.")
        hydro_outputs = stage_hydrology(dtm_outputs=dtm_outputs, output_dir=output_dir)
        print(f"  hydrology outputs: {len(hydro_outputs)}")

    if "features" in selected:
        print("[stage] features")
        if not hydro_outputs:
            hydro_outputs = _load_existing_hydrology(output_dir)
        if not hydro_outputs:
            raise RuntimeError("Features stage requires hydrology artifacts in outputs/interim/hydrology.")
        feature_files = stage_features(hydro_outputs=hydro_outputs, output_dir=output_dir)
        print(f"  feature files: {len(feature_files)}")

    summary_path = output_dir / "run_summary.txt"
    lines = [
        f"run_time={datetime.now().isoformat()}",
        f"input_dir={input_dir}",
        f"output_dir={output_dir}",
        f"stages={','.join(selected)}",
        f"preflight_files={len(preflight)}",
        f"prepared_files={len(prepared)}",
        f"dtm_files={len(dtm_outputs)}",
        f"hydrology_files={len(hydro_outputs)}",
        f"feature_files={len(feature_files)}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()