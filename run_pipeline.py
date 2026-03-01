"""
run_pipeline.py
────────────────
Main entry point for the DTM Drainage AI pipeline.
Run this file directly — it handles logging setup, argument parsing,
error recovery, and the final summary report.

Usage (after activating dtm-env):
    python run_pipeline.py --input data\input\DEVDI_511671.las
    python run_pipeline.py --input data\input\DEVDI_511671.las --output data\output\devdi
    python run_pipeline.py --input data\input\DEVDI_511671.las --no-ml --stream-threshold 500
    python run_pipeline.py --batch                           # process all villages in config
    python run_pipeline.py --input data\input\DEVDI_511671.las --evaluate
    python run_pipeline.py --help
"""

import sys
import click
from pathlib import Path

# ── Logging must be set up before anything else is imported ──────────────
from src.logger import setup_logging, StageLogger, print_summary, log_exception
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════
#  CLI Definition
# ══════════════════════════════════════════════════════════════════════════

@click.command()
@click.option(
    "--input", "-i",
    default=None,
    type=click.Path(),
    help="Path to input LAS or LAZ file (omit with --batch)",
)
@click.option(
    "--batch",
    is_flag=True,
    default=False,
    help="Process all villages defined in config.data.villages",
)
@click.option(
    "--output", "-o",
    default="data/output",
    show_default=True,
    help="Output directory for all results",
)
@click.option(
    "--config", "-c",
    default="config/config.yaml",
    show_default=True,
    help="Path to config YAML file",
)
@click.option(
    "--log-dir",
    default="logs",
    show_default=True,
    help="Directory to write log files",
)
@click.option(
    "--log-level",
    default="DEBUG",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="File log verbosity level",
)
@click.option(
    "--no-ml",
    is_flag=True,
    default=False,
    help="Skip ML refinement of ground classification (faster, less accurate)",
)
@click.option(
    "--no-pointnet",
    is_flag=True,
    default=False,
    help="Skip PointNet deep learning (use only SMRF + Random Forest)",
)
@click.option(
    "--stream-threshold",
    default=1000,
    show_default=True,
    type=int,
    help="Min flow accumulation (cells) to define a stream channel",
)
@click.option(
    "--resolution",
    default=0.5,
    show_default=True,
    type=float,
    help="DTM resolution in metres per pixel",
)
@click.option(
    "--stages",
    default="1,2,3,4,5,6",
    show_default=True,
    help="Comma-separated stages to run (e.g. '3,4,5' to skip classification)",
)
@click.option(
    "--evaluate",
    is_flag=True,
    default=False,
    help="Run accuracy evaluation metrics after pipeline completes",
)
def main(
    input, output, config, log_dir, log_level,
    no_ml, no_pointnet, stream_threshold, resolution, stages, batch, evaluate
):
    """
    DTM Drainage AI Pipeline — MoPR Geospatial Intelligence Hackathon

    \b
    Stages:
      1. Data Inspection
      2. Ground Classification (SMRF + ML)
      3. DTM Generation (IDW → COG)
      4. Hydrological Analysis (Flow → TWI)
      5. Waterlogging Prediction (XGBoost)
      6. Drainage Network Design (MST + Manning)

    \b
    Output Formats (OGC-compliant):
      Rasters  → Cloud-Optimized GeoTIFF (.tif)
      Vectors  → GeoPackage (.gpkg)
      LiDAR    → LAS 1.4 (.las)
    """

    # ── Setup logging ────────────────────────────────────────────────────
    run_id = setup_logging(log_dir=log_dir, level=log_level)
    logger.info(f"Input  : {input}")
    logger.info(f"Output : {output}")
    logger.info(f"Config : {config}")
    logger.info(f"Batch  : {batch}")
    logger.info(f"Options: no_ml={no_ml}, stream_threshold={stream_threshold}, resolution={resolution}")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Lazy imports (only after logging is set up) ──────────────────────
    from pipelines.full_pipeline import DTMDrainagePipeline, BatchPipelineRunner

    # ── Batch mode ────────────────────────────────────────────────────────
    if batch:
        logger.info("Batch mode: processing all villages from config")
        runner = BatchPipelineRunner(
            config_path      = config,
            base_output_dir  = output,
            use_ml_refine    = not no_ml,
            stream_threshold = stream_threshold,
        )
        runner.run_all()
        print_summary(output_dir=output_dir, save_json=True)
        return

    if input is None:
        raise click.UsageError("--input is required when not using --batch mode.")

    if not Path(input).exists():
        raise click.UsageError(f"Input file not found: {input}")

    # Parse which stages to run
    stages_to_run = {int(s.strip()) for s in stages.split(",")}
    logger.info(f"Running stages: {sorted(stages_to_run)}")

    pipeline = DTMDrainagePipeline(
        config_path = config,
        input_las   = input,
        output_dir  = output,
    )

    # ── Override config with CLI flags ───────────────────────────────────
    if resolution != 0.5:
        pipeline.cfg["dtm"]["resolution"] = resolution

    # ════════════════════════════════════════════════════════════════════
    #  Run each stage with StageLogger
    # ════════════════════════════════════════════════════════════════════

    total = len(stages_to_run)
    step  = 0

    # ── Stage 1: Inspect ─────────────────────────────────────────────────
    if 1 in stages_to_run:
        step += 1
        with StageLogger("Data Inspection", stage_num=1, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage1_inspect()
            if pipeline.metadata:
                sl.set_result({
                    "points":   f"{pipeline.metadata.point_count:,}",
                    "density":  f"{pipeline.metadata.density_pts_sqm:.1f} pts/m²",
                    "crs":      pipeline.metadata.crs_wkt or "MISSING",
                    "intensity": f"{pipeline.metadata.intensity_range[0]:.0f}–{pipeline.metadata.intensity_range[1]:.0f}",
                })

    # ── Stage 2: Ground Classification ───────────────────────────────────
    if 2 in stages_to_run:
        step += 1
        with StageLogger("Ground Classification", stage_num=2, total_stages=total, log_dir=log_dir) as sl:

            # Check PDAL availability and warn accordingly
            with log_exception("PDAL import check", reraise=False):
                import pdal
                sl.info("PDAL available — using SMRF filter")

            use_ml = not no_ml
            pipeline.stage2_classify(use_ml_refine=use_ml)

            if pipeline.classified_las:
                try:
                    import laspy
                    las = laspy.read(str(pipeline.classified_las))
                    import numpy as np
                    n_ground = int((np.array(las.classification) == 2).sum())
                    sl.set_result({
                        "ground_pts":  f"{n_ground:,}",
                        "total_pts":   f"{las.header.point_count:,}",
                        "ground_pct":  f"{100*n_ground/las.header.point_count:.1f}%",
                        "output_file": pipeline.classified_las.name,
                    })
                except Exception as e:
                    sl.warning(f"Could not read classification stats: {e}")

    # ── Stage 3: DTM Generation ───────────────────────────────────────────
    if 3 in stages_to_run:
        step += 1
        with StageLogger("DTM Generation", stage_num=3, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage3_dtm()

            if pipeline.dtm_path:
                from src.dtm.dtm_generator import get_dtm_stats
                stats = get_dtm_stats(pipeline.dtm_path)
                sl.set_result({
                    "resolution_m":  stats["resolution_m"],
                    "relief_m":      f"{stats['relief_m']:.2f}",
                    "min_elev_m":    f"{stats['min_elevation_m']:.2f}",
                    "max_elev_m":    f"{stats['max_elevation_m']:.2f}",
                    "nodata_pct":    f"{stats['nodata_pct']:.1f}%",
                    "output_file":   pipeline.dtm_path.name,
                })

    # ── Stage 4: Hydrological Analysis ────────────────────────────────────
    if 4 in stages_to_run:
        step += 1
        with StageLogger("Hydrological Analysis", stage_num=4, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage4_hydrology(stream_threshold=stream_threshold)
            sl.set_result({
                "layers_exported": len(pipeline.hydro_paths),
                "stream_threshold": stream_threshold,
                "outputs": ", ".join(Path(v).name for v in pipeline.hydro_paths.values()),
            })

    # ── Stage 5: Waterlogging Prediction ─────────────────────────────────
    if 5 in stages_to_run:
        step += 1
        with StageLogger("Waterlogging Prediction", stage_num=5, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage5_waterlogging()
            sl.set_result({
                "model": "XGBoost",
                "threshold": pipeline.wl_predictor.threshold if pipeline.wl_predictor else "N/A",
                "output": "waterlogging_probability.tif",
            })

    # ── Stage 6: Drainage Network Design ─────────────────────────────────
    if 6 in stages_to_run:
        step += 1
        with StageLogger("Drainage Network Design", stage_num=6, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage6_drainage_design()

            if pipeline.designer and pipeline.designer.summary:
                s = pipeline.designer.summary
                sl.set_result({
                    "segments":      s.get("total_segments", 0),
                    "total_length_m": f"{s.get('total_length_m', 0):,.0f}",
                    "cost_INR":      f"₹{s.get('total_cost_inr', 0):,.0f}",
                })

    # ── Evaluation ───────────────────────────────────────────────────────
    if evaluate:
        with StageLogger("Accuracy Evaluation", stage_num=7, total_stages=total+1, log_dir=log_dir) as sl:
            eval_results = pipeline.run_evaluation()
            sl.set_result({
                "ground_f1":   eval_results.get("ground_classification", {}).get("f1_score", "N/A"),
                "dtm_rmse_m":  eval_results.get("dtm", {}).get("rmse_m", "N/A"),
                "wl_auc":      eval_results.get("waterlogging", {}).get("mean_metrics", {}).get("roc_auc", "N/A"),
                "metrics_file": "metrics.json",
            })

    # ════════════════════════════════════════════════════════════════════
    #  Final Summary
    # ════════════════════════════════════════════════════════════════════
    print_summary(output_dir=output_dir, save_json=True)


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        print_summary(save_json=True)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print_summary(save_json=True)
        sys.exit(1)
