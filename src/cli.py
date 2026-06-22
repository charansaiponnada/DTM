"""
src/cli.py
──────────
CLI entry points for the DTM Drainage AI pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path


def _bootstrap():
    """Ensure the repo root is on sys.path so 'from src' works."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_bootstrap()
import click
from loguru import logger
from src.logger import setup_logging, StageLogger, print_summary


@click.command()
@click.option("--input", "-i", default=None, type=click.Path(), help="Path to input LAS/LAZ")
@click.option("--batch", is_flag=True, default=False, help="Process all villages in config")
@click.option("--output", "-o", default="data/output", show_default=True, help="Output directory")
@click.option("--config", "-c", default="config/config.yaml", show_default=True, help="Config YAML")
@click.option("--log-dir", default="logs", show_default=True, help="Log directory")
@click.option("--log-level", default="DEBUG", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), show_default=True, help="Log verbosity")
@click.option("--no-ml", is_flag=True, default=False, help="Skip RF ground refinement")
@click.option("--stream-threshold", default=1000, show_default=True, type=int, help="Min accumulation cells for stream")
@click.option("--resolution", default=0.5, show_default=True, type=float, help="DTM resolution (m)")
@click.option("--stages", default="1,2,3,4,5,6", show_default=True, help="Comma-separated stage numbers")
@click.option("--evaluate", is_flag=True, default=False, help="Run accuracy evaluation")
@click.option("--pointnet", is_flag=True, default=False, help="Use PointNet classification")
def main(input, output, config, log_dir, log_level, no_ml, stream_threshold, resolution, stages, batch, evaluate, pointnet):
    """DTM Drainage AI Pipeline — LiDAR → DTM + risk + costed drainage."""
    _bootstrap()
    run_id = setup_logging(log_dir=log_dir, level=log_level)
    logger.info(f"Input  : {input}  Output : {output}  Config : {config}  Batch : {batch}")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    from src.pipeline import DTMDrainagePipeline, BatchPipelineRunner

    if batch:
        logger.info("Batch mode: processing all villages from config")
        runner = BatchPipelineRunner(config_path=config, base_output_dir=output, use_ml_refine=not no_ml, stream_threshold=stream_threshold, use_pointnet=pointnet)
        runner.run_all()
        print_summary(output_dir=output_dir, save_json=True)
        return

    if input is None:
        raise click.UsageError("--input required unless using --batch")
    if not Path(input).exists():
        raise click.UsageError(f"Input not found: {input}")

    stages_to_run = {int(s.strip()) for s in stages.split(",")}
    logger.info(f"Stages: {sorted(stages_to_run)}")
    pipeline = DTMDrainagePipeline(config_path=config, input_las=input, output_dir=output)

    if resolution != 0.5:
        pipeline.cfg["dtm"]["resolution"] = resolution

    total = len(stages_to_run)
    step = 0

    if 1 in stages_to_run:
        step += 1
        with StageLogger("Data Inspection", stage_num=1, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage1_inspect()
            if pipeline.metadata:
                sl.set_result({"points": f"{pipeline.metadata.point_count:,}", "density": f"{pipeline.metadata.density_pts_sqm:.1f} pts/m²", "crs": pipeline.metadata.crs_wkt or "MISSING"})

    if 2 in stages_to_run:
        step += 1
        with StageLogger("Ground Classification", stage_num=2, total_stages=total, log_dir=log_dir) as sl:
            try:
                import pdal
                sl.info("PDAL available — using SMRF filter")
            except ImportError:
                pass
            pipeline.stage2_classify(use_ml_refine=not no_ml, use_pointnet=pointnet)

    if 3 in stages_to_run:
        step += 1
        with StageLogger("DTM Generation", stage_num=3, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage3_dtm()
            if pipeline.dtm_path:
                from src.dtm.dtm_generator import get_dtm_stats
                stats = get_dtm_stats(pipeline.dtm_path)
                sl.set_result({"resolution_m": stats["resolution_m"], "relief_m": f"{stats['relief_m']:.2f}"})

    if 4 in stages_to_run:
        step += 1
        with StageLogger("Hydrological Analysis", stage_num=4, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage4_hydrology(stream_threshold=stream_threshold)
            sl.set_result({"layers": len(pipeline.hydro_paths)})

    if 5 in stages_to_run:
        step += 1
        with StageLogger("Waterlogging Prediction", stage_num=5, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage5_waterlogging()
            sl.set_result({"model": "XGBoost"})

    if 6 in stages_to_run:
        step += 1
        with StageLogger("Drainage Network Design", stage_num=6, total_stages=total, log_dir=log_dir) as sl:
            pipeline.stage6_drainage_design()

    if evaluate:
        with StageLogger("Accuracy Evaluation", stage_num=7, total_stages=total + 1, log_dir=log_dir) as sl:
            eval_results = pipeline.run_evaluation()
            sl.set_result({"dtm_rmse_m": eval_results.get("dtm", {}).get("rmse_m", "N/A")})

    print_summary(output_dir=output_dir, save_json=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unhandled: {e}")
        sys.exit(1)
