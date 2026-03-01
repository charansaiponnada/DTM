"""
pipelines/full_pipeline.py
───────────────────────────
End-to-end pipeline orchestrator for the MoPR Hackathon DTM + Drainage
AI challenge.

Pipeline stages:
  1. Data inspection & tiling
  2. Ground classification (SMRF + ML refinement)
  3. DTM generation (IDW interpolation → COG)
  4. Hydrological analysis (fill → flow direction → accumulation → TWI)
  5. Waterlogging prediction (XGBoost)
  6. Drainage network design (MST + Manning's sizing)
  7. GIS output packaging (COG rasters + GPKG vectors)

All outputs conform to OGC format standards as required by the hackathon:
  - Raster  → Cloud-Optimized GeoTIFF (.tif)
  - Vector  → GeoPackage (.gpkg)
  - LiDAR   → LAS 1.4 (.las)
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import yaml
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


class DTMDrainagePipeline:
    """
    Orchestrates the complete DTM + Drainage AI workflow.

    Parameters
    ----------
    config_path : path to config/config.yaml
    input_las   : override input LAS/LAZ path (else uses config)
    output_dir  : override output directory
    """

    def __init__(
        self,
        config_path: str | Path = "config/config.yaml",
        input_las: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
    ):
        with open(config_path, encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        self.input_las  = Path(input_las) if input_las else None
        self.output_dir = Path(output_dir or self.cfg["data"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated as pipeline runs
        self.metadata        = None
        self.classified_las  = None
        self.dtm_path        = None
        self.hydro_analyzer  = None
        self.hydro_paths     = {}
        self.wl_predictor    = None
        self.prob_map        = None
        self.designer        = None
        self.results         = {}

        # Auto-detect: if the input file is a TIF, treat it as the DTM
        # (allows running stages 4+ directly on a pre-built DTM)
        if self.input_las and self.input_las.suffix.lower() in (".tif", ".tiff"):
            self.dtm_path = self.input_las
            logger.info(f"Input is a raster — treating as DTM for stages 4+: {self.dtm_path.name}")
        # Auto-detect: if a dtm.tif already exists in the output dir, use it
        elif (self.output_dir / "dtm.tif").exists():
            self.dtm_path = self.output_dir / "dtm.tif"
            logger.info(f"Found existing DTM → {self.dtm_path}")

        # Auto-detect hydro layer paths from a previous Stage 4 run
        _hydro_names = {
            "twi": "twi.tif",
            "slope": "slope.tif",
            "flow_accumulation": "flow_accumulation.tif",
            "flow_direction": "flow_direction.tif",
            "gpkg": "drainage_network.gpkg",
        }
        for key, fname in _hydro_names.items():
            candidate = self.output_dir / fname
            if candidate.exists():
                self.hydro_paths[key] = candidate
        if self.hydro_paths:
            logger.info(f"Found existing hydro layers: {', '.join(self.hydro_paths)}")

        logger.info(
            f"Pipeline initialised\n"
            f"  Output dir : {self.output_dir}\n"
            f"  CRS        : {self.cfg['project']['crs']}"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Stage 1 – Data Inspection
    # ══════════════════════════════════════════════════════════════════════

    def stage1_inspect(self) -> "DTMDrainagePipeline":
        from src.preprocessing.point_cloud_loader import inspect

        las_path = self._resolve_input()
        logger.info("═" * 60)
        logger.info("STAGE 1: Data Inspection")
        logger.info("═" * 60)

        self.metadata = inspect(las_path)
        self.results["metadata"] = {
            "point_count":        self.metadata.point_count,
            "density_pts_sqm":    round(self.metadata.density_pts_sqm, 2),
            "has_classification": self.metadata.has_classification,
            "intensity_range":    self.metadata.intensity_range,
            "crs":                self.metadata.crs_wkt or "MISSING (assuming EPSG:32643)",
        }
        return self

    # ══════════════════════════════════════════════════════════════════════
    #  Stage 2 – Ground Classification
    # ══════════════════════════════════════════════════════════════════════

    def stage2_classify(
        self,
        use_ml_refine: bool = True,
        use_tiling: bool    = True,
    ) -> "DTMDrainagePipeline":
        from src.preprocessing.ground_classifier import classify_ground_full_pipeline
        from src.preprocessing.point_cloud_loader import load_tiles

        las_path = self._resolve_input()
        logger.info("═" * 60)
        logger.info("STAGE 2: Ground Classification")
        logger.info("═" * 60)

        classified_path = self.output_dir / "classified_ground.las"
        cfg_gc = self.cfg["ground_classification"]

        if use_tiling and self.metadata and self.metadata.point_count > 10_000_000:
            logger.info(
                f"Large file ({self.metadata.point_count:,} pts) – using tiled processing"
            )
            tile_dir    = self.output_dir / "_tiles"
            tile_paths  = load_tiles(
                las_path,
                tile_size=self.cfg["preprocessing"]["tile_size"],
                buffer=self.cfg["preprocessing"]["tile_buffer"],
                output_dir=tile_dir,
            )
            # Classify each tile and merge
            import laspy, numpy as np
            classified_tiles = []
            for tile in tile_paths:
                out_tile = tile.parent / f"classified_{tile.name}"
                classify_ground_full_pipeline(
                    tile, out_tile,
                    use_ml_refine=use_ml_refine,
                    smrf_kwargs={
                        "slope":     cfg_gc["smrf"]["slope"],
                        "window":    cfg_gc["smrf"]["window"],
                        "threshold": cfg_gc["smrf"]["threshold"],
                    }
                )
                classified_tiles.append(out_tile)

            # Merge tiles → single output
            _merge_las_tiles(classified_tiles, classified_path)
        else:
            classify_ground_full_pipeline(
                las_path, classified_path,
                use_ml_refine=use_ml_refine,
                smrf_kwargs={
                    "slope":     cfg_gc["smrf"]["slope"],
                    "window":    cfg_gc["smrf"]["window"],
                    "threshold": cfg_gc["smrf"]["threshold"],
                }
            )

        self.classified_las = classified_path
        self.results["classified_las"] = str(classified_path)
        return self

    # ══════════════════════════════════════════════════════════════════════
    #  Stage 3 – DTM Generation
    # ══════════════════════════════════════════════════════════════════════

    def stage3_dtm(self) -> "DTMDrainagePipeline":
        from src.dtm.dtm_generator import generate_dtm, get_dtm_stats
        from src.dtm.terrain_analysis import compute_all_derivatives

        logger.info("═" * 60)
        logger.info("STAGE 3: DTM Generation & Terrain Derivatives")
        logger.info("═" * 60)

        cfg_dtm      = self.cfg["dtm"]
        dtm_path     = self.output_dir / cfg_dtm["output"]["dtm"] if "output" in cfg_dtm \
                        else self.output_dir / "dtm.tif"
        classified   = self.classified_las or self._resolve_input()

        self.dtm_path = generate_dtm(
            classified,
            dtm_path,
            resolution   = cfg_dtm["resolution"],
            idw_power    = cfg_dtm["interpolation"]["idw_power"],
            idw_radius   = cfg_dtm["interpolation"]["idw_radius"],
            smooth_sigma = cfg_dtm["smoothing"]["sigma"],
            crs          = self.cfg["project"]["crs"],
        )

        stats = get_dtm_stats(self.dtm_path)
        self.results["dtm"] = {"path": str(self.dtm_path), **stats}
        logger.info(f"DTM stats: {stats}")

        # ── Terrain derivatives (slope, aspect, curvature, TPI, hillshade) ──
        logger.info("Computing terrain derivatives …")
        deriv_paths = compute_all_derivatives(
            dtm_path   = self.dtm_path,
            output_dir = self.output_dir,
            crs        = self.cfg["project"]["crs"],
        )
        self.results["terrain_derivatives"] = {k: str(v) for k, v in deriv_paths.items()}
        logger.info(f"Terrain layers: {list(deriv_paths.keys())}")
        return self

    # ══════════════════════════════════════════════════════════════════════
    #  Stage 4 – Hydrological Analysis
    # ══════════════════════════════════════════════════════════════════════

    def stage4_hydrology(self, stream_threshold: int = 1000) -> "DTMDrainagePipeline":
        from src.hydrology.flow_analysis import HydrologicalAnalyzer

        if self.dtm_path is None or not Path(self.dtm_path).exists():
            raise RuntimeError(
                "Stage 4 requires a DTM raster from Stage 3. "
                "Run Stage 3 first or provide dtm_path."
            )

        logger.info("═" * 60)
        logger.info("STAGE 4: Hydrological Analysis")
        logger.info("═" * 60)

        ha = HydrologicalAnalyzer.from_dtm(self.dtm_path, crs=self.cfg["project"]["crs"])
        self.hydro_paths = ha.export_all(
            self.output_dir,
            stream_threshold=stream_threshold,
        )
        self.hydro_analyzer = ha
        self.results["hydrology"] = {k: str(v) for k, v in self.hydro_paths.items()}
        return self

    # ══════════════════════════════════════════════════════════════════════
    #  Stage 5 – Waterlogging Prediction
    # ══════════════════════════════════════════════════════════════════════

    def stage5_waterlogging(self) -> "DTMDrainagePipeline":
        from src.hydrology.waterlogging_predictor import (
            build_feature_stack, generate_terrain_labels, WaterloggingPredictor
        )

        if self.dtm_path is None or not Path(self.dtm_path).exists():
            raise RuntimeError(
                "Stage 5 requires a DTM raster from Stage 3. "
                "Run Stages 3-4 first or ensure dtm_path is set."
            )
        # Graceful fallback: files may exist on disk even if not in session hydro_paths
        _twi   = self.hydro_paths.get("twi",              self.output_dir / "twi.tif")
        _facc  = self.hydro_paths.get("flow_accumulation", self.output_dir / "flow_accumulation.tif")
        _slope = self.hydro_paths.get("slope",             self.output_dir / "slope.tif")
        missing = [str(p) for p in (_twi, _facc, _slope) if not Path(p).exists()]
        if missing:
            raise RuntimeError(
                "Stage 5 requires hydrological layers from Stage 4 "
                f"(missing: {', '.join(missing)}). Run Stage 4 first."
            )

        logger.info("═" * 60)
        logger.info("STAGE 5: Waterlogging Prediction")
        logger.info("═" * 60)

        feature_stack, valid_mask, transform = build_feature_stack(
            dtm_path       = self.dtm_path,
            twi_path       = _twi,
            flow_acc_path  = _facc,
            slope_path     = _slope,
        )

        labels = generate_terrain_labels(feature_stack, valid_mask)

        cfg_wl = self.cfg["waterlogging"]
        self.wl_predictor = WaterloggingPredictor(
            n_estimators     = cfg_wl["xgboost"]["n_estimators"],
            max_depth        = cfg_wl["xgboost"]["max_depth"],
            learning_rate    = cfg_wl["xgboost"]["learning_rate"],
            scale_pos_weight = cfg_wl["xgboost"]["scale_pos_weight"],
            threshold        = cfg_wl["threshold"],
        )
        self.wl_predictor.fit(feature_stack, labels, valid_mask)

        self.prob_map  = self.wl_predictor.predict_proba_map(feature_stack, valid_mask)
        prob_cog_path  = self.output_dir / "waterlogging_probability.tif"
        gpkg_path      = self.output_dir / "drainage_network.gpkg"

        self.wl_predictor.export_probability_cog(self.prob_map, transform, prob_cog_path)
        self.wl_predictor.export_hotspot_gpkg(self.prob_map, transform, gpkg_path)
        self.wl_predictor.save(self.output_dir / "models" / "waterlogging_xgb.joblib")

        self.results["waterlogging"] = {"probability_cog": str(prob_cog_path)}
        return self

    # ══════════════════════════════════════════════════════════════════════
    #  Stage 6 – Drainage Design
    # ══════════════════════════════════════════════════════════════════════

    def stage6_drainage_design(self) -> "DTMDrainagePipeline":
        from src.hydrology.drainage_network import (
            DrainageNetworkDesigner, DrainageDesignParameters
        )
        import geopandas as gpd

        if self.dtm_path is None or not Path(self.dtm_path).exists():
            raise RuntimeError(
                "Stage 6 requires a DTM raster from Stage 3. "
                "Run Stage 3 first or ensure dtm_path is set."
            )

        logger.info("═" * 60)
        logger.info("STAGE 6: Drainage Network Design")
        logger.info("═" * 60)

        gpkg_path = self.output_dir / "drainage_network.gpkg"
        cfg_dr = self.cfg["drainage"]

        params = DrainageDesignParameters(
            return_period_yr        = cfg_dr["design_return_period"],
            rainfall_intensity_mmhr = cfg_dr["rainfall_intensity"],
            runoff_coefficient      = cfg_dr["runoff_coefficient"],
            manning_n_earthen       = cfg_dr["manning_n"],
            cost_earthen_inr_m      = cfg_dr["cost_per_metre_channel"],
            cost_pipe_inr_m         = cfg_dr["cost_per_metre_pipe"],
        )

        # Load streams from GPKG
        try:
            streams_gdf = gpd.read_file(str(gpkg_path), layer="drainage_channels")
        except Exception:
            logger.warning("No stream layer found – using empty GeoDataFrame")
            streams_gdf = gpd.GeoDataFrame(columns=["geometry", "order"])

        self.designer = DrainageNetworkDesigner(self.dtm_path, params)
        self.designer.load_inputs(streams_gdf)
        self.designer.design()
        self.designer.export(gpkg_path)
        self.designer.print_summary()

        self.results["drainage"] = {
            "gpkg": str(gpkg_path),
            **self.designer.summary,
        }
        return self

    # ══════════════════════════════════════════════════════════════════════
    #  Full Run
    # ══════════════════════════════════════════════════════════════════════

    def run(
        self,
        use_ml_refine: bool = True,
        stream_threshold: int = 1000,
    ) -> dict:
        """Execute all six pipeline stages in sequence."""
        t0 = time.time()
        console.rule("[bold blue]DTM Drainage AI Pipeline Starting[/bold blue]")

        self.stage1_inspect()
        self.stage2_classify(use_ml_refine=use_ml_refine)
        self.stage3_dtm()
        self.stage4_hydrology(stream_threshold=stream_threshold)
        self.stage5_waterlogging()
        self.stage6_drainage_design()

        elapsed = time.time() - t0
        self.results["runtime_seconds"] = round(elapsed, 1)

        console.rule("[bold green]Pipeline Complete[/bold green]")
        logger.success(
            f"All stages complete in {elapsed:.1f}s\n"
            f"Output directory: {self.output_dir}"
        )
        _print_output_summary(self.results, self.output_dir)

        # Save results summary JSON
        import json
        summary_path = self.output_dir / "pipeline_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results summary saved → {summary_path.name}")

        return self.results

    def run_evaluation(self) -> dict:
        """
        Run all evaluation metrics on completed pipeline outputs.
        Saves metrics to output_dir/metrics.json.

        Must be called AFTER run() has completed at least Stages 2–6.
        """
        import json
        from src.evaluation import (
            evaluate_ground_classification,
            evaluate_dtm_accuracy,
            evaluate_drainage_design,
        )

        logger.info("═" * 60)
        logger.info("EVALUATION: Computing accuracy metrics")
        logger.info("═" * 60)

        eval_results: dict = {}

        # ── Ground classification ─────────────────────────────────────
        # Search multiple candidate locations for classified_ground.las
        _las_candidates = [
            self.classified_las,
            self.output_dir / "classified_ground.las",
            self.input_las,
            Path("data/output/classified_ground.las"),
        ]
        classified_las = next(
            (p for p in _las_candidates if p and Path(p).exists()), None
        )
        if classified_las and Path(classified_las).exists():
            try:
                eval_results["ground_classification"] = evaluate_ground_classification(
                    classified_las_path = classified_las
                )
            except Exception as exc:
                logger.warning(f"Ground classification eval failed: {exc}")

        # ── DTM accuracy ──────────────────────────────────────────────
        dtm_path = self.dtm_path or self.output_dir / "dtm.tif"
        if dtm_path and Path(dtm_path).exists():
            try:
                eval_results["dtm"] = evaluate_dtm_accuracy(dtm_path)
            except Exception as exc:
                logger.warning(f"DTM eval failed: {exc}")

        # ── Waterlogging model ────────────────────────────────────────
        # Load saved model from disk if not already in session memory
        if self.wl_predictor is None:
            _model_path = self.output_dir / "models" / "waterlogging_xgb.joblib"
            if _model_path.exists():
                try:
                    from src.hydrology.waterlogging_predictor import WaterloggingPredictor
                    self.wl_predictor = WaterloggingPredictor.load(_model_path)
                    logger.info(f"Loaded saved predictor from {_model_path.name}")
                except Exception as exc:
                    logger.warning(f"Could not load saved predictor: {exc}")

        if self.wl_predictor is not None:
            try:
                from src.hydrology.waterlogging_predictor import build_feature_stack, generate_terrain_labels
                from src.evaluation import evaluate_waterlogging_model
                feature_stack, valid_mask, _ = build_feature_stack(
                    dtm_path      = self.dtm_path,
                    twi_path      = self.hydro_paths.get("twi", self.output_dir / "twi.tif"),
                    flow_acc_path = self.hydro_paths.get("flow_accumulation", self.output_dir / "flow_accumulation.tif"),
                    slope_path    = self.hydro_paths.get("slope", self.output_dir / "slope.tif"),
                )
                labels = generate_terrain_labels(feature_stack, valid_mask)
                eval_results["waterlogging"] = evaluate_waterlogging_model(
                    predictor     = self.wl_predictor,
                    feature_stack = feature_stack,
                    labels        = labels,
                    valid_mask    = valid_mask,
                    cv_folds      = int(self.cfg.get("waterlogging", {}).get("cv_folds", 5)),
                )
            except Exception as exc:
                logger.warning(f"Waterlogging eval failed: {exc}")

        # ── Drainage design ───────────────────────────────────────────
        gpkg_path = self.output_dir / "drainage_network.gpkg"
        if gpkg_path.exists():
            try:
                eval_results["drainage"] = evaluate_drainage_design(gpkg_path)
            except Exception as exc:
                logger.warning(f"Drainage eval failed: {exc}")

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, default=str)
        logger.success(f"Metrics saved → {metrics_path}")

        self.results["evaluation"] = eval_results
        return eval_results

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_input(self) -> Path:
        if self.input_las:
            return self.input_las
        files = self.cfg["data"].get("files", [])
        if files:
            return Path(files[0]["path"])
        raise ValueError("No input LAS file specified in config or constructor.")


# ══════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════

def _merge_las_tiles(tile_paths: list, output_path: Path):
    """Merge classified tile LAS files into a single output LAS."""
    import laspy
    logger.info(f"Merging {len(tile_paths)} classified tiles …")
    first = laspy.read(str(tile_paths[0]))
    # Stream all tiles into a single writer so no tile is silently dropped
    with laspy.open(str(output_path), mode="w", header=first.header) as writer:
        writer.write_points(first.points)
        for tp in tile_paths[1:]:
            las = laspy.read(str(tp))
            writer.write_points(las.points)
    logger.success(f"Merged LAS → {output_path.name}")


def _print_output_summary(results: dict, output_dir: Path):
    """Print a rich table summarising all output files."""
    from rich.table import Table
    table = Table(title="Output Files Summary", show_header=True)
    table.add_column("Format", style="cyan")
    table.add_column("File")
    table.add_column("Description")

    for tif in output_dir.glob("*.tif"):
        table.add_row("COG (raster)", tif.name, "Cloud-Optimized GeoTIFF")
    for gpkg in output_dir.glob("*.gpkg"):
        table.add_row("GPKG (vector)", gpkg.name, "GeoPackage – all vector layers")
    for las in output_dir.glob("*.las"):
        table.add_row("LAS (point cloud)", las.name, "Classified ground points")

    console.print(table)


# ══════════════════════════════════════════════════════════════════════════
#  Batch Multi-Village Runner
# ══════════════════════════════════════════════════════════════════════════

class BatchPipelineRunner:
    """
    Runs the full DTM + Drainage pipeline over all villages defined in config.

    Usage
    -----
      runner = BatchPipelineRunner("config/config.yaml", base_output_dir="data/output")
      summary = runner.run_all()
    """

    def __init__(
        self,
        config_path: str | Path = "config/config.yaml",
        base_output_dir: Optional[str | Path] = None,
        use_ml_refine: bool = True,
        stream_threshold: int = 1000,
    ):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.config_path      = Path(config_path)
        self.base_output_dir  = Path(base_output_dir or self.cfg["data"]["output_dir"])
        self.use_ml_refine    = use_ml_refine
        self.stream_threshold = stream_threshold
        self.summary: dict    = {}

    def run_all(self) -> dict:
        """
        Process every village listed in config.data.villages.

        Returns
        -------
        dict  mapping village_name -> stage results dict
        """
        villages = self.cfg.get("data", {}).get("villages", [])
        if not villages:
            logger.warning(
                "No villages defined in config.data.villages – "
                "falling back to config.data.files"
            )
            villages = [
                {
                    "name": f["name"],
                    "path": f["path"],
                    "output_subdir": f["name"],
                }
                for f in self.cfg["data"].get("files", [])
            ]

        if not villages:
            raise ValueError("No input files found in config. Add villages to config.yaml.")

        total = len(villages)
        logger.info(f"Batch runner: {total} village(s) to process")
        console.rule(f"[bold blue]Batch Pipeline – {total} villages[/bold blue]")

        for i, village in enumerate(villages, 1):
            name       = village["name"]
            las_path   = Path(village["path"])
            out_subdir = self.base_output_dir / village.get("output_subdir", name)

            console.rule(f"[cyan]Village {i}/{total}: {name}[/cyan]")
            logger.info(f"Starting village: {name}  ({las_path})")

            if not las_path.exists():
                logger.error(f"Input file for {name} not found: {las_path} – SKIPPING")
                self.summary[name] = {"status": "skipped", "reason": "file not found"}
                continue

            try:
                pipeline = DTMDrainagePipeline(
                    config_path = self.config_path,
                    input_las   = las_path,
                    output_dir  = out_subdir,
                )
                results = pipeline.run(
                    use_ml_refine    = self.use_ml_refine,
                    stream_threshold = self.stream_threshold,
                )
                self.summary[name] = {"status": "success", **results}
                logger.success(f"Village {name} complete → {out_subdir}")

            except Exception as exc:
                logger.exception(f"Village {name} FAILED: {exc}")
                self.summary[name] = {"status": "failed", "error": str(exc)}

        self._print_batch_summary()
        return self.summary

    def _print_batch_summary(self):
        from rich.table import Table
        table = Table(title="Batch Run Summary", show_header=True)
        table.add_column("Village", style="bold")
        table.add_column("Status")
        table.add_column("DTM")
        table.add_column("Runtime (s)")

        for name, res in self.summary.items():
            status = res.get("status", "?")
            dtm    = res.get("dtm", {}).get("path", "–") if isinstance(res.get("dtm"), dict) else "–"
            rt     = str(res.get("runtime_seconds", "–"))
            color  = "green" if status == "success" else ("yellow" if status == "skipped" else "red")
            table.add_row(name, f"[{color}]{status}[/{color}]", dtm, rt)

        console.print(table)


# ══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════

import click

@click.group()
def cli():
    """DTM Drainage AI — MoPR Geospatial Hackathon"""
    pass


@cli.command()
@click.option("--input",  "-i", required=True,  help="Input LAS/LAZ file path")
@click.option("--output", "-o", default="data/output", help="Output directory")
@click.option("--config", "-c", default="config/config.yaml", help="Config YAML path")
@click.option("--no-ml", is_flag=True, default=False, help="Skip ML refinement of ground classification")
@click.option("--stream-threshold", default=1000, help="Flow accumulation threshold for stream extraction")
def run(input, output, config, no_ml, stream_threshold):
    """Run the full pipeline on a single LAS/LAZ file."""
    pipeline = DTMDrainagePipeline(
        config_path = config,
        input_las   = input,
        output_dir  = output,
    )
    pipeline.run(
        use_ml_refine    = not no_ml,
        stream_threshold = stream_threshold,
    )


@cli.command()
@click.option("--output", "-o", default="data/output", help="Base output directory")
@click.option("--config", "-c", default="config/config.yaml", help="Config YAML path")
@click.option("--no-ml", is_flag=True, default=False, help="Skip ML refinement")
@click.option("--stream-threshold", default=1000, help="Flow accumulation threshold")
def batch(output, config, no_ml, stream_threshold):
    """Run the pipeline over all villages defined in config.data.villages."""
    runner = BatchPipelineRunner(
        config_path      = config,
        base_output_dir  = output,
        use_ml_refine    = not no_ml,
        stream_threshold = stream_threshold,
    )
    runner.run_all()


# Legacy single-command entry point (backwards compat)
@click.command()
@click.option("--input",  "-i", required=True,  help="Input LAS/LAZ file path")
@click.option("--output", "-o", default="data/output", help="Output directory")
@click.option("--config", "-c", default="config/config.yaml", help="Config YAML path")
@click.option("--no-ml", is_flag=True, default=False, help="Skip ML refinement of ground classification")
@click.option("--stream-threshold", default=1000, help="Flow accumulation threshold for stream extraction")
def main(input, output, config, no_ml, stream_threshold):
    """
    DTM Drainage AI — MoPR Geospatial Hackathon

    Processes a point cloud LAS/LAZ file to produce:
      - Digital Terrain Model (COG)
      - Waterlogging probability map (COG)
      - Optimized drainage network design (GPKG)
    """
    pipeline = DTMDrainagePipeline(
        config_path = config,
        input_las   = input,
        output_dir  = output,
    )
    pipeline.run(
        use_ml_refine    = not no_ml,
        stream_threshold = stream_threshold,
    )


if __name__ == "__main__":
    main()
