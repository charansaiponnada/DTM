"""
run_all.py — Single-command entry point for the full DTM Drainage AI pipeline.

Usage
-----
    python run_all.py                          # batch process all villages
    python run_all.py --village DEVDI          # single village
    python run_all.py --no-ml --no-pointnet    # fast mode (no ML refinement)
    python run_all.py --evaluate-only          # only re-run evaluation on existing outputs

This script:
  1. Loads config and processes all 10 villages (2 base + 8 tiles) in batch mode.
  2. Runs the full 6-stage pipeline for each village.
  3. If --pointnet is set, trains PointNet on DEVDI and applies across villages.
  4. If --evaluate is set (default), computes ablation, gold-standard waterlogging,
     cross-village, and drainage metrics for each village.
  5. Generates a cross-village evaluation summary.
  6. Writes a consolidated metrics report to data/output/_reports/.
"""

import sys, os, json, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import yaml
except ImportError:
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DTM Drainage AI — Full Pipeline Runner")
    parser.add_argument("--village", "-v", type=str, default=None,
                        help="Process a single village by name (default: all)")
    parser.add_argument("--no-ml", action="store_true", default=False,
                        help="Skip RF refinement of ground classification")
    parser.add_argument("--no-pointnet", action="store_true", default=False,
                        help="Skip PointNet classification")
    parser.add_argument("--pointnet", action="store_true", default=False,
                        help="Run PointNet deep-learning classification")
    parser.add_argument("--no-evaluate", action="store_true", default=False,
                        help="Skip evaluation after pipeline completes")
    parser.add_argument("--evaluate-only", action="store_true", default=False,
                        help="Only re-run evaluation on existing outputs (no pipeline stages)")
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml",
                        help="Config YAML path")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                        help="Number of villages to process in parallel")
    parser.add_argument("--stages", type=str, default="1,2,3,4,5,6",
                        help="Comma-separated stages to run")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip villages that already have output metrics.json")
    parser.add_argument("--output", "-o", type=str, default="data/output",
                        help="Base output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) if yaml else json.load(f)

    villages = cfg["data"]["villages"]
    if args.village:
        villages = [v for v in villages if v["name"] == args.village]
        if not villages:
            print(f"[ERROR] Village '{args.village}' not found in config")
            sys.exit(1)

    if args.evaluate_only:
        logger.info("=== EVALUATE-ONLY MODE ===")
        for v in villages:
            out_dir = Path(args.output) / v["output_subdir"]
            metrics_file = out_dir / "metrics.json"
            if metrics_file.exists():
                from src.pipeline import DTMDrainagePipeline
                pipe = DTMDrainagePipeline(config_path=cfg_path, output_dir=str(out_dir))
                pipe.run_evaluation()
        return

    # ── Batch pipeline ─────────────────────────────────────────
    logger.info(f"Processing {len(villages)} villages (parallel={args.parallel}) ...")

    def process_village(v):
        name = v["name"]
        out_dir = Path(args.output) / v["output_subdir"]
        if args.skip_existing and (out_dir / "metrics.json").exists():
            logger.info(f"[SKIP] {name} — metrics.json already exists")
            return {"village": name, "status": "skipped"}

        from src.pipeline import DTMDrainagePipeline
        pipe = DTMDrainagePipeline(
            config_path=cfg_path,
            input_las=v["path"],
            output_dir=str(out_dir),
            tile_filter=v.get("tile_filter"),
        )
        pipe.run(
            use_ml_refine=not args.no_ml,
            use_pointnet=args.pointnet,
            run_evaluation=not args.no_evaluate,
            stages=args.stages,
        )
        return {"village": name, "status": "done"}

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(process_village, v): v["name"] for v in villages}
            for f in as_completed(futures):
                result = f.result()
                logger.success(f"  {result['village']}: {result['status']}")
    else:
        for v in villages:
            result = process_village(v)
            logger.success(f"  {result['village']}: {result['status']}")

    # ── Cross-village summary ──────────────────────────────────
    logger.info("=== GENERATING CROSS-VILLAGE SUMMARY ===")
    try:
        from src.evaluation import run_cross_village_evaluation

        all_metrics = {}
        for v in villages:
            metrics_file = Path(args.output) / v["output_subdir"] / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    all_metrics[v["name"]] = json.load(f)

        # Cross-village RF transfer
        village_paths = [(v["name"], v["path"]) for v in villages]
        if len(village_paths) >= 2:
            cv_summary = {}
            for train_name, train_path in village_paths:
                test_paths = [p for n, p in village_paths if n != train_name]
                cv = run_cross_village_evaluation(
                    train_las_path=train_path,
                    test_las_paths=test_paths[:3],  # limit to 3 test villages for speed
                    train_name=train_name,
                    sample_n=20_000,
                )
                cv_summary[train_name] = cv

            report_dir = Path(args.output) / "_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            with open(report_dir / "cross_village_summary.json", "w") as f:
                json.dump(cv_summary, f, indent=2, default=str)
            logger.success(f"Cross-village summary → {report_dir/'cross_village_summary.json'}")

        # Consolidated metrics dashboard
        report_path = Path(args.output) / "_reports" / "all_village_metrics.json"
        with open(report_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.success(f"All-village metrics → {report_path}")

    except Exception as exc:
        logger.warning(f"Cross-village summary failed: {exc}")

    print("\n[DONE] Batch pipeline complete. See data/output/_reports/ for consolidated metrics.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} minutes")
