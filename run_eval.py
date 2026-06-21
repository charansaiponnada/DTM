"""
run_eval.py
───────────
Run cross-village and ablation evaluation.
"""

import sys
from pathlib import Path


def main():
    from src.evaluation import run_cross_village_evaluation, evaluate_ground_classification_ablation
    from loguru import logger

    # ── Cross-village evaluation ────────────────────────────────────────
    train_las = "data/input/DEVDI_511671.las"
    test_las_paths = [
        "data/input/DEVDI_511671.las",
        "data/input/Gujrat_Point_Cloud/KHAPRETA_510206.laz",
    ]

    if Path(train_las).exists():
        logger.info("=" * 60)
        logger.info("CROSS-VILLAGE EVALUATION")
        logger.info("=" * 60)
        cv_results = run_cross_village_evaluation(
            train_las_path=train_las,
            test_las_paths=test_las_paths,
            train_name="DEVDI",
        )
        logger.info(f"Cross-village results: {cv_results}")

    # ── Ablation on DEVDI ──────────────────────────────────────────────
    output_dir = Path("data/output/DEVDI")
    classified_las = output_dir / "classified_ground.las"
    smrf_las = output_dir / "_smrf_only.las"
    pn_las = output_dir / "classified_pointnet.las"

    if classified_las.exists() and smrf_las.exists():
        logger.info("=" * 60)
        logger.info("ABLATION EVALUATION")
        logger.info("=" * 60)
        ablation = evaluate_ground_classification_ablation(
            classified_las_path=classified_las,
            smrf_only_las_path=smrf_las,
            pointnet_las_path=pn_las if pn_las.exists() else None,
        )
        logger.info(f"Ablation: {ablation}")

    logger.success("Evaluation complete.")


if __name__ == "__main__":
    main()
