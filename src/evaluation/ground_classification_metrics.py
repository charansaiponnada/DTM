"""
src/evaluation/ground_classification_metrics.py
────────────────────────────────────────────────
Evaluate ground / non-ground classification accuracy.

Strategy
--------
Primary: Compare against a hand-labelled or consensus reference.
Consensus reference = SMRF + CSF agreement (both classifiers label the same
point as ground → highly reliable pseudo-ground-truth).

If no reference is available, falls back to the z-percentile heuristic.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
        jaccard_score,
    )
except ImportError:
    raise ImportError("scikit-learn is required: pip install scikit-learn")

try:
    import laspy
except ImportError:
    raise ImportError("laspy is required: pip install laspy[lazrs]")


def evaluate_ground_classification(
    classified_las_path: str | Path,
    reference_las_path: Optional[str | Path] = None,
    sample_n: int = 50_000,
    ground_class_code: int = 2,
    seed: int = 42,
) -> Dict:
    classified_las_path = Path(classified_las_path)
    logger.info(f"Evaluating classification: {classified_las_path.name}")

    classified_las = laspy.read(str(classified_las_path))
    classification = np.array(classified_las.classification, dtype=np.int32)
    y_pred = (classification == ground_class_code).astype(np.int32)

    if reference_las_path is not None:
        ref_path = Path(reference_las_path)
        logger.info(f"  Reference: {ref_path.name}")
        ref_las  = laspy.read(str(ref_path))
        y_true   = (np.array(ref_las.classification) == ground_class_code).astype(np.int32)
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Point count mismatch: classified={len(y_pred)}, reference={len(y_true)}"
            )
        ref_type = "external"
    else:
        logger.warning(
            "No reference LAS provided – using z-percentile heuristic as pseudo ground-truth."
        )
        z = np.array(classified_las.z, dtype=np.float32)
        z_low   = np.percentile(z, 5)
        y_true  = (z <= z_low + 0.5).astype(np.int32)
        ref_type = "heuristic_proxy"

    rng = np.random.default_rng(seed)
    N   = len(y_true)
    if N > sample_n:
        idx    = rng.choice(N, size=sample_n, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    n_samples = len(y_true)

    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    iou  = float(jaccard_score(y_true, y_pred, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred).tolist()
    rpt  = classification_report(y_true, y_pred, target_names=["non-ground", "ground"])

    metrics = {
        "accuracy":              round(acc,  4),
        "precision":             round(prec, 4),
        "recall":                round(rec,  4),
        "f1_score":              round(f1,   4),
        "iou":                   round(iou,  4),
        "confusion_matrix":      cm,
        "classification_report": rpt,
        "n_samples":             n_samples,
        "reference_type":        ref_type,
    }

    logger.success(f"Ground classification  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  F1={f1:.4f}  IoU={iou:.4f}")
    logger.info("\n" + rpt)
    return metrics


def evaluate_ground_classification_ablation(
    classified_las_path: str | Path,
    smrf_only_las_path: str | Path,
    pointnet_las_path: Optional[str | Path] = None,
    reference_las_path: Optional[str | Path] = None,
    sample_n: int = 50_000,
    ground_class_code: int = 2,
    seed: int = 42,
) -> Dict:
    """
    Run ablation: compare SMRF-only vs RF-refined vs PointNet results
    against the same reference. Returns a table of metrics per method.
    """
    methods = {
        "SMRF only": smrf_only_las_path,
        "SMRF + RF": classified_las_path,
    }
    if pointnet_las_path:
        methods["PointNet"] = pointnet_las_path

    ablation = {}
    for method, las_path in methods.items():
        las_path = Path(las_path)
        if not las_path.exists():
            logger.warning(f"Ablation: {method} LAS not found at {las_path}")
            continue
        metrics = evaluate_ground_classification(
            classified_las_path=las_path,
            reference_las_path=reference_las_path,
            sample_n=sample_n,
            ground_class_code=ground_class_code,
            seed=seed,
        )
        ablation[method] = {
            "accuracy":  metrics["accuracy"],
            "f1_score":  metrics["f1_score"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "iou":       metrics["iou"],
        }

    logger.info("=" * 60)
    logger.info("ABLATION: Ground Classification Methods")
    logger.info(f"{'Method':<20} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'IoU':>8}")
    logger.info("-" * 60)
    for method, m in ablation.items():
        logger.info(f"{method:<20} {m['accuracy']:>8.4f} {m['f1_score']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['iou']:>8.4f}")
    logger.info("=" * 60)

    return {"ablation": ablation, "reference_type": metrics.get("reference_type", "unknown")}
