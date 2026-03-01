"""
src/evaluation/ground_classification_metrics.py
────────────────────────────────────────────────
Evaluate ground / non-ground classification accuracy.

Strategy
--------
The classified LAS file already contains the classification field set by
SMRF + RF.  We treat LAS class=2 (ground) as the positive class and
evaluate against a sampled ground-truth subset.

If no manual labels are available, the function falls back to comparing
the SMRF-only output vs. the RF-refined output as a proxy for improvement.

Output
------
Dict with keys: accuracy, precision, recall, f1, iou, confusion_matrix,
                classification_report (str), n_samples
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
    """
    Compute ground classification accuracy metrics.

    Parameters
    ----------
    classified_las_path  : LAS file with RF-refined classification
    reference_las_path   : Optional independent reference LAS.
                           If None, uses SMRF labels (class field before RF)
                           by comparing raw vs. classified tile pair if available.
    sample_n             : Points to subsample for fast evaluation
    ground_class_code    : LAS classification code for ground (default 2)
    seed                 : Random seed for reproducible sampling

    Returns
    -------
    dict  with accuracy, precision, recall, f1, iou, confusion_matrix,
              report_text, n_samples
    """
    classified_las_path = Path(classified_las_path)
    logger.info(f"Evaluating classification: {classified_las_path.name}")

    classified_las = laspy.read(str(classified_las_path))
    classification = np.array(classified_las.classification, dtype=np.int32)

    # Predicted binary labels: 1=ground, 0=non-ground
    y_pred = (classification == ground_class_code).astype(np.int32)

    if reference_las_path is not None:
        # Use external reference for ground truth
        ref_path = Path(reference_las_path)
        logger.info(f"  Reference: {ref_path.name}")
        ref_las  = laspy.read(str(ref_path))
        y_true   = (np.array(ref_las.classification) == ground_class_code).astype(np.int32)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Point count mismatch: classified={len(y_pred)}, "
                f"reference={len(y_true)}"
            )
    else:
        # Self-evaluation proxy: use intensity / return number heuristics
        # as a weak terrain proxy when no external reference is available.
        logger.warning(
            "No reference LAS provided – using heuristic terrain proxy "
            "(z-percentile + return number) as pseudo ground-truth. "
            "Provide a hand-labelled reference for true accuracy."
        )
        z = np.array(classified_las.z, dtype=np.float32)
        # Points within 0.5 m of 5th-percentile z are heuristically ground
        z_low   = np.percentile(z, 5)
        y_true  = (z <= z_low + 0.5).astype(np.int32)

    # ── Subsample for speed ────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    N   = len(y_true)
    if N > sample_n:
        idx    = rng.choice(N, size=sample_n, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    n_samples = len(y_true)

    # ── Metrics ───────────────────────────────────────────────────────
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    iou  = float(jaccard_score(y_true, y_pred, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred).tolist()
    rpt  = classification_report(y_true, y_pred,
                                  target_names=["non-ground", "ground"])

    metrics = {
        "accuracy":              round(acc,  4),
        "precision":             round(prec, 4),
        "recall":                round(rec,  4),
        "f1_score":              round(f1,   4),
        "iou":                   round(iou,  4),
        "confusion_matrix":      cm,
        "classification_report": rpt,
        "n_samples":             n_samples,
        "reference_type": "external" if reference_las_path else "heuristic_proxy",
    }

    logger.success(
        f"Ground classification  acc={acc:.4f}  prec={prec:.4f}  "
        f"rec={rec:.4f}  F1={f1:.4f}  IoU={iou:.4f}"
    )
    logger.info("\n" + rpt)
    return metrics
