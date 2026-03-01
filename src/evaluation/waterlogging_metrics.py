"""
src/evaluation/waterlogging_metrics.py
───────────────────────────────────────
Evaluate the XGBoost waterlogging prediction model.

Metrics reported
----------------
  Per-fold CV:  ROC-AUC, F1, precision, recall, average_precision (PR-AUC)
  Aggregate:    mean ± std for all CV metrics
  Feature importance: sorted permutation importances
  Calibration:  Brier score (probability accuracy)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger


def evaluate_waterlogging_model(
    predictor,                           # WaterloggingPredictor instance (already fitted)
    feature_stack: np.ndarray,           # (H, W, C) raster feature array
    labels: np.ndarray,                  # (H*W,) or (N,) binary labels
    valid_mask: Optional[np.ndarray] = None,  # (H, W) boolean mask
    cv_folds: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Evaluate a fitted WaterloggingPredictor using cross-validation.

    Parameters
    ----------
    predictor     : fitted WaterloggingPredictor with .model and .scaler
    feature_stack : (H, W, C) feature array (same as used in training)
    labels        : flat binary label array (1=waterlogging, 0=no risk)
    valid_mask    : (H, W) boolean; if None uses all pixels
    cv_folds      : number of stratified CV folds
    seed          : random seed

    Returns
    -------
    dict  with per_fold_metrics, mean_metrics, feature_importances,
              brier_score, threshold, n_samples
    """
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score, recall_score,
            average_precision_score, brier_score_loss,
            classification_report,
        )
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    H, W, C = feature_stack.shape

    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=bool)

    X_flat = feature_stack.reshape(-1, C)[valid_mask.ravel()]
    y_flat = labels.ravel()[:valid_mask.ravel().sum()] if labels.ndim == 2 else labels

    # Ensure label array matches feature rows
    if len(y_flat) != len(X_flat):
        y_flat = labels[valid_mask.ravel()] if labels.shape == valid_mask.shape \
                 else labels[:len(X_flat)]

    # Filter out nodata labels (-1) so only binary {0,1} classes remain
    labelled = y_flat != -1
    X_flat, y_flat = X_flat[labelled], y_flat[labelled]

    X_scaled = predictor.scaler.transform(X_flat)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    logger.info(f"Running {cv_folds}-fold cross-validation on {len(X_flat):,} samples …")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_scaled, y_flat), 1):
        X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val = y_flat[tr_idx],   y_flat[val_idx]

        # Re-train a clone on this fold (avoids data leakage)
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators     = predictor.model.n_estimators,
            max_depth        = predictor.model.max_depth,
            learning_rate    = predictor.model.learning_rate,
            scale_pos_weight = predictor.model.scale_pos_weight,
            tree_method      = "hist",
            random_state     = seed,
            eval_metric      = "logloss",
        )
        model.fit(X_tr, y_tr, verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= predictor.threshold).astype(int)

        fold_metrics.append({
            "fold":              fold,
            "roc_auc":           round(float(roc_auc_score(y_val, y_prob)), 4),
            "f1":                round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
            "precision":         round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
            "recall":            round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
            "avg_precision":     round(float(average_precision_score(y_val, y_prob)), 4),
        })
        logger.info(
            f"  Fold {fold}: AUC={fold_metrics[-1]['roc_auc']:.4f}  "
            f"F1={fold_metrics[-1]['f1']:.4f}  "
            f"Prec={fold_metrics[-1]['precision']:.4f}  "
            f"Rec={fold_metrics[-1]['recall']:.4f}"
        )

    # Aggregate
    keys_agg = ["roc_auc", "f1", "precision", "recall", "avg_precision"]
    mean_metrics = {
        k: round(float(np.mean([fm[k] for fm in fold_metrics])), 4)
        for k in keys_agg
    }
    std_metrics = {
        f"{k}_std": round(float(np.std([fm[k] for fm in fold_metrics])), 4)
        for k in keys_agg
    }
    mean_metrics.update(std_metrics)

    # Brier score on full scaled dataset
    y_prob_full = predictor.model.predict_proba(X_scaled)[:, 1]
    brier = float(brier_score_loss(y_flat, y_prob_full))

    # Feature importances from fitted model
    feat_names = [
        "elevation_norm", "slope", "aspect", "twi", "tpi",
        "log_flow_acc", "plan_curv", "profile_curv",
        "depression_depth", "stream_distance",
    ]
    importances = predictor.model.feature_importances_
    feat_imp = sorted(
        [{"feature": fn, "importance": round(float(imp), 4)}
         for fn, imp in zip(feat_names[:len(importances)], importances)],
        key=lambda x: x["importance"], reverse=True,
    )

    metrics = {
        "per_fold_metrics":   fold_metrics,
        "mean_metrics":       mean_metrics,
        "feature_importances": feat_imp,
        "brier_score":        round(brier, 4),
        "threshold":          predictor.threshold,
        "n_samples":          len(X_flat),
        "positive_rate":      round(float(y_flat.mean()), 4),
    }

    logger.success(
        f"Waterlogging model  mean AUC={mean_metrics['roc_auc']:.4f}  "
        f"F1={mean_metrics['f1']:.4f}  Brier={brier:.4f}"
    )
    return metrics
