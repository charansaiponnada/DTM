"""
src/evaluation/cross_village.py
────────────────────────────────
Cross-village evaluation: train on one village, evaluate on another.
Shows model generalisation across different terrain and point-density regimes.

Output
------
Table of per-fold metrics showing how well ground classification and
waterlogging models transfer between villages.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger


def run_cross_village_evaluation(
    train_las_path: str | Path,
    test_las_paths: List[str | Path],
    train_name: str = "train",
    sample_n: int = 50_000,
    random_seed: int = 42,
) -> Dict:
    """
    Train Random Forest on one village's point cloud, then evaluate
    against z-percentile heuristic on multiple test villages.

    This provides an upper-bound on classification transferability.

    Returns
    -------
    dict with per-test-village accuracy, F1, etc.
    """
    import laspy
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from src.preprocessing.ground_classifier import compute_geometric_features

    rng = np.random.default_rng(random_seed)
    results = {}

    # ── Load & sample training data ────────────────────────────────────
    logger.info(f"Loading training village: {train_name} ({train_las_path})")
    train_las = laspy.read(str(train_las_path))
    train_xyz = np.column_stack([train_las.x, train_las.y, train_las.z]).astype(np.float64)
    train_z = np.array(train_las.z, dtype=np.float32)
    z_low = np.percentile(train_z, 5)
    train_labels = (train_z <= z_low + 0.5).astype(int)

    N_train = len(train_xyz)
    half = min(100_000, N_train // 2)
    if half < 1000:
        logger.warning(f"Training village too small ({N_train} pts). Skipping cross-eval.")
        return {"error": "insufficient training points", "n_train": N_train}

    pos_idx = np.where(train_labels == 1)[0]
    neg_idx = np.where(train_labels == 0)[0]
    n_pos = min(half, len(pos_idx))
    n_neg = min(half, len(neg_idx))
    if n_pos < 100 or n_neg < 100:
        return {"error": "insufficient class balance", "n_pos": len(pos_idx), "n_neg": len(neg_idx)}

    idx = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ])
    rng.shuffle(idx)

    X_train = compute_geometric_features(train_xyz[idx])
    y_train = train_labels[idx]

    # ── Train RF ───────────────────────────────────────────────────────
    logger.info(f"Training RF on {len(X_train):,} points from {train_name} …")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1,
        class_weight="balanced", random_state=random_seed,
    )
    rf.fit(X_train, y_train)
    logger.success("RF trained.")

    # ── Evaluate on each test village ──────────────────────────────────
    for test_path in test_las_paths:
        test_path = Path(test_path)
        test_name = test_path.stem
        logger.info(f"  Evaluating on {test_name} …")

        test_las = laspy.read(str(test_path))
        test_xyz = np.column_stack([test_las.x, test_las.y, test_las.z]).astype(np.float64)
        test_z = np.array(test_las.z, dtype=np.float32)

        N_test = len(test_xyz)
        if N_test > sample_n:
            samp_idx = rng.choice(N_test, size=sample_n, replace=False)
            test_xyz = test_xyz[samp_idx]
            test_z = test_z[samp_idx]

        z_low_test = np.percentile(test_z, 5)
        y_true_test = (test_z <= z_low_test + 0.5).astype(int)

        X_test = compute_geometric_features(test_xyz)
        y_pred_test = rf.predict(X_test)

        results[test_name] = {
            "accuracy":  round(float(accuracy_score(y_true_test, y_pred_test)), 4),
            "f1_score":  round(float(f1_score(y_true_test, y_pred_test, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true_test, y_pred_test, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true_test, y_pred_test, zero_division=0)), 4),
            "n_train":   len(X_train),
            "n_test":    len(X_test),
        }

    # ── Print table ────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"CROSS-VILLAGE EVALUATION  (train={train_name})")
    logger.info(f"{'Test Village':<25} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    logger.info("-" * 70)
    for test_name, m in results.items():
        if isinstance(m, dict) and "accuracy" in m:
            logger.info(f"{test_name:<25} {m['accuracy']:>8.4f} {m['f1_score']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}")
    logger.info("=" * 70)

    return {"train_village": train_name, "results": results}
