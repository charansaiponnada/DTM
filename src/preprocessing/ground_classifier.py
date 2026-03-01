"""
src/preprocessing/ground_classifier.py
────────────────────────────────────────
Three-stage ground classification pipeline designed for the MoPR
Gujarat datasets which lack pre-existing ground labels and have
zero intensity variation.

Stage 1 – SMRF via PDAL  (fast, robust baseline)
Stage 2 – CSF  via PDAL  (fallback / cross-validation)
Stage 3 – ML refinement  (Random Forest on geometric features)
"""

from __future__ import annotations
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import os
import subprocess
import tempfile

import laspy
import numpy as np
from loguru import logger
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# ── PDAL subprocess helper (replaces Python pdal bindings) ───────────────
# PDAL Python bindings require conda on Windows; instead we call the PDAL
# CLI that ships with QGIS (available via pip-less install).
_QGIS_BIN = r"C:\Program Files\QGIS 3.40.15\bin"
_PDAL_EXE = os.path.join(_QGIS_BIN, "pdal.exe")


def _run_pdal_pipeline(pipeline_json: str) -> None:
    """
    Execute a PDAL pipeline expressed as a JSON string by writing it to a
    temporary file and calling the QGIS-bundled ``pdal.exe`` via subprocess.

    The QGIS bin directory is prepended to PATH so that pdal.exe can locate
    its shared libraries (gdal, proj, etc.).
    """
    if not os.path.isfile(_PDAL_EXE):
        raise FileNotFoundError(
            f"pdal.exe not found at {_PDAL_EXE}.\n"
            "Check _QGIS_BIN path in ground_classifier.py matches your QGIS install."
        )

    env = os.environ.copy()
    env["PATH"] = _QGIS_BIN + os.pathsep + env.get("PATH", "")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(pipeline_json)
        tmp_path = tf.name

    try:
        result = subprocess.run(
            [_PDAL_EXE, "pipeline", tmp_path],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PDAL pipeline failed (exit {result.returncode}):\n"
                f"{result.stderr.strip()}"
            )
    finally:
        os.unlink(tmp_path)


# ── Label constants (LAS classification codes) ───────────────────────────
CLASS_UNCLASSIFIED = 0
CLASS_GROUND       = 2
CLASS_LOW_VEG      = 3
CLASS_MED_VEG      = 4
CLASS_HIGH_VEG     = 5
CLASS_BUILDING     = 6


# ══════════════════════════════════════════════════════════════════════════
#  Stage 1 & 2 – PDAL-based Filters
# ══════════════════════════════════════════════════════════════════════════

def classify_smrf(
    input_path: str | Path,
    output_path: str | Path,
    slope: float = 0.15,
    window: float = 18.0,
    threshold: float = 0.5,
    scalar: float = 1.25,
) -> Path:
    """
    Run PDAL SMRF (Simple Morphological Filter) ground classification.

    SMRF works well on flat-to-moderately-sloped terrain such as the
    Gujarat abadi villages (largely flat, <5° slopes).
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_json = json.dumps({
        "pipeline": [
            str(input_path),
            {
                "type": "filters.smrf",
                "slope":     slope,
                "window":    window,
                "threshold": threshold,
                "scalar":    scalar,
                "ignore":    "Classification[7:7]",  # skip noise
            },
            {
                "type":     "filters.range",
                "limits":   "Classification[2:2]",   # keep ground pass-through
            },
            str(output_path),
        ]
    })

    # Full pipeline (all classes, labelled)
    pipeline_full = json.dumps({
        "pipeline": [
            str(input_path),
            {
                "type":      "filters.smrf",
                "slope":     slope,
                "window":    window,
                "threshold": threshold,
                "scalar":    scalar,
                "ignore":    "Classification[7:7]",
            },
            str(output_path),
        ]
    })

    logger.info(f"SMRF classifying {input_path.name} …")
    _run_pdal_pipeline(pipeline_full)
    logger.success(f"SMRF done → {output_path.name}")
    return output_path


def classify_csf(
    input_path: str | Path,
    output_path: str | Path,
    resolution: float = 0.5,
    rigidness: int = 3,
    class_threshold: float = 0.5,
) -> Path:
    """
    Cloth Simulation Filter – good for densely vegetated areas.
    Use as fallback or for cross-checking SMRF results.
    rigidness=3 is best for flat terrain (abadi).
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_json = json.dumps({
        "pipeline": [
            str(input_path),
            {
                "type":            "filters.csf",
                "resolution":      resolution,
                "rigidness":       rigidness,
                "threshold":       class_threshold,
                "cloth_resolution": resolution,
            },
            str(output_path),
        ]
    })
    logger.info(f"CSF classifying {input_path.name} …")
    _run_pdal_pipeline(pipeline_json)
    logger.success(f"CSF done → {output_path.name}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════
#  Stage 3 – Geometric Feature Extraction for ML
# ══════════════════════════════════════════════════════════════════════════

def compute_geometric_features(
    xyz: np.ndarray,
    k: int = 16,
    radius_density: float = 1.0,
    z_range_radius: float = 2.5,
) -> np.ndarray:
    """
    Compute per-point geometric features for ML-based classification.

    Features (12 columns – same order as original):
        0  eigenvalue_sum
        1  omnivariance
        2  eigenentropy
        3  anisotropy
        4  planarity
        5  linearity
        6  surface_variation (change_of_curvature)
        7  sphericity
        8  verticality
        9  density_1m
        10 z_range_neighbourhood
        11 height_above_ground_approx

    This implementation is fully vectorised using batched numpy einsum;
    no per-point Python loop, so it scales to millions of points.
    """
    N = len(xyz)
    logger.info(f"Computing geometric features for {N:,} points …")
    tree = cKDTree(xyz)

    features = np.zeros((N, 12), dtype=np.float32)
    BATCH = 50_000   # memory-safe chunk size
    eps   = 1e-10

    # ── PCA / covariance features (cols 0–8) — processed in batches ────
    for start in range(0, N, BATCH):
        end   = min(start + BATCH, N)
        bxyz  = xyz[start:end]                   # (B, 3)
        _, bidxs = tree.query(bxyz, k=k + 1)    # +1 = self
        bidxs = bidxs[:, 1:]                     # (B, k) – exclude self

        nbrs    = xyz[bidxs]                     # (B, k, 3)
        mu      = nbrs.mean(axis=1, keepdims=True)
        c       = nbrs - mu                      # (B, k, 3) centred
        covs    = np.einsum('bki,bkj->bij', c, c) / max(k - 1, 1)  # (B,3,3)

        eigvals, eigvecs = np.linalg.eigh(covs)  # ascending order, (B,3)
        eigvals = eigvals[:, ::-1]               # descending
        eigvals = np.clip(eigvals, eps, None)

        ev_sum  = eigvals.sum(axis=1, keepdims=True) + eps
        l       = eigvals / ev_sum               # normalised (B,3)
        l1, l2, l3 = l[:, 0], l[:, 1], l[:, 2]

        features[start:end, 0] = eigvals.sum(axis=1)
        features[start:end, 1] = np.cbrt(l1 * l2 * l3)
        features[start:end, 2] = -(
            l1*np.log(l1+eps) + l2*np.log(l2+eps) + l3*np.log(l3+eps))
        features[start:end, 3] = (l1 - l3) / (l1 + eps)
        features[start:end, 4] = (l2 - l3) / (l1 + eps)
        features[start:end, 5] = (l1 - l2) / (l1 + eps)
        features[start:end, 6] = l3 / (l1 + eps)
        features[start:end, 7] = l3 / (l1 + eps)
        # verticality: 1 − |z-component of the principal eigenvector|
        # eigh returns eigvecs as columns, ascending → last col = largest eigvec
        features[start:end, 8] = 1.0 - np.abs(eigvecs[:, 2, -1])

    # ── Density within radius_density (col 9) ─────────────────────
    counts = tree.query_ball_point(xyz, r=radius_density, return_length=True)
    features[:, 9] = np.array(counts, dtype=np.float32) / (np.pi * radius_density**2)

    # ── Z range in neighbourhood (col 10) ───────────────────────
    k_range = min(50, N)
    _, nbr_5m = tree.query(xyz, k=k_range)
    z_nbr = xyz[nbr_5m, 2]                      # (N, k_range)
    features[:, 10] = (z_nbr.max(axis=1) - z_nbr.min(axis=1)).astype(np.float32)

    # ── Height above ground (col 11) ───────────────────────────
    k_floor = min(100, N)
    _, nbr_floor = tree.query(xyz, k=k_floor)
    features[:, 11] = (xyz[:, 2] - xyz[nbr_floor, 2].min(axis=1)).astype(np.float32)

    logger.success("Geometric features computed.")
    return features


# ══════════════════════════════════════════════════════════════════════════
#  ML Refinement – Random Forest
# ══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "eigenvalue_sum", "omnivariance", "eigenentropy", "anisotropy",
    "planarity", "linearity", "surface_variation", "sphericity",
    "verticality", "density_1m", "z_range_nbr", "height_above_ground",
]


def train_rf_classifier(
    xyz: np.ndarray,
    labels: np.ndarray,           # 0=non-ground, 1=ground  (from SMRF)
    model_save_path: Optional[str | Path] = None,
    n_estimators: int = 200,
    max_depth: int = 15,
    n_jobs: int = -1,
    test_size: float = 0.2,
    max_samples: int = 200_000,   # cap training set to keep wall-time sane
) -> RandomForestClassifier:
    """
    Train a Random Forest on geometric features to refine SMRF labels.

    A random stratified subsample of at most ``max_samples`` points is
    used for training; the full cloud is classified via apply_rf_classifier.
    """
    logger.info("Extracting features for RF training …")
    # ── Subsample to keep training fast ───────────────────────────────
    N = len(xyz)
    if N > max_samples:
        rng        = np.random.default_rng(42)
        neg_idx    = np.where(labels == 0)[0]
        pos_idx    = np.where(labels == 1)[0]
        half       = max_samples // 2
        # cap per-class count to available points (tiles may be small)
        n_neg      = min(half, len(neg_idx))
        n_pos      = min(half, len(pos_idx))
        idx        = np.concatenate([
            rng.choice(neg_idx, size=n_neg, replace=False),
            rng.choice(pos_idx, size=n_pos, replace=False),
        ])
        xyz_tr    = xyz[idx]
        labels_tr = labels[idx]
        logger.info(f"Subsampled {N:,} → {max_samples:,} pts for RF training")
    else:
        xyz_tr, labels_tr = xyz, labels

    X = compute_geometric_features(xyz_tr)
    y = labels_tr.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    logger.info(
        f"Training RF on {len(X_train):,} pts "
        f"({y_train.sum():,} ground / {(y_train==0).sum():,} non-ground) …"
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    logger.info(
        "RF Validation Report:\n" +
        classification_report(y_test, y_pred, target_names=["non-ground", "ground"])
    )

    if model_save_path:
        model_save_path = Path(model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_save_path)
        logger.success(f"RF model saved → {model_save_path}")

    return clf


def apply_rf_classifier(
    las: laspy.LasData,
    model: RandomForestClassifier | str | Path,
    overwrite_classification: bool = True,
    max_apply_samples: int = 500_000,
) -> laspy.LasData:
    """
    Apply a trained RF model to re-classify a LasData object in place.

    For clouds with more than ``max_apply_samples`` points the RF is applied
    to a random spatial subsample; remaining points inherit the prediction of
    their nearest sampled neighbour (fast KNN propagation).
    """
    if isinstance(model, (str, Path)):
        model = joblib.load(model)

    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    N   = len(xyz)

    if N <= max_apply_samples:
        # Full cloud — compute features directly
        X     = compute_geometric_features(xyz)
        preds = model.predict(X).astype(np.int8)   # 1 = ground
    else:
        # Subsample → predict → propagate via KNN
        rng      = np.random.default_rng(0)
        samp_idx = rng.choice(N, size=max_apply_samples, replace=False)
        samp_idx.sort()
        X_samp   = compute_geometric_features(xyz[samp_idx])
        p_samp   = model.predict(X_samp).astype(np.int8)   # (S,)

        logger.info(
            f"KNN-propagating {max_apply_samples:,} RF predictions "
            f"to all {N:,} points \u2026"
        )
        # Build a small tree only on the sample to propagate labels
        samp_tree = cKDTree(xyz[samp_idx])
        _, nn_idx = samp_tree.query(xyz, k=1, workers=-1)  # (N,) indices into samp_idx
        preds = p_samp[nn_idx].astype(np.int8)   # (N,)

    if overwrite_classification:
        cls = np.where(preds == 1, CLASS_GROUND, CLASS_UNCLASSIFIED).astype(np.uint8)
        las.classification = cls
    else:
        # Only correct points already labelled as ground by SMRF
        existing   = np.array(las.classification)
        was_ground = existing == CLASS_GROUND
        new_cls    = existing.copy()
        new_cls[was_ground & (preds == 0)] = CLASS_UNCLASSIFIED
        las.classification = new_cls

    n_ground = int((np.array(las.classification) == CLASS_GROUND).sum())
    logger.info(
        f"RF reclassification done: {n_ground:,} / {N:,} points \u2192 ground"
    )
    return las


# ══════════════════════════════════════════════════════════════════════════
#  Convenience: Full Three-Stage Pipeline
# ══════════════════════════════════════════════════════════════════════════

def classify_ground_full_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    model_path: Optional[str | Path] = None,
    use_ml_refine: bool = True,
    smrf_kwargs: Optional[dict] = None,
) -> Tuple[Path, Optional[RandomForestClassifier]]:
    """
    Run full three-stage ground classification:
      1. SMRF via PDAL
      2. (optional) ML refinement with Random Forest
      3. Save classified LAS

    Returns
    -------
    (output_path, rf_model_or_None)
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    tmp_smrf    = output_path.parent / f"_smrf_tmp_{input_path.stem}.las"

    smrf_kwargs = smrf_kwargs or {}
    classify_smrf(input_path, tmp_smrf, **smrf_kwargs)

    rf_model = None
    if use_ml_refine:
        logger.info("Loading SMRF output for ML refinement …")
        las = laspy.read(tmp_smrf)
        xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        smrf_labels = (np.array(las.classification) == CLASS_GROUND).astype(int)

        if model_path and Path(model_path).exists():
            logger.info(f"Loading pre-trained RF model: {model_path}")
            rf_model = joblib.load(model_path)
        else:
            rf_model = train_rf_classifier(
                xyz, smrf_labels,
                model_save_path=model_path,
            )
        las = apply_rf_classifier(las, rf_model, overwrite_classification=False)
        las.write(str(output_path))
        tmp_smrf.unlink(missing_ok=True)
    else:
        tmp_smrf.rename(output_path)

    logger.success(f"Ground classification complete → {output_path}")
    return output_path, rf_model
