"""
src/hydrology/waterlogging_predictor.py
────────────────────────────────────────
XGBoost-based waterlogging hotspot predictor.

Feature engineering uses terrain derivatives from the DTM:
  - Elevation (normalized within village)
  - Slope, Aspect
  - Topographic Wetness Index (TWI)
  - Topographic Position Index (TPI)
  - Plan & Profile Curvature
  - Flow accumulation (log)
  - Depression depth
  - Distance to nearest stream channel

Model is trained on synthetic labels derived from terrain thresholds
(for initial run without historical flood data), and can be retrained
on real flood event shapefiles when available.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from scipy.ndimage import (
    generic_filter, uniform_filter, distance_transform_edt,
    label as ndlabel,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, average_precision_score
)
import xgboost as xgb
import joblib
from loguru import logger

from src.dtm.dtm_generator import NODATA, write_geotiff, convert_to_cog


# ══════════════════════════════════════════════════════════════════════════
#  Feature Raster Computation
# ══════════════════════════════════════════════════════════════════════════

def compute_tpi(dem: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Topographic Position Index = elevation − mean elevation in window.
    Negative TPI → valleys/hollows (waterlogging prone).
    """
    local_mean = uniform_filter(dem.astype(np.float64), size=window)
    return (dem - local_mean).astype(np.float32)


def compute_curvature(dem: np.ndarray, cell_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plan and Profile curvature (Evans method).

    Returns
    -------
    (plan_curv, profile_curv) arrays in 1/m units.
    Concave plan curvature → water convergence.
    """
    z = dem.astype(np.float64)
    cs = cell_size

    # 3x3 polynomial fit coefficients
    D = (np.roll(z, -1, axis=0) + np.roll(z, 1, axis=0) - 2*z) / (2 * cs**2)
    E = (np.roll(z, -1, axis=1) + np.roll(z, 1, axis=1) - 2*z) / (2 * cs**2)
    F = (-np.roll(np.roll(z,-1,0),-1,1) + np.roll(np.roll(z,-1,0),1,1)
         + np.roll(np.roll(z,1,0),-1,1) - np.roll(np.roll(z,1,0),1,1)) / (4*cs**2)
    G = (np.roll(z, 1, axis=1) - np.roll(z, -1, axis=1)) / (2 * cs)
    H = (np.roll(z, 1, axis=0) - np.roll(z, -1, axis=0)) / (2 * cs)

    p = G**2 + H**2 + 1e-10

    plan_curv    = (-2 * (D*G**2 + E*H**2 + F*G*H) / p).astype(np.float32)
    profile_curv = (-2 * (D*G**2 + E*H**2 + F*G*H) / (p * np.sqrt(p))).astype(np.float32)

    return plan_curv, profile_curv


def compute_aspect(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Aspect in degrees (0=N, clockwise)."""
    dy, dx = np.gradient(dem.astype(np.float64), cell_size)
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360
    return aspect.astype(np.float32)


def build_feature_stack(
    dtm_path: str | Path,
    twi_path: str | Path,
    flow_acc_path: str | Path,
    slope_path: str | Path,
    depression_depth_path: Optional[str | Path] = None,
    stream_distance_path:  Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray, rasterio.transform.Affine]:
    """
    Stack terrain feature rasters into (H, W, n_features) array.

    Returns
    -------
    feature_stack : (H, W, n_features) float32
    valid_mask    : (H, W) bool  – cells with complete data
    transform     : rasterio affine transform
    """
    logger.info("Building feature stack …")

    with rasterio.open(dtm_path) as src:
        dem       = src.read(1).astype(np.float32)
        transform = src.transform
        cell_size = src.res[0]
        nodata    = src.nodata or NODATA
        valid     = (dem != nodata) & np.isfinite(dem)

    with rasterio.open(twi_path) as src:
        twi = src.read(1).astype(np.float32)

    with rasterio.open(flow_acc_path) as src:
        log_acc = src.read(1).astype(np.float32)   # already log-scaled

    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float32)

    # Elevation normalized within the village extent
    elev_norm = np.where(
        valid,
        (dem - dem[valid].mean()) / (dem[valid].std() + 1e-6),
        0.0
    ).astype(np.float32)

    # Aspect
    aspect = compute_aspect(np.where(valid, dem, 0.0), cell_size)

    # TPI (15-cell window ≈ 7.5 m at 0.5 m res)
    tpi = compute_tpi(np.where(valid, dem, 0.0), window=15)

    # Curvature
    plan_curv, profile_curv = compute_curvature(np.where(valid, dem, 0.0), cell_size)

    # Depression depth (optional)
    if depression_depth_path and Path(depression_depth_path).exists():
        with rasterio.open(depression_depth_path) as src:
            dep_depth = src.read(1).astype(np.float32)
    else:
        # Approximate: filled DEM − DEM
        dep_depth = np.zeros_like(dem)

    # Distance to stream (optional)
    if stream_distance_path and Path(stream_distance_path).exists():
        with rasterio.open(stream_distance_path) as src:
            stream_dist = src.read(1).astype(np.float32)
    else:
        # Approximate: high flow accumulation → stream pixels
        stream_mask  = (log_acc > np.percentile(log_acc[valid], 90)).astype(np.uint8)
        stream_dist  = distance_transform_edt(~stream_mask.astype(bool)) * cell_size
        stream_dist  = stream_dist.astype(np.float32)

    feature_stack = np.stack([
        elev_norm,       # F0
        slope,           # F1
        aspect,          # F2
        twi,             # F3
        tpi,             # F4
        log_acc,         # F5
        plan_curv,       # F6
        profile_curv,    # F7
        dep_depth,       # F8
        stream_dist,     # F9
    ], axis=-1)          # → (H, W, 10)

    logger.success(
        f"Feature stack: {feature_stack.shape} "
        f"| valid cells: {valid.sum():,}"
    )
    return feature_stack, valid, transform


# ══════════════════════════════════════════════════════════════════════════
#  Label Generation (Terrain-Based Heuristic)
# ══════════════════════════════════════════════════════════════════════════

def generate_terrain_labels(
    feature_stack: np.ndarray,
    valid_mask: np.ndarray,
    twi_threshold: float     = 8.0,
    tpi_threshold: float     = -0.3,
    slope_threshold: float   = 2.0,
    acc_threshold_pct: float = 85.0,
) -> np.ndarray:
    """
    Generate pseudo-labels for waterlogging risk from terrain rules.

    A cell is labelled waterlogging-prone (1) if:
      • TWI  ≥ twi_threshold   (high wetness potential)         AND
      • TPI  ≤ tpi_threshold   (in a relative depression)       AND
      • slope ≤ slope_threshold (flat, water tends to pond)
      
    OR:
      • flow accumulation ≥ top acc_threshold_pct percentile

    This heuristic provides training labels for areas without
    historical flood records.

    Returns
    -------
    labels : (H, W) int8  (1 = waterlogging prone, 0 = safe)
    """
    elev_norm   = feature_stack[:, :, 0]
    slope       = feature_stack[:, :, 1]
    twi         = feature_stack[:, :, 3]
    tpi         = feature_stack[:, :, 4]
    log_acc     = feature_stack[:, :, 5]

    acc_thresh  = np.percentile(log_acc[valid_mask], acc_threshold_pct)

    # Primary rule: classic depression + high TWI
    rule_A = (twi >= twi_threshold) & (tpi <= tpi_threshold) & (slope <= slope_threshold)

    # Secondary rule: high flow accumulation (drains here)
    rule_B = log_acc >= acc_thresh

    labels = (rule_A | rule_B).astype(np.int8)
    labels[~valid_mask] = -1   # masked / nodata

    n_pos = (labels == 1).sum()
    n_tot = valid_mask.sum()
    logger.info(
        f"Terrain labels: {n_pos:,} / {n_tot:,} cells flagged as "
        f"waterlogging-prone ({100*n_pos/n_tot:.1f}%)"
    )
    return labels


# ══════════════════════════════════════════════════════════════════════════
#  XGBoost Waterlogging Model
# ══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "elevation_normalized", "slope_deg", "aspect_deg",
    "twi", "tpi", "log_flow_accumulation",
    "plan_curvature", "profile_curvature",
    "depression_depth_m", "distance_to_stream_m",
]


class WaterloggingPredictor:
    """
    XGBoost model predicting per-pixel waterlogging probability.

    Usage
    -----
    wp = WaterloggingPredictor()
    wp.fit(feature_stack, labels, valid_mask)
    prob_map = wp.predict_proba_map(feature_stack, valid_mask)
    wp.save("models/waterlogging_xgb.joblib")
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 5.0,
        threshold: float = 0.45,
    ):
        self.threshold   = threshold
        self.scaler      = RobustScaler()
        self.model       = xgb.XGBClassifier(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            learning_rate     = learning_rate,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            scale_pos_weight  = scale_pos_weight,
            eval_metric       = "aucpr",
            random_state      = 42,
            n_jobs            = -1,
            tree_method       = "hist",
        )
        self.is_fitted   = False

    def fit(
        self,
        feature_stack: np.ndarray,   # (H, W, F)
        labels: np.ndarray,           # (H, W)  int8, -1=nodata
        valid_mask: np.ndarray,       # (H, W)  bool
        cv_folds: int = 5,
    ) -> "WaterloggingPredictor":
        """
        Train model on flat (N, F) view of valid labelled pixels.
        """
        labeled_mask = valid_mask & (labels >= 0)
        X = feature_stack[labeled_mask]           # (N, F)
        y = labels[labeled_mask].astype(int)       # (N,)

        X = self.scaler.fit_transform(X)

        n_pos = y.sum()
        logger.info(
            f"Training XGBoost: {len(X):,} samples "
            f"({n_pos:,} positive / {len(X)-n_pos:,} negative)"
        )

        # Cross-validation – guard against insufficient class diversity
        n_classes   = len(np.unique(y))
        min_class_n = int(np.bincount(y).min()) if n_classes > 1 else 0
        safe_folds  = min(cv_folds, min_class_n) if min_class_n > 1 else 0

        if safe_folds < 2:
            logger.warning(
                f"Skipping CV: only {n_classes} class(es) present or "
                f"fewest class has {min_class_n} sample(s) "
                f"(need ≥ 2 per fold). Fitting directly."
            )
        else:
            actual_folds = safe_folds
            if actual_folds < cv_folds:
                logger.warning(
                    f"Reducing CV folds from {cv_folds} → {actual_folds} "
                    f"(minority class has only {min_class_n} samples)"
                )
            cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
            auc_scores = cross_val_score(
                self.model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
            )
            logger.info(
                f"CV ROC-AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}"
            )

        self.model.fit(X, y)
        self.is_fitted = True

        # Feature importances
        fi = pd.Series(
            self.model.feature_importances_,
            index=FEATURE_NAMES[:X.shape[1]],
        ).sort_values(ascending=False)
        logger.info("Feature importances:\n" + fi.to_string())

        return self

    def predict_proba_map(
        self,
        feature_stack: np.ndarray,  # (H, W, F)
        valid_mask: np.ndarray,     # (H, W) bool
    ) -> np.ndarray:
        """
        Return (H, W) float32 waterlogging probability map.
        Cells outside valid_mask are set to NODATA.
        """
        H, W, F = feature_stack.shape
        prob_map = np.full((H, W), NODATA, dtype=np.float32)

        X_valid = self.scaler.transform(feature_stack[valid_mask])
        probs   = self.model.predict_proba(X_valid)[:, 1].astype(np.float32)
        prob_map[valid_mask] = probs

        logger.info(
            f"Prediction: mean prob = {probs.mean():.3f}, "
            f"cells > {self.threshold}: {(probs >= self.threshold).sum():,}"
        )
        return prob_map

    def export_probability_cog(
        self,
        prob_map: np.ndarray,       # (H, W)
        transform: rasterio.transform.Affine,
        output_path: str | Path,
        crs: str = "EPSG:32643",
    ) -> Path:
        """Save probability map as Cloud-Optimized GeoTIFF."""
        output_path = Path(output_path)
        tmp = output_path.parent / "_wl_prob_tmp.tif"
        write_geotiff(prob_map, transform, tmp, crs=crs,
                      band_name="waterlogging_probability")
        return convert_to_cog(tmp, cog_path=output_path)

    def export_hotspot_gpkg(
        self,
        prob_map: np.ndarray,
        transform: rasterio.transform.Affine,
        output_gpkg: str | Path,
        crs: str = "EPSG:32643",
        layer: str = "waterlogging_hotspots",
    ) -> Path:
        """Vectorise high-probability areas → GeoPackage polygon layer."""
        output_gpkg = Path(output_gpkg)
        binary = ((prob_map >= self.threshold) & (prob_map != NODATA)).astype(np.uint8)

        crs_obj = rasterio.crs.CRS.from_epsg(int(crs.split(":")[-1]))
        geom_list = [
            {"geometry": g, "probability": float(v)}
            for g, v in shapes(
                prob_map * binary, mask=binary, transform=transform
            )
            if v >= self.threshold
        ]

        if not geom_list:
            logger.warning("No hotspot polygons found above threshold.")
            return output_gpkg

        gdf = gpd.GeoDataFrame(
            [{"probability": g["probability"], "risk_level": self._risk_label(g["probability"]),
              "geometry": shape(g["geometry"])} for g in geom_list],
            crs=crs_obj,
        )
        gdf["area_m2"] = gdf.geometry.area
        gdf = gdf[gdf["area_m2"] > 1.0]   # filter tiny slivers

        mode = "a" if output_gpkg.exists() else "w"
        gdf.to_file(str(output_gpkg), layer=layer, driver="GPKG")
        logger.success(f"Hotspot layer '{layer}' → {output_gpkg.name}: {len(gdf)} polygons")
        return output_gpkg

    @staticmethod
    def _risk_label(prob: float) -> str:
        if prob >= 0.75: return "HIGH"
        if prob >= 0.55: return "MEDIUM"
        return "LOW"

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "threshold": self.threshold}, path)
        logger.success(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "WaterloggingPredictor":
        data = joblib.load(path)
        obj = cls(threshold=data["threshold"])
        obj.model     = data["model"]
        obj.scaler    = data["scaler"]
        obj.is_fitted = True
        logger.info(f"Model loaded from {path}")
        return obj
