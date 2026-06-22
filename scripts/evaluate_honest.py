"""
scripts/evaluate_honest.py
──────────────────────────
Recompute *defensible* accuracy metrics for processed villages, replacing the
earlier misleading / buggy evaluation:

  • DTM accuracy  → leave-out cross-validation of the IDW surface against
                    *withheld* LiDAR ground returns (real ASPRS-style vertical
                    accuracy), not a flat-plane self-comparison.
  • Waterlogging  → 5-fold CV fidelity of the XGBoost surrogate to the
                    physically-derived risk index (label/feature alignment bug
                    fixed) PLUS a true cross-village transfer matrix, which is
                    the honest generalisation signal.
  • Ground class. → honest descriptive stats (ground fraction, point counts).
                    No fabricated "accuracy" against a fake z-percentile proxy.

Writes data/output/_reports/honest_metrics.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import laspy
import rasterio

OUT = Path("data/output")
VILLAGES_DEFAULT = ["DEVDI", "KHAPRETA", "DHAL_HOSHIARPUR", "CHAKHIRASINGH", "DHUNDA"]
RNG = np.random.default_rng(42)


# ════════════════════════════════════════════════════════════════════════
#  DTM: leave-out cross-validation against withheld LiDAR ground returns
# ════════════════════════════════════════════════════════════════════════

def read_ground_points(las_path: Path, keep_per_chunk: int = 25_000,
                       chunk: int = 5_000_000, cap: int = 250_000):
    """Memory-safe capped sample of ground (class 2) points from a large LAS."""
    acc = []
    total = total_ground = 0
    with laspy.open(str(las_path)) as f:
        for pts in f.chunk_iterator(chunk):
            cls = np.asarray(pts.classification)
            total += len(cls)
            m = cls == 2
            ng = int(m.sum())
            total_ground += ng
            if ng:
                xyz = np.column_stack([
                    np.asarray(pts.x)[m], np.asarray(pts.y)[m], np.asarray(pts.z)[m]
                ]).astype(np.float64)
                if len(xyz) > keep_per_chunk:
                    xyz = xyz[RNG.choice(len(xyz), keep_per_chunk, replace=False)]
                acc.append(xyz)
    if not acc:
        return np.empty((0, 3)), total, total_ground
    allp = np.concatenate(acc)
    if len(allp) > cap:
        allp = allp[RNG.choice(len(allp), cap, replace=False)]
    return allp, total, total_ground


def idw_points(train_xyz, query_xy, power=2.0, k=12):
    """IDW estimate of z at query_xy from train points (KDTree)."""
    tree = cKDTree(train_xyz[:, :2])
    k = min(k, len(train_xyz))
    d, idx = tree.query(query_xy, k=k, workers=-1)
    if k == 1:
        d = d[:, None]; idx = idx[:, None]
    d = np.maximum(d, 1e-6)
    w = 1.0 / (d ** power)
    z = (w * train_xyz[idx, 2]).sum(1) / w.sum(1)
    return z


def evaluate_dtm_loocv(las_path: Path, folds: int = 5):
    """k-fold spatial holdout of the IDW interpolation vs real LiDAR z."""
    pts, total, total_ground = read_ground_points(las_path)
    if len(pts) < 5000:
        return {"error": "insufficient ground points", "n_ground_sampled": len(pts)}
    n = len(pts)
    order = RNG.permutation(n)
    pts = pts[order]
    res = np.array_split(np.arange(n), folds)
    all_resid = []
    for test_idx in res:
        mask = np.ones(n, bool); mask[test_idx] = False
        pred = idw_points(pts[mask], pts[test_idx, :2])
        all_resid.append(pred - pts[test_idx, 2])
    r = np.concatenate(all_resid)
    r = r[np.isfinite(r)]
    rmse = float(np.sqrt(np.mean(r ** 2)))
    mae = float(np.mean(np.abs(r)))
    bias = float(np.mean(r))
    nmad = float(1.4826 * np.median(np.abs(r - np.median(r))))
    le90 = float(np.percentile(np.abs(r), 90))
    le95 = float(np.percentile(np.abs(r), 95))
    return {
        "method": "leave_out_cv_vs_lidar_ground_returns",
        "rmse_m": round(rmse, 4), "mae_m": round(mae, 4),
        "bias_m": round(bias, 4), "nmad_m": round(nmad, 4),
        "le90_m": round(le90, 4), "le95_m": round(le95, 4),
        "n_check_points": int(len(r)), "cv_folds": folds,
        "total_points": int(total), "ground_points": int(total_ground),
        "ground_fraction": round(total_ground / max(total, 1), 4),
    }


# ════════════════════════════════════════════════════════════════════════
#  Waterlogging: fixed-alignment CV fidelity + cross-village transfer
# ════════════════════════════════════════════════════════════════════════

def build_village_xy(village: str):
    """Return aligned (X, y) for valid pixels of a village's risk surface."""
    from src.hydrology.waterlogging_predictor import (
        build_feature_stack, read_terrain_rasters,
        compute_depression_depth, generate_gold_standard_labels,
    )
    d = OUT / village
    dtm = d / "dtm.tif"
    twi = d / "twi.tif"
    acc = d / "flow_accumulation.tif"
    slope = d / "slope.tif"
    feats, valid, _ = build_feature_stack(dtm, twi, acc, slope)
    dem, twi_a, log_acc, slope_a, valid2, _, _ = read_terrain_rasters(dtm, twi, acc, slope)
    dep = compute_depression_depth(dtm, d, valid, dem)
    gold = generate_gold_standard_labels(
        dem=dem, valid_mask=valid, twi=twi_a,
        flow_accumulation=log_acc, slope=slope_a, depression_depth=dep,
    )
    # ── ALIGNMENT FIX: index features and labels with the SAME 2-D mask ──
    X = feats[valid]                 # (Nvalid, C)
    y = gold[valid].astype(int)      # (Nvalid,) — same pixels as X
    keep = y != -1
    X, y = X[keep], y[keep]
    # memory/time cap — 400k cells is ample for a stable AUC estimate
    if len(y) > 400_000:
        idx = RNG.choice(len(y), 400_000, replace=False)
        X, y = X[idx], y[idx]
    return np.ascontiguousarray(X, dtype=np.float32), y


def evaluate_waterlogging(village: str, folds: int = 5):
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

    X, y = build_village_xy(village)
    pos_rate = float(y.mean())
    if y.sum() < folds * 2 or (len(y) - y.sum()) < folds * 2:
        return {"error": "class imbalance too severe", "positive_rate": round(pos_rate, 4)}

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []
    for tr, va in skf.split(X, y):
        sc = RobustScaler()
        Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        m = xgb.XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.07,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5,
                              tree_method="hist", n_jobs=4, eval_metric="aucpr", random_state=42)
        m.fit(Xtr, y[tr])
        p = m.predict_proba(Xva)[:, 1]
        pred = (p >= 0.45).astype(int)
        rows.append(dict(
            roc_auc=roc_auc_score(y[va], p),
            pr_auc=average_precision_score(y[va], p),
            f1=f1_score(y[va], pred, zero_division=0),
            precision=precision_score(y[va], pred, zero_division=0),
            recall=recall_score(y[va], pred, zero_division=0),
        ))
    agg = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
    agg_std = {f"{k}_std": round(float(np.std([r[k] for r in rows])), 4) for k in rows[0]}
    agg.update(agg_std)
    agg["positive_rate"] = round(pos_rate, 4)
    agg["n_samples"] = int(len(y))
    return agg


def cross_village_transfer(villages):
    """Train surrogate on village A, predict B — honest generalisation signal."""
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import roc_auc_score
    data = {}
    for v in villages:
        try:
            Xv, yv = build_village_xy(v)
            if len(yv) > 150_000:          # subsample for memory during transfer
                idx = RNG.choice(len(yv), 150_000, replace=False)
                Xv, yv = Xv[idx], yv[idx]
            data[v] = (Xv, yv)
        except Exception as e:
            print(f"  [transfer] skip {v}: {e}")
    matrix = {}
    for a in data:
        Xa, ya = data[a]
        if ya.sum() < 10:
            continue
        sc = RobustScaler(); Xa_s = sc.fit_transform(Xa)
        m = xgb.XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.07,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5,
                              tree_method="hist", n_jobs=4, eval_metric="aucpr", random_state=42)
        m.fit(Xa_s, ya)
        matrix[a] = {}
        for b in data:
            Xb, yb = data[b]
            if len(np.unique(yb)) < 2:
                continue
            try:
                auc = roc_auc_score(yb, m.predict_proba(sc.transform(Xb))[:, 1])
                matrix[a][b] = round(float(auc), 4)
            except Exception:
                pass
    return matrix


def dtm_stats(village: str):
    from src.dtm.dtm_generator import get_dtm_stats
    p = OUT / village / "dtm.tif"
    return get_dtm_stats(p) if p.exists() else {}


def load_drainage(village: str):
    m = OUT / village / "metrics.json"
    if not m.exists():
        return {}
    try:
        return json.loads(m.read_text()).get("drainage", {})
    except Exception:
        return {}


def main():
    villages = sys.argv[1:] or [v for v in VILLAGES_DEFAULT if (OUT / v / "dtm.tif").exists()]
    print(f"Villages: {villages}")
    report = {"villages": {}, "_generated": time.strftime("%Y-%m-%d %H:%M")}

    for v in villages:
        print(f"\n{'='*60}\n{v}\n{'='*60}")
        rec = {}
        t0 = time.time()
        print("  DTM leave-out CV ...")
        try:
            rec["dtm"] = evaluate_dtm_loocv(OUT / v / "classified_ground.las")
            print(f"    RMSE={rec['dtm'].get('rmse_m')} m  MAE={rec['dtm'].get('mae_m')} m  "
                  f"LE90={rec['dtm'].get('le90_m')} m  n={rec['dtm'].get('n_check_points')}")
        except Exception as e:
            rec["dtm"] = {"error": str(e)}; print(f"    DTM failed: {e}")
        print("  Waterlogging CV ...")
        try:
            rec["waterlogging"] = evaluate_waterlogging(v)
            print(f"    AUC={rec['waterlogging'].get('roc_auc')}  PR-AUC={rec['waterlogging'].get('pr_auc')}  "
                  f"F1={rec['waterlogging'].get('f1')}  pos_rate={rec['waterlogging'].get('positive_rate')}")
        except Exception as e:
            rec["waterlogging"] = {"error": str(e)}; print(f"    WL failed: {e}")
        rec["dtm_stats"] = dtm_stats(v)
        rec["drainage"] = load_drainage(v)
        rec["_seconds"] = round(time.time() - t0, 1)
        report["villages"][v] = rec

    print(f"\n{'='*60}\nCross-village waterlogging transfer\n{'='*60}")
    try:
        report["cross_village_transfer"] = cross_village_transfer(villages)
        for a, row in report["cross_village_transfer"].items():
            print(f"  train {a:16s} -> " + "  ".join(f"{b[:6]}:{auc}" for b, auc in row.items()))
    except Exception as e:
        report["cross_village_transfer"] = {"error": str(e)}; print(f"  transfer failed: {e}")

    out = OUT / "_reports" / "honest_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[OK] wrote {out}")


if __name__ == "__main__":
    main()
