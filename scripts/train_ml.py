from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["elevation", "slope", "relative_depth", "flow_proxy"]


def _read_feature_files(features_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows: list[dict] = []
    x_data: list[list[float]] = []
    y_data: list[int] = []

    files = sorted(features_dir.glob("*_features.csv"))
    if not files:
        raise FileNotFoundError(f"No feature CSV files found in {features_dir}")

    for file_path in files:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for item in reader:
                row = {
                    "file": item["file"],
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                }
                feature_values = [float(item[col]) for col in FEATURE_COLUMNS]
                label = int(float(item["label"]))

                rows.append(row)
                x_data.append(feature_values)
                y_data.append(label)

    return rows, np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.int32)


def _make_pseudo_labels(features: np.ndarray, positive_quantile: float) -> np.ndarray:
    relative_depth = features[:, 2]
    flow_proxy = features[:, 3]
    slope = features[:, 1]

    depth_norm = (relative_depth - relative_depth.min()) / (relative_depth.max() - relative_depth.min() + 1e-6)
    flow_norm = (flow_proxy - flow_proxy.min()) / (flow_proxy.max() - flow_proxy.min() + 1e-6)
    slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)

    risk_score = 0.5 * depth_norm + 0.4 * flow_norm + 0.1 * (1.0 - slope_norm)
    threshold = np.quantile(risk_score, positive_quantile)
    return (risk_score >= threshold).astype(np.int32)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    unique_classes = np.unique(y_true)
    if unique_classes.shape[0] == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def _save_model(model: RandomForestClassifier, output_path: Path) -> str:
    try:
        import joblib

        joblib.dump(model, output_path)
        return "joblib"
    except Exception:
        with output_path.with_suffix(".pkl").open("wb") as f:
            pickle.dump(model, f)
        return "pickle"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RandomForest waterlogging model from feature CSV files")
    parser.add_argument("--features-dir", default="outputs/training_data")
    parser.add_argument("--output-dir", default="outputs/ml")
    parser.add_argument("--pseudo-label", action="store_true", help="Generate labels if all labels are -1")
    parser.add_argument("--positive-quantile", type=float, default=0.8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, x_all, y_all = _read_feature_files(features_dir)
    labeled_mask = y_all >= 0

    label_mode = "human"
    if np.count_nonzero(labeled_mask) == 0:
        if not args.pseudo_label:
            raise RuntimeError(
                "No training labels found (all labels are -1). "
                "Provide labels in feature CSV or run with --pseudo-label for hackathon demo."
            )
        y_all = _make_pseudo_labels(x_all, positive_quantile=args.positive_quantile)
        labeled_mask = np.ones_like(y_all, dtype=bool)
        label_mode = "pseudo"

    x_train_source = x_all[labeled_mask]
    y_train_source = y_all[labeled_mask]

    if np.unique(y_train_source).shape[0] < 2:
        raise RuntimeError("Training labels must contain both classes 0 and 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        x_train_source,
        y_train_source,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_train_source,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    metrics = _compute_metrics(y_test, y_pred, y_prob)
    metrics.update(
        {
            "label_mode": label_mode,
            "train_samples": int(x_train.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "feature_columns": FEATURE_COLUMNS,
        }
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_path = output_dir / "rf_waterlogging_model.joblib"
    model_format = _save_model(model, model_path)

    predictions = model.predict_proba(x_all)[:, 1]
    predicted_labels = (predictions >= 0.5).astype(np.int32)

    pred_path = output_dir / "predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "x", "y", "risk_probability", "predicted_label"])
        for row, prob, pred in zip(rows, predictions, predicted_labels):
            writer.writerow([row["file"], f"{row['x']:.3f}", f"{row['y']:.3f}", f"{prob:.6f}", int(pred)])

    importance_path = output_dir / "feature_importance.csv"
    with importance_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for name, score in zip(FEATURE_COLUMNS, model.feature_importances_):
            writer.writerow([name, f"{float(score):.8f}"])

    print(f"Label mode: {label_mode}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Model saved ({model_format}): {model_path if model_format == 'joblib' else model_path.with_suffix('.pkl')}")
    print(f"Predictions saved: {pred_path}")
    print(f"Feature importance saved: {importance_path}")


if __name__ == "__main__":
    main()