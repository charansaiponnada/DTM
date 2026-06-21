"""Tests for waterlogging model evaluation (CV scaler fix)."""
import numpy as np
from src.evaluation.waterlogging_metrics import evaluate_waterlogging_model


def test_evaluate_waterlogging_model_basic():
    """Should return expected metric keys with a trivial classifier."""
    # Create a trivial predictor-like object
    class DummyPredictor:
        threshold = 0.5
        class Model:
            n_estimators = 10
            max_depth = 3
            learning_rate = 0.1
            scale_pos_weight = 1.0
            def fit(self, X, y, verbose=False):
                from xgboost import XGBClassifier
                self._real = XGBClassifier(n_estimators=2, max_depth=2, tree_method="hist")
                self._real.fit(X, y, verbose=False)
            def predict_proba(self, X):
                return self._real.predict_proba(X)
            @property
            def feature_importances_(self):
                return self._real.feature_importances_
        model = Model()
        scaler = None

    predictor = DummyPredictor()
    predictor.model = DummyPredictor.Model()
    predictor.scaler = type("Scaler", (), {"transform": lambda self, X: X})()

    np.random.seed(42)
    H, W, C = 10, 10, 4
    X = np.random.rand(H, W, C).astype(np.float32)
    y = np.random.randint(0, 2, (H, W)).astype(np.int8)
    y[0, 0] = -1  # nodata

    # Pre-fit the model on full data
    valid = y >= 0
    predictor.model.fit(X[valid].reshape(-1, C), y[valid].ravel())

    metrics = evaluate_waterlogging_model(predictor, X, y, cv_folds=2)
    assert "per_fold_metrics" in metrics
    assert "mean_metrics" in metrics
    assert "feature_importances" in metrics
    assert "brier_score" in metrics
    assert "roc_auc" in metrics["mean_metrics"]
    assert len(metrics["per_fold_metrics"]) == 2
