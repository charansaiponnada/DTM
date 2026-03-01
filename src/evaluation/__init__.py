"""
src/evaluation
──────────────
Accuracy and quality metrics for each pipeline stage.

Modules
-------
ground_classification_metrics  – F1, precision, recall, confusion matrix for SMRF+RF
dtm_metrics                    – RMSE/MAE vs reference DEM, vertical accuracy
waterlogging_metrics           – XGBoost CV scores, ROC-AUC, feature importance
drainage_metrics               – Channel coverage ratio, cost summary, hydraulic check
"""

from .ground_classification_metrics import evaluate_ground_classification
from .dtm_metrics import evaluate_dtm_accuracy
from .waterlogging_metrics import evaluate_waterlogging_model
from .drainage_metrics import evaluate_drainage_design

__all__ = [
    "evaluate_ground_classification",
    "evaluate_dtm_accuracy",
    "evaluate_waterlogging_model",
    "evaluate_drainage_design",
]
