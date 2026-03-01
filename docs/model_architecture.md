# Model Architecture

## 1) Current Implemented Model

- Model type: `RandomForestClassifier`
- Script: `scripts/train_ml.py`
- Training source: `outputs/training_data/*_features.csv`

## 2) Input Features

Current features used:
- `elevation`
- `slope`
- `relative_depth`
- `flow_proxy`

## 3) Label Policy

- `label = -1` means unlabeled.
- If no real labels are present, pseudo-label mode can be used for pipeline bootstrap.
- Final scientific reporting should use real labels.

## 4) Outputs

- `outputs/ml/metrics.json`
- `outputs/ml/predictions.csv`
- `outputs/ml/feature_importance.csv`
- `outputs/ml/rf_waterlogging_model.joblib`

## 5) Future Upgrade Path

- Add real field labels.
- Add more hydrology-derived features.
- Compare RF with boosting models.
- Add strict validation split strategy per village.
