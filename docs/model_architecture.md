# Model Architecture and Training Process

## Overview

The solution follows an automation-first pipeline:

1. Point cloud preflight and preparation
2. DTM generation from ground-like points
3. Hydrology feature derivation
4. ML-based waterlogging risk prediction
5. Drainage optimization proposal

## Current Baseline Model

- Model: Random Forest Classifier
- Input features:
  - `elevation`
  - `slope`
  - `relative_depth`
  - `flow_proxy`
- Outputs:
  - Risk probability per grid cell
  - Predicted class (`0` / `1`)
  - Feature importance

## Label Handling

- `label = -1` means unlabeled and not directly trainable.
- Training modes:
  - Real-label mode (recommended): uses true labels from field/GIS reference.
  - Pseudo-label mode (hackathon bootstrap): auto-generates labels from terrain-risk heuristic.

## Training Command

```bash
python scripts/train_ml.py --features-dir outputs/training_data --output-dir outputs/ml --pseudo-label
```

Replace pseudo-label mode with real labels once available.
