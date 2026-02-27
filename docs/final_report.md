# Final Report Summary

## Problem Coverage

This project implements a data-driven workflow for:

- Automated preprocessing from drone point cloud to DTM-derived features
- Waterlogging hotspot prediction using ML baseline
- Optimized drainage network proposal as GIS-ready layers

## Key Outputs

- Preprocessing artifacts under `outputs/interim` and `outputs/training_data`
- ML outputs under `outputs/ml`
- Optimization outputs under `outputs/optimization`

## Accuracy and Metrics

- Metrics file: `outputs/ml/metrics.json`
- Current run uses pseudo-label mode for bootstrap experimentation.
- Real-field or curated labels are required for final scientific accuracy reporting.

## Design Recommendations

1. Replace pseudo labels with true waterlogging ground truth for robust evaluation.
2. Upgrade terrain preprocessing to PDAL-based CSF/SMRF classification.
3. Export final vectors to GeoPackage in addition to GeoJSON for submission packaging.
4. Calibrate risk thresholds village-wise for better drainage prioritization.

## Future Improvements

- Add temporal rainfall and land-use features.
- Add DEM/DTM uncertainty quantification.
- Compare Random Forest with Gradient Boosting and XGBoost.
