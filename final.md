# Final Expected Deliverables (Project Completion Checklist)

This file defines what must exist at completion, aligned to the organizer’s expected deliverables.

## A) Automated AI/ML Processing (Point-cloud to DTM)

### Required scripts

- `scripts/info.py` (preflight/validation)
- `scripts/run_pipeline.py` (single runner)
- `scripts/stages.py` (pipeline stages)
- `scripts/train_ml.py` (baseline model training)
- `scripts/optimize_drainage.py` (drainage design stage)

### Required processing outputs

- Point cloud quality/preflight report
- Ground-classified point cloud outputs
- DTM and key terrain derivatives
- Hydrology products (flow paths, low-lying zones)
- Training dataset for ML

## B) Optimized Drainage Network Design

### GIS-ready layers

- Proposed drainage network lines (`GeoJSON` and/or `GPKG`)
- Hotspot points/polygons
- Outlet points/polygons

### Design parameters and rationale

- Thresholds and constraints used
- Length/slope/elevation-drop assumptions
- Village-wise optimization summary

## C) Documentation (Model + Training + Deployment)

### Must include

- Model architecture and feature set
- Label strategy (true labels vs pseudo labels)
- Training procedure and evaluation metrics
- Deployment/run instructions for single village and multi-village
- Parameter configuration guidance

### Current documentation files

- `docs/model_architecture.md`
- `docs/deployment_guidelines.md`
- `docs/final_report.md`
- `README.md`

## D) Final Report

### Must include

- Objective coverage against problem statement
- Accuracy metrics (`accuracy`, `precision`, `recall`, `f1`, optional `roc_auc`)
- Optimization outcomes (proposed lines, average lengths, expected impact)
- Limitations and future improvements
- Recommendations for scale-up to 10 villages

## E) Submission-Ready Folder Expectation

At minimum, final package should contain:

- Source scripts
- Config file (`pipeline_config.json`)
- Key outputs under `outputs/`
- Documentation under `docs/`
- Activity log under `logs/activity_log.md`

---

## Completion Definition

Project is considered complete when:

1. End-to-end run succeeds from preflight to optimization.
2. Required GIS-ready drainage outputs are produced.
3. ML metrics are generated using valid labels for final reporting.
4. Final report and deployment docs are updated with latest results.
