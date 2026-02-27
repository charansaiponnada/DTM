# Newbie Guide: What We Built, Why, and What Is Left

This file explains what has been implemented so far, how we approached the problem, and what remains to fully match the hackathon deliverables.

## 1) Approach We Followed

We used a **2-phase, automation-first approach**:

- Phase 1: Deterministic data processing to create training-ready artifacts from point cloud data
- Phase 2: ML risk prediction + drainage optimization on standardized outputs

Design choices used:

- Script/CLI first (fast and repeatable)
- Notebook only for analysis/debug (not execution engine)
- Lightweight config and logs (hackathon style, no heavy platform engineering)

## 2) What Has Been Implemented (Current Reality)

### Data processing pipeline

Implemented scripts:

- `scripts/info.py` → reusable LAS/LAZ preflight scanner
- `scripts/stages.py` → preflight, prepare, dtm, hydrology, features
- `scripts/pipeline.py` + `scripts/run_pipeline.py` → thin stage runner
- `pipeline_config.json` → single config source for path + core params

Generated outputs (already validated on Gujarat dataset):

- `outputs/reports/preflight_summary.json`
- `outputs/interim/prepared/*_prepared.npz`
- `outputs/interim/dtm/*_dtm.npz`
- `outputs/interim/hydrology/*_hydro.npz`
- `outputs/training_data/*_features.csv`
- `outputs/run_summary.txt`

### ML stage

Implemented script:

- `scripts/train_ml.py`

Generated outputs:

- `outputs/ml/metrics.json`
- `outputs/ml/predictions.csv`
- `outputs/ml/feature_importance.csv`
- `outputs/ml/rf_waterlogging_model.joblib`

Important label note:

- `label = -1` means unlabeled point
- Training with `--pseudo-label` is for bootstrap/demo only
- Final accuracy claims should use true labels (`0/1`)

### Optimization stage

Implemented script:

- `scripts/optimize_drainage.py`

Generated outputs:

- `outputs/optimization/proposed_drainage_lines.geojson`
- `outputs/optimization/hotspots.geojson`
- `outputs/optimization/outlets.geojson`
- `outputs/optimization/design_parameters.json`
- `outputs/optimization/optimization_summary.json`

### Documentation + traceability

Added docs:

- `docs/model_architecture.md`
- `docs/deployment_guidelines.md`
- `docs/final_report.md`
- `logs/activity_log.md` (timestamped success/failure trail)

## 3) Draft Plan vs Implementation Status

Below is the exact status against your drafted MVP plan.

1. Define execution contract in README/tasks/design  
   **Status:** Done (existing docs + updated run sections)

2. Single runtime config source  
   **Status:** Done (`pipeline_config.json`)

3. Refactor `info.py` for reusable checks  
   **Status:** Done

4. Stage 1 preprocessing/class filtering + QA  
   **Status:** Partially done (heuristic fallback; QA summary exists)

5. Stage 2 DTM generation with proper geospatial outputs (COG)  
   **Status:** Partially done (DTM npz generated; COG export not yet added)

6. Stage 3 hydrology with flow/accumulation + GPKG vectors  
   **Status:** Partially done (hydrology proxy npz done; formal GPKG outputs pending)

7. Training-data assembly for ML  
   **Status:** Done

8. Stage 4 ML + Stage 5 optimization  
   **Status:** Done (baseline versions)

9. Orchestrator with stage flags + run summary  
   **Status:** Done (lightweight summary instead of heavy manifests)

10. Analysis notebook (inspection only)  
    **Status:** Added (`analysis_notebook.ipynb`)

## 4) Clear TODO Order to Finish Strong

Follow this order strictly (finish one before moving next):

1. Replace heuristic ground extraction with PDAL CSF/SMRF classification.
2. Export DTM and derived layers as COG-compatible GeoTIFF.
3. Add hydrology stream/vector export to GPKG.
4. Replace pseudo-labels with true labels for final ML metrics.
5. Calibrate optimization parameters per village and freeze design values.
6. Regenerate final report metrics and submission package.

## 5) Run Commands You Need

Preflight:

```bash
python scripts/info.py Gujrat_Point_Cloud --json outputs/reports/preflight_from_info.json
```

Pipeline:

```bash
python scripts/run_pipeline.py --config pipeline_config.json --max-points 150000
```

ML:

```bash
python scripts/train_ml.py --features-dir outputs/training_data --output-dir outputs/ml --pseudo-label
```

Optimization:

```bash
python scripts/optimize_drainage.py --predictions outputs/ml/predictions.csv --features-dir outputs/training_data --output-dir outputs/optimization
```

## 6) What Failed Earlier and How It Was Fixed

- Failure: DTM stage produced 0 outputs on class-0-only files.
- Root cause: Dataset had classification field but no ground class (2).
- Fix: Added fallback ground selection from lower elevation quantile.
- Trace: See `logs/activity_log.md`.

---

If you are new: start from commands in section 5, then track remaining TODOs in section 4.
