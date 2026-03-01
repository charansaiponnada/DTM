# Deployment Guidelines

## 1) Environment

This project uses **workspace `.venv` only**.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Pipeline Run Order

### Stage A: Preflight + Prepare

```bash
.venv\Scripts\python.exe scripts\run_pipeline.py --config pipeline_config.json --from-stage preflight --to-stage prepare
```

### Stage B: DTM + Hydrology + Features

```bash
.venv\Scripts\python.exe scripts\run_pipeline.py --config pipeline_config.json --from-stage dtm --to-stage features
```

### Stage C: ML

```bash
.venv\Scripts\python.exe scripts\train_ml.py --features-dir outputs/training_data --output-dir outputs/ml --pseudo-label
```

### Stage D: Optimization

```bash
.venv\Scripts\python.exe scripts\optimize_drainage.py --predictions outputs/ml/predictions.csv --features-dir outputs/training_data --output-dir outputs/optimization
```

## 3) Validation Checks

- Confirm `outputs/run_summary.txt` updated after each run.
- Confirm `outputs/reports/prepare_summary.json` has non-zero `point_count` and `ground_count`.
- Confirm ML outputs and optimization outputs are regenerated.

## 4) Multi-Dataset Reuse

- Change `input_dir` in `pipeline_config.json`.
- Re-run same command sequence.
- Keep output folder structure consistent for comparison.
