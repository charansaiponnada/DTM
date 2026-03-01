# First-Run Guide

This is the shortest path to run the project from scratch.

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Verify input files

Dataset folder:
- `Gujrat_Point_Cloud/`

Should contain `.las` / `.laz` files.

## 3) Run Stage 1

```bash
.venv\Scripts\python.exe scripts\run_pipeline.py --config pipeline_config.json --from-stage preflight --to-stage prepare
```

Check:
- `outputs/reports/preflight_summary.json`
- `outputs/reports/prepare_summary.json`

## 4) Run full baseline

```bash
.venv\Scripts\python.exe scripts\run_pipeline.py --config pipeline_config.json
.venv\Scripts\python.exe scripts\train_ml.py --features-dir outputs/training_data --output-dir outputs/ml --pseudo-label
.venv\Scripts\python.exe scripts\optimize_drainage.py --predictions outputs/ml/predictions.csv --features-dir outputs/training_data --output-dir outputs/optimization
```

## 5) Where to read next

- `README.md` for full workflow
- `newbie.md` for current status and what remains
- `docs/master.md` for full PS2 target specification
