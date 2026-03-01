# Newbie Guide (Current Status)

## What this project does

It transforms point cloud data into:
1. terrain-ready outputs
2. ML-ready feature data
3. waterlogging risk predictions
4. drainage optimization proposals

## How to run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then run:

```bash
.venv\Scripts\python.exe scripts\run_pipeline.py --config pipeline_config.json
.venv\Scripts\python.exe scripts\train_ml.py --features-dir outputs/training_data --output-dir outputs/ml --pseudo-label
.venv\Scripts\python.exe scripts\optimize_drainage.py --predictions outputs/ml/predictions.csv --features-dir outputs/training_data --output-dir outputs/optimization
```

## What is already implemented

- Stage runner and stage modules
- Preflight + prepare + dtm + hydrology + features baseline
- ML baseline (RandomForest)
- Optimization baseline
- Structured output folders and reports

## What still needs improvement

- richer geospatial export formats (later phase)
- stronger hydrology rigor and validation
- real labels for final ML claims
- full PS2 metric evidence package

## Most important files

- `README.md`
- `tasks.md`
- `design.md`
- `final.md`
- `docs/master.md`
- `logs/activity_log.md`
