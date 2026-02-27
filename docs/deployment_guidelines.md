# Deployment Guidelines

## Environment

- Python 3.10+
- Install dependencies:

```bash
pip install laspy[lazrs] numpy scikit-learn joblib
```

## Run Sequence

1) Preprocess point-cloud data to training-ready features:

```bash
python scripts/pipeline.py --config pipeline_config.json --max-points 150000
```

2) Train baseline ML model:

```bash
python scripts/train_ml.py --features-dir outputs/training_data --output-dir outputs/ml --pseudo-label
```

3) Generate optimized drainage design layers:

```bash
python scripts/optimize_drainage.py --predictions outputs/ml/predictions.csv --features-dir outputs/training_data --output-dir outputs/optimization
```

## Multi-Dataset Use

- Update `input_dir` in `pipeline_config.json` to new village dataset path.
- Re-run the same 3-step sequence.
