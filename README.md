# AI-Driven DTM and Drainage Optimization (PS2)

This repository is for **Problem Statement 2**:
- DTM creation from point cloud data
- waterlogging risk prediction
- drainage network design for village (abadi) areas

The project is implemented as a **script-first pipeline** with clear stage outputs.

## 1) Environment (Strict)

Use **workspace virtual environment only**.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run scripts with:

```bash
.venv\Scripts\python.exe <script>
```

## 2) Dataset

Input folder (current baseline):
- `Gujrat_Point_Cloud/`

Expected input file types:
- `.las`
- `.laz`

## 3) Pipeline Stages

1. **Preflight**: CRS, density, bounds, classes
2. **Prepare**: sampling + noise filtering + ground filtering (`laspy + scipy`, WhiteboxTools-assisted mode)
3. **DTM**: gridding and terrain base output
4. **Hydrology**: slope/depth/flow proxy baseline
5. **Features**: training-table export
6. **ML**: RandomForest risk baseline
7. **Optimization**: graph-like routing proposal

## 4) Core Commands

Preflight only:

```bash
.venv\Scripts\python.exe scripts\run_pipeline.py --config pipeline_config.json --from-stage preflight --to-stage preflight
```

Stage 1 (preflight + prepare):

```bash
.venv\Scripts\python.exe scripts\run_pipeline.py --config pipeline_config.json --from-stage preflight --to-stage prepare
```

Full data-prep baseline:

```bash
.venv\Scripts\python.exe scripts\run_pipeline.py --config pipeline_config.json
```

ML training:

```bash
.venv\Scripts\python.exe scripts\train_ml.py --features-dir outputs/training_data --output-dir outputs/ml --pseudo-label
```

Optimization:

```bash
.venv\Scripts\python.exe scripts\optimize_drainage.py --predictions outputs/ml/predictions.csv --features-dir outputs/training_data --output-dir outputs/optimization
```

## 5) Key Output Paths

- `outputs/reports/preflight_summary.json`
- `outputs/reports/prepare_summary.json`
- `outputs/interim/prepared/*_prepared.npz`
- `outputs/interim/dtm/*_dtm.npz`
- `outputs/interim/hydrology/*_hydro.npz`
- `outputs/training_data/*_features.csv`
- `outputs/ml/metrics.json`
- `outputs/ml/predictions.csv`
- `outputs/optimization/proposed_drainage_lines.geojson`
- `outputs/optimization/design_parameters.json`
- `outputs/optimization/optimization_summary.json`

## 6) Project Documents

- Execution and architecture: `design.md`
- Stage plan: `tasks.md`
- Role ownership: `roles.md`
- New contributor guide: `newbie.md`
- Final completion checklist: `final.md`
- Process protocol: `docs/documentation_protocol.md`
- Deployment/run guide: `docs/deployment_guidelines.md`
- Model notes: `docs/model_architecture.md`
- Final report template: `docs/final_report.md`
- Master PS2 reference: `docs/master.md`
- Activity history: `logs/activity_log.md`

## 7) Current Scope Note

Current implementation is a practical hackathon baseline. Some full PS2 target items (advanced hydrology outputs, field-grade validation, and richer design constraints) are still planned in later stages.
