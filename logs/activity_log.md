# Activity Log

This file tracks implementation progress with timestamps, outcomes, and notable failures.

## 2026-02-27

- 2026-02-27T10:20:00 | CREATED | [scripts/info.py](scripts/info.py) upgraded to CLI preflight scanner (file/folder input, JSON export).
- 2026-02-27T10:28:00 | CREATED | [scripts/stages.py](scripts/stages.py) with lightweight stages: preflight, prepare, dtm, hydrology, features.
- 2026-02-27T10:32:00 | CREATED | [scripts/pipeline.py](scripts/pipeline.py) thin runner with `--from-stage` and `--to-stage`.
- 2026-02-27T10:34:00 | CREATED | [pipeline_config.json](pipeline_config.json) minimal config for Gujarat-first run.
- 2026-02-27T10:37:00 | CREATED | [README.md](README.md) quickstart commands for pipeline usage.
- 2026-02-27T10:45:00 | FAILED | Initial full pipeline run produced `dtm outputs: 0`, then hydrology failed with missing DTM artifacts.
- 2026-02-27T10:50:00 | FIXED | [scripts/stages.py](scripts/stages.py) ground selection fallback added for class-0-only datasets.
- 2026-02-27T10:55:25 | SUCCESS | Full run completed with outputs summary in [outputs/run_summary.txt](outputs/run_summary.txt).
- 2026-02-27T11:10:00 | CREATED | [scripts/train_ml.py](scripts/train_ml.py) baseline RandomForest trainer with real-label and pseudo-label modes.
- 2026-02-27T11:12:00 | SUCCESS | ML run completed (`--pseudo-label`) with artifacts under [outputs/ml](outputs/ml).
- 2026-02-27T11:20:00 | CREATED | [logs/activity_log.md](logs/activity_log.md) for timestamped implementation history.
- 2026-02-27T11:30:00 | CREATED | [scripts/optimize_drainage.py](scripts/optimize_drainage.py) optimization stage for GIS-ready drainage design.
- 2026-02-27T11:32:00 | SUCCESS | Optimization run completed with 400 proposed drainage lines in [outputs/optimization/proposed_drainage_lines.geojson](outputs/optimization/proposed_drainage_lines.geojson).
- 2026-02-27T11:33:00 | CREATED | Deliverable docs added: [docs/model_architecture.md](docs/model_architecture.md), [docs/deployment_guidelines.md](docs/deployment_guidelines.md), [docs/final_report.md](docs/final_report.md).
- 2026-02-27T11:45:00 | CREATED | [newbie.md](newbie.md) explaining implementation approach, status, and step-by-step remaining TODOs.
- 2026-02-27T11:46:00 | CREATED | [final.md](final.md) with final expected deliverables and completion definition.
- 2026-02-27T11:47:00 | CREATED | [analysis_notebook.ipynb](analysis_notebook.ipynb) as analysis-only notebook (no pipeline execution ownership).
- 2026-02-27T11:48:00 | CREATED | [scripts/run_pipeline.py](scripts/run_pipeline.py) simple runner alias.
- 2026-02-27T11:49:00 | SUCCESS | Preflight run validated through `run_pipeline.py`.
- 2026-02-27T12:05:00 | CREATED | Dependency specification baseline created.
- 2026-02-27T12:07:00 | CREATED | [docs/documentation_protocol.md](docs/documentation_protocol.md) for mandatory step-by-step documentation discipline.
- 2026-02-27T12:09:00 | UPDATED | [README.md](README.md) environment setup workflow updated.
- 2026-02-27T12:30:00 | CREATED | [scripts/env_policy.py](scripts/env_policy.py) runtime environment policy enforcement added.
- 2026-02-27T12:34:00 | UPDATED | Stage 1 in [scripts/stages.py](scripts/stages.py) with SOR/ROR noise filtering and PDAL classification path with fallback.
- 2026-02-27T12:36:00 | UPDATED | [scripts/pipeline.py](scripts/pipeline.py) and [pipeline_config.json](pipeline_config.json) wired for classification/noise parameters.
- 2026-02-27T12:38:00 | UPDATED | Runtime environment enforcement added to CLI entries: [scripts/info.py](scripts/info.py), [scripts/run_pipeline.py](scripts/run_pipeline.py), [scripts/train_ml.py](scripts/train_ml.py), [scripts/optimize_drainage.py](scripts/optimize_drainage.py).
- 2026-02-27T12:42:00 | FAILED | Stage 1 noise filtering over-pruned sampled points (prepare output point_count became 0).
- 2026-02-27T12:44:00 | FIXED | Adaptive fallback added in [scripts/stages.py](scripts/stages.py) to prevent over-pruning from SOR/ROR combination.
- 2026-02-27T12:47:00 | SUCCESS | Stage 1 validated in isolated environment with non-zero prepared outputs and summary in [outputs/reports/prepare_summary.json](outputs/reports/prepare_summary.json).
- 2026-02-27T12:48:00 | INFO | PDAL classification path requested but Python PDAL module missing; prepare stage falls back to heuristic and records note in summary.
- 2026-02-27T13:02:00 | INFO | Native Python `pdal` install failed due missing `PDALConfig.cmake` SDK.
- 2026-02-27T13:06:00 | INFO | QGIS LTR installed and `C:\Program Files\QGIS 3.40.15\bin\pdal.exe` discovered.
- 2026-02-27T13:12:00 | FIXED | [scripts/stages.py](scripts/stages.py) updated to run PDAL SMRF on sampled points through CLI fallback when Python module is unavailable.
- 2026-02-27T13:15:00 | SUCCESS | Stage 1 prepare rerun completed with `classification_used = pdal_smrf` in [outputs/reports/prepare_summary.json](outputs/reports/prepare_summary.json).
- 2026-02-27T13:25:00 | UPDATED | Environment policy in [scripts/env_policy.py](scripts/env_policy.py) changed to isolated environment gating.
- 2026-02-27T13:27:00 | CREATED | [environment.yml](environment.yml) for Conda-first setup (`dataset-dtm`) including PDAL.
- 2026-02-27T13:29:00 | UPDATED | Conda-first setup docs in [README.md](README.md), [docs/deployment_guidelines.md](docs/deployment_guidelines.md), and [docs/documentation_protocol.md](docs/documentation_protocol.md).
