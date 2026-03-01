# Task Plan (Simple and Actionable)

## Stage 0 - Environment
- [ ] Create `.venv`
- [ ] Install `requirements.txt`
- [ ] Confirm scripts run with `.venv\Scripts\python.exe`

## Stage 1 - Data Validation and Preparation
- [ ] Run preflight
- [ ] Validate CRS, density, classes, bounds
- [ ] Run prepare
- [ ] Confirm noise-filtered and ground-filtered outputs
- [ ] Review `outputs/reports/prepare_summary.json`

## Stage 2 - DTM
- [ ] Generate DTM baseline files
- [ ] Check non-empty outputs in `outputs/interim/dtm`
- [ ] Confirm run summary updates

## Stage 3 - Hydrology
- [ ] Generate hydrology baseline files
- [ ] Verify `outputs/interim/hydrology`
- [ ] Validate expected dimensions/data ranges

## Stage 4 - Feature + ML
- [ ] Generate feature CSV files
- [ ] Train baseline RF model
- [ ] Export metrics, predictions, feature importance

## Stage 5 - Optimization
- [ ] Run drainage optimization script
- [ ] Validate GeoJSON output layers
- [ ] Validate design parameter and summary JSON

## Stage 6 - Documentation and Submission Readiness
- [ ] Update `logs/activity_log.md`
- [ ] Update `newbie.md` status
- [ ] Update `final.md` completion checklist
- [ ] Prepare final report in `docs/final_report.md`
