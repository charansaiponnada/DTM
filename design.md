# System Design

## Objective

Build an automated pipeline that converts village point cloud data into:
- terrain products (DTM and derivatives)
- waterlogging risk predictions
- drainage network design suggestions

## Architecture (Implemented Baseline)

1. Input point cloud (`.las/.laz`)
2. Preflight validation
3. Stage-1 preparation:
   - sampling
   - noise filtering (SOR/ROR)
   - ground filtering (`laspy + scipy`, WhiteboxTools-assisted mode)
4. DTM grid generation
5. Hydrology baseline generation
6. Feature table generation
7. ML risk prediction (RandomForest)
8. Drainage optimization proposal

## Design Principles

- Script-first, reproducible pipeline
- Strict `.venv` execution policy
- Stage outputs written to `outputs/`
- Lightweight implementation first, extensible later

## Data Contracts

### Input
- Folder containing `.las`/`.laz`
- CRS expected: EPSG:32643 (current Gujarat baseline)

### Output
- Reports: JSON summaries
- Intermediates: NPZ artifacts
- ML: metrics/predictions/model file
- Optimization: GeoJSON + parameter summaries

## Stage Ownership

- Preflight + Prepare + DTM: Terrain pipeline
- Hydrology + features: Terrain intelligence
- ML: Risk modeling
- Optimization: Drainage proposal

## Non-Goals in Current Baseline

- Deep point-cloud model training (RandLA-Net/PointNet++)
- Full hydraulic simulation
- Full field-validation metric suite

These are planned as next-phase upgrades.
