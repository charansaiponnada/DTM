# DTM Drainage AI

AI/ML pipeline for Digital Terrain Modelling and optimized drainage network design from drone LiDAR point clouds. Built for the MoPR Geospatial Intelligence Hackathon (IIT Tirupati).

## Pipeline Overview

```
LAS/LAZ  ──► Stage 1: Data Inspection ──► Stage 2: Ground Classification ──► Stage 3: DTM Generation ──► Stage 4: Hydrology ──► Stage 5: Waterlogging ──► Stage 6: Drainage Design
              laspy / numpy               PDAL SMRF + scikit-learn RF       IDW + rio-cogeo              pysheds                  XGBoost                  networkx + Manning's
```

| Stage | What it does | Tools | Output |
|-------|-------------|-------|--------|
| 1 | Inspect LAS header, sample points, spatial tiling | `laspy`, `numpy` | `PointCloudMetadata` + tile index |
| 2 | SMRF ground filter + optional Random Forest refinement | `pdal.exe`, `scikit-learn` | `classified_ground.las` |
| 3 | IDW interpolation → COG DTM + terrain derivatives | `scipy.spatial.cKDTree`, `rio-cogeo` | `dtm.tif`, `slope.tif`, curvature, TPI |
| 4 | Fill depressions, D8 flow, accumulation, TWI, stream extraction | `pysheds`, `geopandas` | `flow_direction.tif`, `twi.tif`, stream GPKG |
| 5 | 10-feature XGBoost waterlogging risk model | `xgboost`, `scikit-learn` | `waterlogging_probability.tif`, hotspots GPKG |
| 6 | MST channel optimization + Manning's hydraulic sizing | `networkx`, `geopandas` | Designed `drainage_network.gpkg` with cost |

## Quick Start

### Prerequisites

- Python 3.12+
- PDAL (for SMRF ground classification)
  ```bash
  conda install -c conda-forge pdal
  ```

### Setup

```bash
# Clone
git clone https://github.com/your-org/dtm-drainage-ai
cd dtm-drainage-ai

# Create environment
uv venv
uv pip install -r requirements.txt

# Install dev/test deps (optional)
uv pip install -r requirements_dev.txt
```

### Run

```bash
# Single village
python run_pipeline.py --input data/input/DEVDI_511671.las

# With evaluation
python run_pipeline.py --input data/input/DEVDI_511671.las --evaluate

# Skip ML refinement (faster)
python run_pipeline.py --input data/input/DEVDI_511671.las --no-ml

# Process multiple villages from config
python run_pipeline.py --batch

# Select specific stages (e.g. skip classification, re-run hydrology)
python run_pipeline.py --input data/input/DEVDI_511671.las --stages 3,4,5,6

# Windows batch script
run_pipeline.bat
```

### Python API

```python
from src.pipeline import DTMDrainagePipeline

pipeline = DTMDrainagePipeline(
    config_path="config/config.yaml",
    input_las="data/input/DEVDI_511671.las",
    output_dir="data/output",
)
results = pipeline.run(
    use_ml_refine=True,
    stream_threshold=1000,
)
```

## Repository Structure

```
dtm-drainage-ai/
├── run_pipeline.py              # CLI entry point (click)
├── run_pipeline.bat             # Windows batch runner
├── install.bat                  # Windows environment setup
├── setup.py                     # pip installable package
├── pyproject.toml               # Project metadata + pytest config
├── requirements.txt             # Core dependencies
├── requirements_dev.txt         # Dev/test dependencies
│
├── src/                         # Python package
│   ├── pipeline.py              # DTMDrainagePipeline + BatchPipelineRunner
│   ├── logger.py                # Rich console + structured logging
│   ├── features.py              # Shared terrain feature computation
│   ├── preprocessing/           # LAS I/O, SMRF, geometric features
│   ├── dtm/                     # IDW interpolation, COG, derivatives
│   ├── hydrology/               # Flow analysis, waterlogging, drainage
│   └── evaluation/              # Accuracy metrics for all stages
│
├── app/                         # Streamlit web application
│   ├── app.py
│   └── README.md
│
├── scripts/                     # Standalone utility scripts
│   ├── generate_docs.py         # PowerPoint/Word report generation
│   ├── generate_images.py       # Visualisation images
│   ├── create_word_doc.py       # Hackathon report
│   └── generate_pipeline_diagram.py  # Architecture diagram
│
├── config/
│   └── config.yaml              # All tunable parameters
│
├── tests/
│   ├── test_cli.py              # CLI smoke test
│   └── test_features.py         # Terrain feature unit tests
│
├── data/
│   ├── input/                   # LAS/LAZ point clouds
│   └── output/                  # Pipeline results (COG, GPKG, LAS)
│
├── docs/                        # Documentation, reports, images
├── notebooks/                   # Jupyter exploration
└── logs/                        # Structured JSONL logs
```

## Output Files

All outputs conform to OGC standards.

| File | Format | Description |
|------|--------|-------------|
| `dtm.tif` | Cloud-Optimized GeoTIFF | Digital Terrain Model @ 0.5 m |
| `slope.tif` | COG | Slope in degrees |
| `aspect.tif` | COG | Aspect in degrees |
| `twi.tif` | COG | Topographic Wetness Index |
| `flow_direction.tif` | COG | D8 direction codes |
| `flow_accumulation.tif` | COG | Log-scaled accumulation |
| `plan_curvature.tif` | COG | Plan curvature (Evans) |
| `profile_curvature.tif` | COG | Profile curvature (Evans) |
| `tpi_*.tif` | COG | Topographic Position Index |
| `roughness.tif` | COG | Terrain roughness |
| `hillshade.tif` | COG | Hillshade relief |
| `waterlogging_probability.tif` | COG | Risk probability 0–1 |
| `classified_ground.las` | LAS 1.4 | Ground-classified point cloud |
| `drainage_network.gpkg` | GeoPackage | All vector layers |

### GeoPackage Layers

| Layer | Type | Description |
|-------|------|-------------|
| `drainage_channels` | LineString | Designed segments with hydraulic specs |
| `waterlogging_hotspots` | Polygon | LOW / MEDIUM / HIGH risk zones |
| `depression_polygons` | Polygon | Topographic sinks |
| `catchment_boundaries` | Polygon | Sub-catchment areas |
| `design_summary` | Point | Aggregated design statistics |

## Models

- **SMRF** (PDAL): Morphological ground filter for flat terrain
- **Random Forest** (scikit-learn): 12 PCA-based geometric features, refines SMRF
- **XGBoost**: 10 terrain features, rule-based pseudo-labels, 5-fold CV, AUC-PR metric
- **MST** (NetworkX): Minimum-cost channel routing
- **Manning's Equation**: Trapezoidal channel hydraulic sizing

## Configuration

All parameters in `config/config.yaml`. Key settings:

```yaml
dtm:
  resolution: 0.5              # metres
  interpolation:
    idw_power: 2
    idw_radius: 5.0

drainage:
  design_return_period: 10     # years
  rainfall_intensity: 50       # mm/hr
  runoff_coefficient: 0.65
  cost_per_metre_channel: 1200 # INR

waterlogging:
  model: xgboost
  threshold: 0.45
  xgboost:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.1
    scale_pos_weight: 5
```

## Data Sources

Point cloud data from [SVAMITVA Portal](https://svamitva.nic.in) (CRS: EPSG:32643 for Gujarat).

## Tech Stack

| Domain | Libraries |
|--------|-----------|
| Point cloud I/O | `laspy` |
| Ground classification | `pdal`, `scikit-learn` |
| Raster processing | `rasterio`, `rio-cogeo`, `scipy` |
| Hydrology | `pysheds` |
| ML | `xgboost`, `scikit-learn` |
| Graph optimization | `networkx` |
| GIS/Vector | `geopandas`, `shapely`, `pyproj` |
| CLI | `click` |
| Logging/UI | `loguru`, `rich`, `tqdm` |

## License

MIT
