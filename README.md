# DTM Drainage AI 🛰️🌊
### MoPR Geospatial Intelligence Hackathon — IIT Tirupati

> **AI/ML-driven Digital Terrain Modelling and Optimized Drainage Network Design  
> for Flood-Prone Village Abadi Areas using Drone Point Cloud Data**

---

## 🏆 Problem Statement

Leverage drone-captured LiDAR/point cloud data from SVAMITVA villages to:

1. **Delineate** surface-water flow paths and low-lying flood zones  
2. **Predict** waterlogging hotspots using terrain-derived ML features  
3. **Design** an optimized, cost-effective drainage system for densely populated abadi areas

---

## 📁 Repository Structure

```
dtm-drainage-ai/
├── config/
│   └── config.yaml              # All tunable parameters
├── src/
│   ├── preprocessing/
│   │   ├── point_cloud_loader.py    # LAS/LAZ I/O, tiling for large files
│   │   └── ground_classifier.py     # SMRF → CSF → Random Forest pipeline
│   ├── dtm/
│   │   └── dtm_generator.py         # IDW interpolation → COG export
│   ├── hydrology/
│   │   ├── flow_analysis.py         # Fill → Flow Dir → Accumulation → TWI
│   │   ├── waterlogging_predictor.py # XGBoost waterlogging risk model
│   │   └── drainage_network.py      # MST + Manning's hydraulic design
│   └── ml/
│       └── pointnet_classifier.py   # PointNet deep learning classifier
├── pipelines/
│   └── full_pipeline.py         # End-to-end orchestrator + CLI
├── requirements.txt
└── config/config.yaml
```

---

## ⚙️ Installation

```bash
# 1. Clone
git clone https://github.com/your-username/dtm-drainage-ai
cd dtm-drainage-ai

# 2. Create environment
conda create -n dtm-ai python=3.11 -y
conda activate dtm-ai

# 3. Install PDAL (best via conda for native binaries)
conda install -c conda-forge pdal python-pdal -y

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. (Optional) GPU support for PointNet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Quick Start

### Run the full pipeline via CLI

```bash
python pipelines/full_pipeline.py \
  --input  data/input/DEVDI_POINT_CLOUD_511671.las \
  --output data/output/devdi/ \
  --config config/config.yaml
```

### Or use the Python API

```python
from pipelines.full_pipeline import DTMDrainagePipeline

pipeline = DTMDrainagePipeline(
    config_path = "config/config.yaml",
    input_las   = "data/input/DEVDI_POINT_CLOUD_511671.las",
    output_dir  = "data/output/devdi/",
)
results = pipeline.run()
```

---

## 🔬 Pipeline Architecture

```
LAS/LAZ Input
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — Data Inspection                                  │
│  • Point count, density, CRS, intensity range               │
│  • Flags: no classification, zero intensity (Gujarat data)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 — Ground Classification                            │
│  1. SMRF (Simple Morphological Filter) via PDAL             │
│  2. CSF (Cloth Simulation Filter) fallback                  │
│  3. Random Forest refinement on 12 geometric features       │
│  Output: Classified .las (OGC LAS 1.4)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 — DTM Generation                                   │
│  • Extract ground points (Class 2)                          │
│  • IDW interpolation @ 0.5 m resolution                     │
│  • Gaussian smoothing                                       │
│  Output: dtm.tif (Cloud-Optimized GeoTIFF, OGC COG)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4 — Hydrological Analysis (pysheds)                  │
│  • Depression filling (Wang & Liu)                           │
│  • D8 flow direction                                        │
│  • Flow accumulation                                        │
│  • Slope, Aspect, TWI (Topographic Wetness Index)           │
│  • Stream extraction & catchment delineation                │
│  Outputs: slope.tif, flow_direction.tif, twi.tif (COG)     │
│           drainage_channels, depressions (GeoPackage GPKG)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5 — Waterlogging Prediction (XGBoost)                │
│  Features: elevation_norm, slope, TWI, TPI, curvature,      │
│            flow accumulation, depression depth, dist2stream  │
│  Labels: terrain-heuristic (TWI ≥ 8 ∧ TPI ≤ -0.3 ∧ slope ≤ 2°)│
│  Validation: 5-fold stratified CV, ROC-AUC                  │
│  Outputs: waterlogging_probability.tif (COG)                │
│           waterlogging_hotspots layer (GPKG)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6 — Drainage Network Design                          │
│  • Minimum Spanning Tree on flow graph                       │
│  • Rational Method: Q = C · i · A                           │
│  • Manning's equation: trapezoidal channel sizing           │
│  • Channel type selection: earthen / concrete / pipe        │
│  • Cost estimation (INR)                                    │
│  Output: drainage_channels, design_summary (GPKG)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📤 Output Files (OGC-Compliant)

| File | Format | Description |
|------|--------|-------------|
| `dtm.tif` | **COG** | Digital Terrain Model @ 0.5 m |
| `slope.tif` | **COG** | Slope in degrees |
| `twi.tif` | **COG** | Topographic Wetness Index |
| `flow_direction.tif` | **COG** | D8 flow direction codes |
| `flow_accumulation.tif` | **COG** | Log-scaled flow accumulation |
| `waterlogging_probability.tif` | **COG** | Waterlogging risk (0–1) |
| `drainage_network.gpkg` | **GPKG** | All vector layers (multi-layer) |
| `classified_ground.las` | **LAS** | Classified point cloud |

### GeoPackage Layers

| Layer | Type | Description |
|-------|------|-------------|
| `drainage_channels` | LineString | Designed drain segments with hydraulic specs |
| `waterlogging_hotspots` | Polygon | Risk zones (LOW / MEDIUM / HIGH) |
| `flow_paths` | LineString | Natural surface-water flow lines |
| `catchment_boundaries` | Polygon | Sub-catchment delineation |
| `depression_polygons` | Polygon | Topographic sinks and hollows |
| `design_summary` | Point | Aggregated design statistics |

---

## 🧠 AI/ML Models

### 1. Random Forest — Ground Classification Refinement
- 12 geometric features (eigenvalue decomposition, density, height above ground)
- Post-processes SMRF labels to remove false positives near buildings
- Handles zero-intensity data (Gujarat datasets)

### 2. XGBoost — Waterlogging Prediction
- 10 terrain-derived features per pixel
- Scale-positive-weight to handle imbalanced classes
- 5-fold stratified CV with ROC-AUC metric
- Outputs per-pixel probability map + polygon hotspots

### 3. PointNet (Deep Learning) — Optional Advanced Classification
- Patch-based training (1024-point patches)
- T-Net for input alignment
- Per-point binary segmentation
- Requires GPU; falls back to RF for CPU-only environments

---

## 🔧 Configuration

All parameters are controlled via `config/config.yaml`. Key settings:

```yaml
dtm:
  resolution: 0.5              # metres per pixel
  interpolation:
    method: "idw"
    idw_radius: 5.0

drainage:
  design_return_period: 10     # years
  rainfall_intensity: 50       # mm/hr (Gujarat 10-yr storm)
  runoff_coefficient: 0.65

waterlogging:
  model: "xgboost"
  threshold: 0.45              # probability threshold
```

---

## 📊 Data Sources

Point cloud data from [SVAMITVA Portal](https://svamitva.nic.in):

| State | District | File | Points | Density |
|-------|----------|------|--------|---------|
| Gujarat | Ahmedabad | DEVDI_511671.las | 64.6 M | 65 pts/m² |
| Gujarat | Sabar Kantha | KHAPRETA_510206.laz | 163.7 M | 245 pts/m² |
| Punjab | Hoshiarpur | Dhal_31235.las | — | — |
| Rajasthan | Hanumangarh | 67169_5NKR.las | — | — |
| Tamil Nadu | Thiruvallur | PIRAYANKUPPAM.las | — | — |

CRS: **EPSG:32643** (UTM Zone 43N) for Gujarat datasets.

---

## 📐 Known Data Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| No pre-existing ground classification | SMRF → CSF → RF three-stage pipeline |
| Zero intensity range | Intensity excluded from features; pure geometry used |
| Very high point density (245 pts/m²) | Voxel downsampling + tiled processing |
| Large file sizes (163M+ points) | 500m tiles with 25m overlap buffer |
| No historical flood records | Terrain-heuristic pseudo-labelling for XGBoost |

---

## 📋 Hackathon Deliverable Checklist

- [x] Automated DTM Creation using AI/ML
- [x] Ground classification (SMRF + Random Forest)  
- [x] Optimized Drainage Network (MST + Manning's)
- [x] GIS-ready outputs (OGC COG + GPKG)
- [x] Waterlogging hotspot prediction
- [x] Flow path delineation
- [x] Catchment boundary delineation
- [x] Cost estimation for drainage design
- [x] Fully documented codebase
- [x] CLI for reproducibility
- [ ] Integration with real-time weather API *(extension)*
- [ ] HEC-RAS hydraulic simulation export *(extension)*

---

## 🏗️ Tech Stack

| Component | Technology |
|-----------|------------|
| Point cloud I/O | laspy, PDAL |
| Ground classification | PDAL SMRF/CSF, scikit-learn RF |
| Deep learning | PyTorch, PointNet |
| DTM interpolation | SciPy IDW, pykrige |
| Raster processing | rasterio, rio-cogeo, GDAL |
| Hydrological modelling | pysheds |
| Waterlogging prediction | XGBoost |
| Network optimization | NetworkX (MST) |
| Vector GIS | GeoPandas, Shapely, Fiona |
| Visualization | matplotlib, rich |

---

## 🤝 Team

Built for the **MoPR Geospatial Intelligence Hackathon** organized by:
- Ministry of Panchayati Raj (MoPR)
- IIT Tirupati Navavishkar I-Hub Foundation (IITNiF)
- National Informatics Centre (NIC)

Output formats comply with **OGC standards** as recommended by the Open Geospatial Consortium guidance document.
