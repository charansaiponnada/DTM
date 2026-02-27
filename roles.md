# Team Roles

## AI-Driven DTM Creation and Drainage Optimization
Geospatial Intelligence Hackathon – Problem Statement 2

---

## Integration Flow

**Terrain & DTM Engineer → Hydrology Engineer → ML Engineer → Optimization Engineer → Systems & Documentation Engineer**

Each stage depends on the previous stage being correct.

---

## 1. Terrain & DTM Engineer

**Responsibilities:**
- Point cloud preparation and inspection
- Ground classification using filtering algorithms
- Digital Terrain Model (DTM) generation

**Tasks:**

| Task | Objective | Key Outputs |
|------|-----------|-------------|
| Task 1 – Point Cloud Preparation | Understand and prepare raw LAS/LAZ data | Clean working dataset, Data summary report |
| Task 2 – Ground Classification | Separate terrain from buildings and vegetation | Ground-classified LAS file, Filtering documentation |
| Task 3 – DTM Creation | Generate DTM from ground points | DTM (COG), Slope map, Flow direction map, Flow accumulation map |

---

## 2. Hydrology Engineer

**Responsibilities:**
- Hydrological analysis and modeling
- Natural drainage network extraction
- Watershed delineation

**Tasks:**

| Task | Objective | Key Outputs |
|------|-----------|-------------|
| Task 4 – Hydrological Modeling | Understand natural surface water movement | Natural stream network (GPKG), Watershed polygons, Low-lying zone map |
| Task 7 – Natural Drainage Network Extraction | Convert hydrology outputs into GIS-ready drainage network | Natural drainage network (GPKG), Validated drainage structure |

---

## 3. Machine Learning Engineer

**Responsibilities:**
- Feature engineering from terrain data
- Waterlogging risk prediction model development
- Model evaluation and probability mapping

**Tasks:**

| Task | Objective | Key Outputs |
|------|-----------|-------------|
| Task 5 – Waterlogging Risk Feature Preparation | Prepare terrain-derived features for model training | ML-ready dataset, Feature summary report |
| Task 6 – ML-Based Waterlogging Prediction | Predict waterlogging probability using AI | Trained model, Evaluation metrics, Waterlogging probability raster (COG) |

---

## 4. Drainage Optimization Engineer

**Responsibilities:**
- Graph-based drainage network analysis
- Drainage system optimization
- Improvement proposal generation

**Tasks:**

| Task | Objective | Key Outputs |
|------|-----------|-------------|
| Task 8 – Drainage Network Optimization | Improve drainage system using graph modeling | Optimized drainage network (GPKG), Comparison results, Design explanation |

---

## 5. Systems & Documentation Engineer

**Responsibilities:**
- Output formatting and validation
- Documentation and submission preparation
- Quality assurance and standards compliance

**Tasks:**

| Task | Objective | Key Outputs |
|------|-----------|-------------|
| Task 9 – Output Formatting and Validation | Ensure all outputs meet hackathon requirements | Submission-ready files, Validation checklist |
| Task 10 – Documentation and Final Submission | Prepare professional submission package | README.md, design.md, Architecture diagrams, Presentation materials |

---

## Critical Rule

No task is considered complete until:
- Outputs are reproducible
- CRS is consistent (EPSG:32643)
- Formats follow required standards (COG, GeoPackage)
- Results are validated
