# Project Task Plan  
## AI-Driven DTM Creation and Drainage Optimization  
Geospatial Intelligence Hackathon – Problem Statement 2

---

# Project Goal

Transform raw drone point cloud data into:

- A clean Digital Terrain Model (DTM)
- Natural drainage network
- Waterlogging risk prediction map
- Optimized drainage design
- Standards-compliant geospatial outputs

The system must be automated, modular, and reproducible.

---

# Team Roles

1. Terrain & DTM Engineer  
2. Hydrology Engineer  
3. Machine Learning Engineer  
4. Drainage Optimization Engineer  
5. Systems & Documentation Engineer  

Each task below clearly states the responsible role.

---

# Task 1 – Point Cloud Preparation  
Responsible: Terrain & DTM Engineer

Objective:
Understand and prepare raw LAS/LAZ data for processing.

Work to be Done:

- Inspect CRS and ensure consistency (EPSG:32643).
- Verify point density and bounds.
- Confirm classification status (currently all class 0).
- Plan memory-safe processing (tiling if needed).
- Organize raw data folder structure.

Output:

- Clean working dataset
- Data summary report
- Ready-to-process point cloud files

---

# Task 2 – Ground Classification  
Responsible: Terrain & DTM Engineer

Objective:
Separate terrain (ground) from buildings and vegetation.

Work to be Done:

- Apply Cloth Simulation Filter (CSF) or equivalent.
- Generate new classification labels.
- Validate visually or statistically.
- Remove non-ground points.

Output:

- Ground-classified LAS file
- Documentation of filtering parameters used

---

# Task 3 – DTM Creation  
Responsible: Terrain & DTM Engineer

Objective:
Generate Digital Terrain Model from ground points.

Work to be Done:

- Interpolate ground points to 1m grid resolution.
- Create raster DTM.
- Perform sink filling to ensure hydrological correctness.
- Generate slope raster.
- Generate flow direction raster.
- Generate flow accumulation raster.

Output:

- DTM (Cloud Optimized GeoTIFF)
- Slope map
- Flow direction map
- Flow accumulation map

This is the foundation of the entire project.

---

# Task 4 – Hydrological Modeling  
Responsible: Hydrology Engineer

Objective:
Understand natural surface water movement.

Work to be Done:

- Validate flow direction.
- Identify high flow accumulation zones.
- Extract natural stream paths using threshold.
- Delineate watershed boundaries.
- Identify depressions and low-lying areas.

Output:

- Natural stream network (GeoPackage)
- Watershed polygons
- Low-lying zone map

---

# Task 5 – Waterlogging Risk Feature Preparation  
Responsible: Machine Learning Engineer

Objective:
Prepare terrain-derived features for model training.

Work to be Done:

- Collect terrain features:
  - Elevation
  - Slope
  - Curvature
  - Flow accumulation
  - Distance to stream
- Prepare feature dataset.
- Normalize and clean feature inputs.

Output:

- Structured ML-ready dataset
- Feature summary report

---

# Task 6 – Machine Learning-Based Waterlogging Prediction  
Responsible: Machine Learning Engineer

Objective:
Predict waterlogging probability using AI.

Work to be Done:

- Train Random Forest or Gradient Boosting model.
- Perform cross-validation.
- Evaluate accuracy, precision, recall, F1-score.
- Generate probability raster map.
- Analyze feature importance.

Output:

- Trained ML model
- Evaluation metrics
- Waterlogging probability raster (COG format)
- Feature importance analysis

---

# Task 7 – Natural Drainage Network Extraction  
Responsible: Hydrology Engineer

Objective:
Convert hydrology outputs into structured GIS-ready drainage network.

Work to be Done:

- Convert stream rasters into vector lines.
- Clean and simplify network.
- Ensure topology correctness.
- Export in GeoPackage format.

Output:

- Natural drainage network (GPKG)
- Validated drainage structure

---

# Task 8 – Drainage Network Optimization  
Responsible: Drainage Optimization Engineer

Objective:
Improve drainage system using graph modeling.

Work to be Done:

- Convert drainage network to graph structure.
- Analyze connectivity and bottlenecks.
- Simulate additional drainage channels.
- Compare before vs after waterlogging risk.
- Generate optimized drainage proposal.

Output:

- Optimized drainage network (GPKG)
- Comparison results
- Design explanation document

---

# Task 9 – Output Formatting and Validation  
Responsible: Systems & Documentation Engineer

Objective:
Ensure all outputs meet hackathon format requirements.

Work to be Done:

- Convert rasters to Cloud Optimized GeoTIFF.
- Ensure vector layers are in GeoPackage format.
- Validate CRS consistency.
- Verify file naming standards.
- Perform integration testing.

Output:

- Submission-ready geospatial files
- Validation checklist

---

# Task 10 – Documentation and Final Submission  
Responsible: Systems & Documentation Engineer

Objective:
Prepare professional submission package.

Work to be Done:

- Write README.md
- Write design.md
- Create architecture diagrams
- Prepare workflow explanation
- Include evaluation metrics
- Prepare presentation deck
- Ensure repository clarity

Output:

- Complete GitHub repository
- Submission package
- Presentation-ready materials

---

# Integration Flow

Terrain & DTM Engineer → Hydrology Engineer → ML Engineer → Optimization Engineer → Systems & Documentation Engineer

Each stage depends on the previous stage being correct.

---

# Critical Rule

No task is considered complete until:

- Outputs are reproducible.
- CRS is consistent.
- Formats follow required standards.
- Results are validated.

---

# Project Completion Criteria

The project is complete when:

- DTM is generated correctly.
- Natural drainage network is extracted.
- Waterlogging risk map is produced.
- Optimized drainage design is proposed.
- All outputs follow required formats.
- Documentation is clean and professional.
