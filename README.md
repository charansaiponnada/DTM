# AI-Driven Digital Terrain Modeling and Drainage Optimization System  
## Geospatial Intelligence Hackathon – IIT Tirupati  
### Problem Statement 2: DTM Creation Using AI/ML from Point Cloud Data
For design and architecture, see [Design Documentation](design.md).
---

# 1. Executive Summary

This project presents a complete terrain intelligence pipeline for generating a high-quality Digital Terrain Model (DTM) from unclassified drone-derived point cloud data and designing an optimized drainage network for densely inhabited village areas.

The system processes raw point cloud data, performs automated ground classification, generates hydrologically corrected terrain surfaces, extracts natural surface-water flow paths, predicts waterlogging hotspots using machine learning, and proposes an optimized drainage design suitable for governance deployment.

The solution is modular, scalable, standards-compliant, and reproducible using a fully Python-based workflow.

---

# 2. Problem Context

Village settlements (abadi areas) frequently experience waterlogging due to:

- Poor surface drainage
- Natural depressions
- Unplanned construction
- Inadequate drainage connectivity

Raw drone point cloud data provides high-resolution surface representation but does not directly deliver terrain intelligence.

This project converts unclassified point cloud data into actionable geospatial intelligence for drainage planning.

---

# 3. Input Data Characteristics (Got info after the GUJARAT DATASET need to check with other datasets too)

Dataset Type: Drone-derived point cloud (.las / .laz)  
CRS: WGS 84 / UTM Zone 43N (EPSG:32643)  
Point Density: 65–245 points per square meter  
Classification: Unclassified (Class 0 only)  
Intensity: Not usable (all values = 0)  

Observations:

- No ground class present
- No usable LiDAR intensity
- High-resolution photogrammetric surface model
- Requires complete ground filtering before DTM generation

---

# 4. System Architecture

The system follows a six-stage terrain intelligence pipeline:

Raw Point Cloud  
→ Ground Classification  
→ DTM Generation  
→ Hydrological Modeling  
→ ML-Based Waterlogging Prediction  
→ Drainage Network Optimization  

Each module is independently executable and configurable.

---

# 5. Methodology

## 5.1 Ground Classification

Since the dataset contains only unclassified points, a physics-based filtering approach was required.

We implemented Cloth Simulation Filter (CSF) to separate ground from surface objects.

Concept:

The point cloud is inverted and a virtual cloth is dropped over the surface.  
Points that the cloth touches are classified as ground.

Output:
- Ground-classified LAS file
- Non-ground separation (buildings, vegetation)

Why this method:

- Stable for high-density drone data
- Computationally efficient on CPU
- Industry-accepted geomatics technique
- Does not require labeled training data

---

## 5.2 DTM Generation

Ground points were interpolated into a 1-meter resolution raster grid.

Interpolation Strategy:
- Grid-based terrain reconstruction
- Noise reduction
- Edge consistency enforcement

Hydrological correction:
- Sink filling applied to ensure continuous flow
- Depression artifacts removed

Generated Raster Outputs:
- Digital Terrain Model (DTM)
- Slope raster
- Flow direction raster
- Flow accumulation raster
- Curvature rasters

All rasters exported as:

Cloud Optimized GeoTIFF (COG)

---

## 5.3 Hydrological Analysis

From the filled DTM:

- Flow direction computed using D8 algorithm
- Flow accumulation derived
- Stream network extracted using threshold-based accumulation
- Watershed boundaries delineated
- Low-lying depressions identified

Vector outputs generated in:

GeoPackage (.gpkg) format

This produces natural drainage networks and potential stagnation zones.

---

## 5.4 AI-Based Waterlogging Prediction

A terrain-feature-based machine learning model was developed to estimate waterlogging risk.

Feature Engineering:

- Elevation
- Slope
- Profile curvature
- Plan curvature
- Flow accumulation
- Distance to natural stream
- Relative terrain position

Model Used:

Random Forest Classifier

Why Random Forest:

- Handles nonlinear terrain relationships
- Robust to noise
- Interpretable feature importance
- Stable under limited labeled data

Output:

Waterlogging Probability Raster (COG format)

Evaluation Metrics:

- Cross-validation accuracy
- Precision
- Recall
- F1-score
- Feature importance ranking

This ML layer enhances predictive capability beyond rule-based GIS analysis.

---

## 5.5 Drainage Network Optimization

The extracted stream network was converted into a graph structure.

Graph-Based Optimization:

- Nodes represent junctions
- Edges represent drainage segments
- Weight defined by length and slope constraints

Optimization Goals:

- Minimize drainage path length
- Maintain gravitational slope feasibility
- Improve connectivity to natural outlets
- Reduce predicted waterlogging zones

Simulation:

Before optimization:
Natural drainage only

After optimization:
Artificial channel suggestions added

Output:

- Optimized drainage network (GeoPackage)
- Comparative impact analysis
- Design parameter summary

This transforms terrain mapping into infrastructure planning.

---

# 6. Output Compliance

All outputs follow recommended open geospatial standards:

DTM and derived rasters → Cloud Optimized GeoTIFF (COG)  
Drainage and vector layers → GeoPackage (GPKG)  
Classified point clouds → LAS  

These formats ensure interoperability across GIS, AI, and governance systems.

---

# 7. Repository Structure

```
project-root/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── ground_classification/
│   ├── dtm_generation/
│   ├── hydrology/
│   ├── ml_model/
│   ├── optimization/
│
├── models/
│
├── outputs/
│   ├── rasters/
│   ├── vectors/
│
├── docs/
│   ├── workflow_diagram.png
│   ├── architecture.png
│
├── requirements.txt
├── design.md
└── README.md
```

---

# 8. Technology Stack

Python 3.x  
PDAL  
laspy  
rasterio  
numpy  
scipy  
geopandas  
shapely  
networkx  
scikit-learn  

Processing Environment:

CPU-based implementation  
Modular design allows GPU extension if required  

---

# 9. Team Structure

Terrain Processing Lead  
Handles ground filtering and DTM generation.

Hydrology Lead  
Responsible for terrain-derived drainage extraction.

AI/ML Lead  
Builds waterlogging prediction model.

Optimization Lead  
Designs graph-based drainage improvements.

Documentation & Integration Lead  
Ensures standard compliance, reporting, evaluation metrics, and final submission quality.

Project Lead  
Architectural decisions and module integration.

---

# 10. Performance Considerations

- Tiling strategy for large point clouds
- Memory-efficient streaming using PDAL
- 1m grid resolution selected for balance between precision and computational efficiency
- Modular execution to avoid redundant computation

---

# 11. Scalability

The pipeline is designed to process additional villages with minimal configuration changes.

Adjustable parameters:

- Grid resolution
- Flow accumulation threshold
- Optimization constraints
- ML hyperparameters

---

# 12. Future Work

- Rainfall intensity simulation integration
- Impervious surface classification
- GPU-accelerated point cloud processing
- Field validation against ground survey data
- Integration into web-based decision support systems

---

# 13. Conclusion

This project delivers a complete AI-assisted terrain intelligence system that transforms raw point cloud data into governance-ready drainage planning outputs.

By integrating geomatics principles, hydrological modeling, machine learning, and graph-based optimization, the solution moves beyond visualization and provides actionable infrastructure design recommendations.

The system is modular, standards-compliant, and scalable for large-scale rural deployment.
