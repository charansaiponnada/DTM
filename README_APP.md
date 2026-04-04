# DTM Drainage AI - Streamlit Web App

A web interface for the MoPR Hackathon DTM + Drainage Network pipeline.

## Quick Start

```bash
# Install Streamlit dependencies
pip install -r requirements_app.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Features

1. **Upload LAS/LAZ** - Upload drone point cloud data
2. **Configure Pipeline** - Select stages, stream threshold, resolution
3. **Run Pipeline** - Execute the full DTM + Drainage AI workflow
4. **View Results** - See metrics, visualizations, and download outputs

## Output Files

After running, you can download:
- `dtm.tif` - Digital Terrain Model (GeoTIFF)
- `waterlogging_probability.tif` - Flood risk zones
- `drainage_network.gpkg` - Drainage network (GeoPackage)
- `flow_direction.tif`, `flow_accumulation.tif` - Hydrology layers
- `slope.tif`, `aspect.tif`, `hillshade.tif` - Terrain derivatives

## Full Visualization

For complete GIS visualization, use **QGIS** to open the GeoPackage and GeoTIFF files.