# DATASET-DTM: Gujarat Point Cloud Dataset

A collection of LiDAR point cloud data from Gujarat, India, with analysis scripts for Digital Terrain Model (DTM) processing.

## Project Structure

```
DATASET-DTM/
├── Gujrat_Point_Cloud/
│   ├── DEVDI_POINT CLOUD (511671).las
│   └── KHAPRETA_510206.laz
├── scripts/
│   └── info.py
└── README.md
```

## Dataset

The `Gujrat_Point_Cloud/` directory contains LiDAR point cloud data in LAS/LAZ formats:

| File | Format | Description |
|------|--------|-------------|
| DEVDI_POINT CLOUD (511671).las | LAS | Point cloud data for Devdi region |
| KHAPRETA_510206.laz | LAZ | Compressed point cloud data for Khapreta region |

## Scripts

### info.py

Analyzes LAS/LAZ point cloud files and displays:
- Available dimensions
- Coordinate Reference System (CRS)
- Number of points
- Bounding box and estimated area
- Point density
- Classification information (including ground class detection)
- Intensity data range

## Requirements

```
laspy[lazrs]
laspy
pyproj
```

## Installation

```bash
pip install laspy[lazrs] pyproj
```

## Usage

Navigate to the `Gujrat_Point_Cloud/` directory and run the analysis script:

```bash
cd Gujrat_Point_Cloud
python ../scripts/info.py
```

Or modify the file paths in `info.py` to match your directory structure.

## Output Example

The script outputs detailed information about each point cloud file including:
- Dimension names (X, Y, Z, intensity, classification, etc.)
- CRS information
- Point count and density statistics
- Classification class presence (particularly ground class 2 for DTM generation)
- Intensity value ranges

## License

This dataset is provided for research and educational purposes.
