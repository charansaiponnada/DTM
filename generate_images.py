"""Generate visualization images for the documentation."""

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io


def colorize_dtm(data, nodata=-9999):
    """Create RGB from elevation data."""
    data = data.astype(np.float32)
    data[data == nodata] = np.nan
    vmin, vmax = np.nanpercentile(data, (2, 98))
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    colors = np.array(
        [
            [20, 20, 120],
            [40, 100, 180],
            [0, 180, 100],
            [100, 220, 50],
            [220, 220, 0],
            [255, 100, 0],
            [180, 20, 20],
        ],
        dtype=np.float32,
    )
    idx = normalized * 6
    idx_floor = np.clip(np.floor(idx).astype(int), 0, 6)
    idx_ceil = np.clip(np.ceil(idx).astype(int), 0, 6)
    t = (idx - idx_floor).reshape(-1, 1)
    rgb = colors[idx_floor.flatten()] * (1 - t) + colors[idx_ceil.flatten()] * t
    rgb = rgb.reshape(data.shape + (3,)).astype(np.uint8)
    rgb[np.isnan(data)] = [0, 0, 0]
    return rgb, vmin, vmax


def colorize_risk(data, nodata=-9999):
    """Create RGB from waterlogging probability."""
    data = data.astype(np.float32)
    data[data == nodata] = np.nan
    data = np.clip(data, 0, 1)
    colors = np.array(
        [
            [255, 255, 255],
            [200, 230, 255],
            [100, 180, 255],
            [30, 100, 220],
            [10, 50, 180],
        ],
        dtype=np.float32,
    )
    idx = data * 4
    idx_floor = np.clip(np.floor(idx).astype(int), 0, 4)
    idx_ceil = np.clip(np.ceil(idx).astype(int), 0, 4)
    t = (idx - idx_floor).reshape(-1, 1)
    rgb = colors[idx_floor.flatten()] * (1 - t) + colors[idx_ceil.flatten()] * t
    rgb = rgb.reshape(data.shape + (3,)).astype(np.uint8)
    rgb[np.isnan(data)] = [240, 240, 240]
    return rgb


# Paths
output_dir = "data/output/DEVDI"
images_dir = "documentation/images"

# 1. Generate DTM visualization
print("Generating DTM visualization...")
with rasterio.open(f"{output_dir}/dtm.tif") as src:
    dtm_data = src.read(1)
    dtm_bounds = src.bounds

rgb, vmin, vmax = colorize_dtm(dtm_data)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(rgb, origin="upper")
ax.set_title("Digital Terrain Model (DTM)", fontsize=14, fontweight="bold")
ax.set_xlabel("Column")
ax.set_ylabel("Row")

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label("Elevation (m)", fontsize=10)

plt.tight_layout()
plt.savefig(
    f"{images_dir}/fig_dtm.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {images_dir}/fig_dtm.png")

# 2. Generate Waterlogging visualization
print("Generating Waterlogging visualization...")
with rasterio.open(f"{output_dir}/waterlogging_probability.tif") as src:
    wl_data = src.read(1)

rgb_wl = colorize_risk(wl_data)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(rgb_wl, origin="upper")
ax.set_title("Waterlogging Risk Probability Map", fontsize=14, fontweight="bold")
ax.set_xlabel("Column")
ax.set_ylabel("Row")

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label("Probability", fontsize=10)

plt.tight_layout()
plt.savefig(
    f"{images_dir}/fig_waterlogging.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {images_dir}/fig_waterlogging.png")

# 3. Generate Drainage Network visualization
print("Generating Drainage Network visualization...")
with rasterio.open(f"{output_dir}/dtm.tif") as src:
    dtm_data = src.read(1)
    dtm_bounds = src.bounds

# Load drainage channels
gdf = gpd.read_file(f"{output_dir}/drainage_network.gpkg", layer="drainage_channels")

fig, ax = plt.subplots(figsize=(10, 8))

# DTM background
dtm_data_f = dtm_data.astype(np.float32)
dtm_data_f[dtm_data_f == -9999] = np.nan
vmin, vmax = np.nanpercentile(dtm_data_f, (2, 98))
colors = (
    np.array(
        [
            [20, 20, 120],
            [40, 100, 180],
            [0, 180, 100],
            [100, 220, 50],
            [220, 220, 0],
            [255, 100, 0],
            [180, 20, 20],
        ],
        dtype=np.float32,
    )
    / 255
)
cmap = LinearSegmentedColormap.from_list("elevation", colors)

ax.imshow(
    dtm_data_f,
    cmap=cmap,
    norm=plt.Normalize(vmin=vmin, vmax=vmax),
    extent=[dtm_bounds.left, dtm_bounds.right, dtm_bounds.bottom, dtm_bounds.top],
    origin="upper",
    alpha=0.8,
)

# Drainage overlay
gdf.plot(ax=ax, color="#2c5282", linewidth=1.5, alpha=0.9)

ax.set_title("Drainage Network Plan View", fontsize=14, fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

# Stats
total_length = gdf["length_m"].sum() / 1000
num_channels = len(gdf)
total_cost = gdf["cost_inr"].sum() / 100000

stats_text = (
    f"Channels: {num_channels}\nLength: {total_length:.1f} km\nCost: ₹{total_cost:.1f}L"
)
ax.text(
    0.02,
    0.98,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

plt.tight_layout()
plt.savefig(
    f"{images_dir}/fig_drainage.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {images_dir}/fig_drainage.png")

# 4. Generate Terrain derivatives grid
print("Generating Terrain derivatives...")
terrain_files = {
    "slope.tif": "Slope",
    "twi.tif": "Topographic Wetness Index",
    "hillshade.tif": "Hillshade",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (fname, label) in enumerate(terrain_files.items()):
    with rasterio.open(f"{output_dir}/{fname}") as src:
        data = src.read(1)
    data = data.astype(np.float32)
    data[data == -9999] = np.nan
    vmin, vmax = np.nanpercentile(data, (2, 98))
    normalized = ((data - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    axes[i].imshow(normalized, cmap="gray", origin="upper")
    axes[i].set_title(label, fontsize=12, fontweight="bold")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(
    f"{images_dir}/fig_terrain.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {images_dir}/fig_terrain.png")

print("\nAll visualization images generated successfully!")
