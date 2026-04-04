"""Generate input point cloud visualization for documentation."""

import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the LAS file
las = laspy.read("data/input/DEVDI_511671.las")

# Get coordinates
x = las.x
y = las.y
z = las.z

print(f"Total points: {len(x):,}")

# Sample for visualization (keep every 100th point for performance)
sample_rate = 100
x_sampled = x[::sample_rate]
y_sampled = y[::sample_rate]
z_sampled = z[::sample_rate]

print(f"Sampled points: {len(x_sampled):,}")

# Create 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# Color by elevation
colors = z_sampled
scatter = ax.scatter(
    x_sampled, y_sampled, z_sampled, c=colors, s=0.5, cmap="jet", alpha=0.3
)

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_zlabel("Elevation (m)")
ax.set_title(
    "Input Point Cloud - DEVDI Village (64.6M points)", fontsize=14, fontweight="bold"
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, label="Elevation (m)")

# Set viewing angle
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig(
    "documentation/images/fig_input_pointcloud.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: documentation/images/fig_input_pointcloud.png")

# Also create a 2D bird's eye view
fig2, ax2 = plt.subplots(figsize=(12, 10))
scatter2 = ax2.scatter(x_sampled, y_sampled, c=z_sampled, s=0.3, cmap="jet", alpha=0.5)
ax2.set_xlabel("Easting (m)")
ax2.set_ylabel("Northing (m)")
ax2.set_title(
    "Input Point Cloud - Top View (DEVDI Village)", fontsize=14, fontweight="bold"
)
ax2.set_aspect("equal")
cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.7, label="Elevation (m)")
plt.tight_layout()
plt.savefig(
    "documentation/images/fig_input_topview.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: documentation/images/fig_input_topview.png")

print("\nInput visualization images generated successfully!")
