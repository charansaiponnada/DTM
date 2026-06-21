"""
scripts/tile_las.py
───────────────────
Spatially tile a LAS/LAZ file into N×N grid tiles for cross-village evaluation.
Each tile is saved as a standalone .las file.

Usage
-----
    python scripts/tile_las.py --input data/input/DEVDI_511671.las --grid 2 2 --output data/input/tiles/
"""

from __future__ import annotations
from pathlib import Path
import click
import numpy as np
import laspy
from loguru import logger


def tile_las(input_path: str | Path, output_dir: str | Path, grid_rows: int = 2, grid_cols: int = 2, prefix: str = ""):
    """
    Split a LAS file into grid_rows × grid_cols spatial tiles.
    Each tile gets points within its spatial quadrant.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    las = laspy.read(str(input_path))
    x = np.array(las.x)
    y = np.array(las.y)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_edges = np.linspace(x_min, x_max, grid_cols + 1)
    y_edges = np.linspace(y_min, y_max, grid_rows + 1)

    if not prefix:
        prefix = input_path.stem

    tile_paths = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            y_lo, y_hi = y_edges[row], y_edges[row + 1]
            x_lo, x_hi = x_edges[col], x_edges[col + 1]

            mask = (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)
            n_pts = mask.sum()
            if n_pts == 0:
                logger.warning(f"Tile ({row},{col}) is empty — skipping")
                continue

            tile_name = f"{prefix}_tile_r{row}c{col}.las"
            tile_path = output_dir / tile_name

            # Use laspy's point format to create subset
            from contextlib import closing
            with laspy.open(str(input_path)) as reader:
                header = reader.header
                # Create new header with correct point count
                new_header = laspy.LasHeader(
                    point_format=header.point_format,
                    version=header.version,
                )
                new_header.offsets = header.offsets
                new_header.scales = header.scales

            tile_las = laspy.LasData(new_header)
            for dim in las.point_format.dimensions:
                name = dim.name
                setattr(tile_las, name, getattr(las, name)[mask])

            tile_las.write(str(tile_path))
            tile_paths.append(tile_path)
            logger.info(f"Tile {tile_name}: {n_pts:,} points ({100*n_pts/len(x):.1f}%) → {tile_path}")

    logger.success(f"Created {len(tile_paths)} tiles in {output_dir}")
    return tile_paths


@click.command()
@click.option("--input", "-i", required=True, help="Input LAS/LAZ file")
@click.option("--output", "-o", default="data/input/tiles", help="Output directory")
@click.option("--grid", nargs=2, type=int, default=[2, 2], help="Grid rows and cols (e.g. 2 2 = 4 tiles)")
@click.option("--prefix", default="", help="Optional tile name prefix")
def main(input, output, grid, prefix):
    tile_las(input, output, grid_rows=grid[0], grid_cols=grid[1], prefix=prefix)


if __name__ == "__main__":
    main()
