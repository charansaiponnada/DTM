"""
src/preprocessing/point_cloud_loader.py
────────────────────────────────────────
Loads LAS/LAZ point clouds using laspy, optionally tiles them for
memory-efficient processing of the Gujarat datasets (64M / 163M pts).
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import laspy
import numpy as np
from loguru import logger
from tqdm import tqdm


@dataclass
class PointCloudMetadata:
    """Summary statistics extracted from a point cloud file."""
    filepath: Path
    point_count: int
    crs_wkt: Optional[str]
    min_bounds: np.ndarray          # (x_min, y_min, z_min)
    max_bounds: np.ndarray          # (x_max, y_max, z_max)
    scale: Tuple[float, float, float]
    offset: Tuple[float, float, float]
    has_classification: bool
    has_intensity: bool
    intensity_range: Tuple[float, float]
    point_format_id: int
    extra_dims: List[str] = field(default_factory=list)

    @property
    def extent_xy(self) -> Tuple[float, float]:
        """Returns (width_m, height_m) of the bounding box."""
        return (
            float(self.max_bounds[0] - self.min_bounds[0]),
            float(self.max_bounds[1] - self.min_bounds[1]),
        )

    @property
    def area_sqm(self) -> float:
        w, h = self.extent_xy
        return w * h

    @property
    def density_pts_sqm(self) -> float:
        return self.point_count / self.area_sqm if self.area_sqm > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"PointCloud: {self.filepath.name}\n"
            f"  Points       : {self.point_count:,}\n"
            f"  Bounds XY    : {self.min_bounds[:2]} → {self.max_bounds[:2]}\n"
            f"  Extent       : {self.extent_xy[0]:.1f} × {self.extent_xy[1]:.1f} m\n"
            f"  Density      : {self.density_pts_sqm:.2f} pts/m²\n"
            f"  CRS          : {'present' if self.crs_wkt else 'MISSING – assume EPSG:32643'}\n"
            f"  Classification: {self.has_classification}\n"
            f"  Intensity     : {self.has_intensity} "
            f"(range {self.intensity_range[0]:.0f}–{self.intensity_range[1]:.0f})\n"
        )


def inspect(filepath: str | Path) -> PointCloudMetadata:
    """
    Fast metadata scan without loading all points into memory.
    Uses laspy header-only read.
    """
    filepath = Path(filepath)
    logger.info(f"Inspecting {filepath.name} …")

    with laspy.open(filepath) as f:
        hdr = f.header
        pt_fmt = hdr.point_format  # laspy v2: point_format is on the header, not the reader

        if hdr.point_count == 0:
            raise ValueError(f"LAS file contains 0 points: {filepath.name}")

        # read a small sample to check intensity range
        sample_size = min(50_000, hdr.point_count)
        chunk = next(f.chunk_iterator(sample_size))
        intensity = np.array(chunk.intensity, dtype=np.float32)
        int_range = (float(intensity.min()), float(intensity.max()))
        has_classification = hasattr(chunk, "classification")
        extra_dims = [d.name for d in pt_fmt.extra_dimensions]

    meta = PointCloudMetadata(
        filepath=filepath,
        point_count=hdr.point_count,
        crs_wkt=hdr.parse_crs().to_wkt() if hdr.parse_crs() is not None else None,
        min_bounds=np.array([hdr.x_min, hdr.y_min, hdr.z_min]),
        max_bounds=np.array([hdr.x_max, hdr.y_max, hdr.z_max]),
        scale=tuple(hdr.scales),
        offset=tuple(hdr.offsets),
        has_classification=has_classification,
        has_intensity=True,
        intensity_range=int_range,
        point_format_id=pt_fmt.id,
        extra_dims=extra_dims,
    )
    logger.info(str(meta))
    return meta


def load_full(
    filepath: str | Path,
    extra_dims: Optional[List[str]] = None,
) -> laspy.LasData:
    """
    Load entire LAS/LAZ file into memory.
    ⚠  Only safe for files < ~4 GB RAM footprint.
    For Gujarat data (163M pts) prefer load_tiles() or load_chunked().
    """
    filepath = Path(filepath)
    logger.info(f"Loading {filepath.name} into memory …")
    las = laspy.read(filepath)
    logger.success(
        f"Loaded {las.header.point_count:,} points "
        f"(format {las.point_format.id})"
    )
    return las


def load_chunked(
    filepath: str | Path,
    chunk_size: int = 5_000_000,
) -> Iterator[laspy.PackedPointRecord]:
    """
    Yield successive chunks for streaming/out-of-core processing.

    Parameters
    ----------
    filepath    : path to .las / .laz file
    chunk_size  : number of points per chunk (default 5 M)

    Yields
    ------
    laspy.PackedPointRecord chunks
    """
    filepath = Path(filepath)
    logger.info(f"Streaming {filepath.name} in chunks of {chunk_size:,} …")
    with laspy.open(filepath) as f:
        n_chunks = int(np.ceil(f.header.point_count / chunk_size))
        for i, chunk in enumerate(
            tqdm(f.chunk_iterator(chunk_size), total=n_chunks, desc="chunks")
        ):
            yield chunk


def load_tiles(
    filepath: str | Path,
    tile_size: float = 500.0,
    buffer: float = 25.0,
    output_dir: Optional[str | Path] = None,
) -> List[Path]:
    """
    Spatially tile a large LAS/LAZ into smaller .las files on disk,
    with an overlap buffer to avoid edge effects.

    Parameters
    ----------
    filepath   : input LAS/LAZ
    tile_size  : tile edge length in metres
    buffer     : overlap margin in metres
    output_dir : where to write tile files (defaults to <input>_tiles/)

    Returns
    -------
    List of paths to created tile files.
    """
    filepath = Path(filepath)
    meta = inspect(filepath)
    x_min, y_min = meta.min_bounds[:2]
    x_max, y_max = meta.max_bounds[:2]

    if output_dir is None:
        output_dir = filepath.parent / f"{filepath.stem}_tiles"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build tile grid
    xs = np.arange(x_min, x_max, tile_size)
    ys = np.arange(y_min, y_max, tile_size)
    tile_paths: List[Path] = []

    logger.info(
        f"Tiling {filepath.name} → {len(xs)*len(ys)} tiles "
        f"({tile_size} m + {buffer} m buffer)"
    )

    # Read full cloud once (chunk-by-chunk) and bucket into tiles
    # Grab the source header first so tile writers inherit the exact point
    # format (including extra dims), scale, and offset of the original file.
    with laspy.open(filepath) as _src:
        src_header = _src.header

    tile_points: dict[Tuple[int, int], List[laspy.PackedPointRecord]] = {}
    for chunk in load_chunked(filepath):
        xs_pts = np.array(chunk.x)
        ys_pts = np.array(chunk.y)
        for ix, tx in enumerate(xs):
            for iy, ty in enumerate(ys):
                mask = (
                    (xs_pts >= tx - buffer) & (xs_pts < tx + tile_size + buffer) &
                    (ys_pts >= ty - buffer) & (ys_pts < ty + tile_size + buffer)
                )
                if mask.sum() == 0:
                    continue
                key = (ix, iy)
                if key not in tile_points:
                    tile_points[key] = []
                tile_points[key].append(chunk[mask])

    # Write tiles
    for (ix, iy), chunks in tqdm(tile_points.items(), desc="writing tiles"):
        tile_path = output_dir / f"tile_{ix:04d}_{iy:04d}.las"
        with laspy.open(tile_path, mode="w", header=src_header) as out:
            for c in chunks:
                out.write_points(c)
        tile_paths.append(tile_path)
        logger.debug(f"Wrote tile ({ix},{iy}) → {tile_path.name}")

    # Persist tile index
    index_path = output_dir / "tile_index.json"
    with open(index_path, "w") as fh:
        json.dump(
            {
                "source": str(filepath),
                "tile_size": tile_size,
                "buffer": buffer,
                "crs": "EPSG:32643",
                "tiles": [str(p) for p in tile_paths],
            },
            fh,
            indent=2,
        )
    logger.success(f"Tiling complete: {len(tile_paths)} tiles → {output_dir}")
    return tile_paths


def extract_xyz_array(
    las: laspy.LasData | laspy.PackedPointRecord,
) -> np.ndarray:
    """Return (N, 3) float64 array of [X, Y, Z] coordinates."""
    return np.column_stack([
        np.array(las.x, dtype=np.float64),
        np.array(las.y, dtype=np.float64),
        np.array(las.z, dtype=np.float64),
    ])
