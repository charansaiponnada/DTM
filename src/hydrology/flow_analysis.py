"""
src/hydrology/flow_analysis.py
───────────────────────────────
Hydrological terrain analysis pipeline:
  - Depression filling (Wang & Liu algorithm via pysheds)
  - Flow direction (D8)
  - Flow accumulation
  - Topographic Wetness Index (TWI)
  - Stream network extraction
  - Catchment delineation

All outputs written as COG (raster) or GPKG (vector).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import shape, mapping, LineString
from loguru import logger

# pysheds for pure-Python hydrological modelling
from pysheds.grid import Grid

from src.dtm.dtm_generator import write_geotiff, convert_to_cog, NODATA


# ══════════════════════════════════════════════════════════════════════════
#  Pysheds-based Hydrological Analysis
# ══════════════════════════════════════════════════════════════════════════

class HydrologicalAnalyzer:
    """
    Wraps pysheds Grid to provide a clean interface for the drainage
    network design pipeline.

    Usage
    -----
    ha = HydrologicalAnalyzer.from_dtm("dtm.tif")
    ha.fill_depressions()
    ha.compute_flow_direction()
    ha.compute_flow_accumulation()
    ha.compute_twi()
    streams_gdf = ha.extract_streams(threshold=1000)
    ha.export_all("data/output/")
    """

    def __init__(self, grid: Grid, dtm_path: Path, crs: str = "EPSG:32643"):
        self.grid     = grid
        self.dtm_path = dtm_path
        self.crs      = crs

        # Will be populated as analysis proceeds
        self.dem_filled  = None
        self.fdir        = None
        self.acc         = None
        self.twi         = None
        self.slope       = None

    # ── Constructors ─────────────────────────────────────────────────────

    @classmethod
    def from_dtm(cls, dtm_path: str | Path, crs: str = "EPSG:32643") -> "HydrologicalAnalyzer":
        dtm_path = Path(dtm_path)
        logger.info(f"Loading DTM for hydrological analysis: {dtm_path.name}")
        grid = Grid.from_raster(str(dtm_path))
        dem  = grid.read_raster(str(dtm_path))
        grid.dem = dem
        obj = cls(grid, dtm_path, crs)
        return obj

    # ── Core Processing Steps ────────────────────────────────────────────

    def fill_depressions(self) -> "HydrologicalAnalyzer":
        """
        Fill sinks in the DEM to ensure continuous flow routing.
        Uses Wang & Liu (2006) algorithm (implemented in pysheds).

        For village abadi data this is critical – small depressions
        caused by interpolation artefacts must be filled.
        """
        logger.info("Filling depressions (Wang & Liu) …")
        self.dem_filled = self.grid.resolve_flats(
            self.grid.fill_depressions(self.grid.dem)
        )
        logger.success("Depressions filled.")
        return self

    def compute_flow_direction(
        self,
        algorithm: str = "d8",
        apply_flats: bool = True,
    ) -> "HydrologicalAnalyzer":
        """
        Compute D8 flow direction from filled DEM.
        D8 assigns each cell to flow to the steepest of its 8 neighbours.

        Parameters
        ----------
        algorithm   : "d8" (deterministic) – only D8 supported by pysheds
        apply_flats : resolve flat areas before computing direction
        """
        if self.dem_filled is None:
            self.fill_depressions()

        logger.info("Computing D8 flow direction …")
        self.fdir = self.grid.flowdir(self.dem_filled)
        logger.success("Flow direction computed.")
        return self

    def compute_flow_accumulation(self) -> "HydrologicalAnalyzer":
        """
        Compute flow accumulation (upslope contributing area in cells).
        High values → natural drainage channels and low-lying risk areas.
        """
        if self.fdir is None:
            self.compute_flow_direction()

        logger.info("Computing flow accumulation …")
        self.acc = self.grid.accumulation(self.fdir)
        logger.success(
            f"Flow accumulation computed. "
            f"Max: {self.acc.max():.0f} cells, "
            f"Mean: {self.acc.mean():.1f} cells"
        )
        return self

    def compute_slope(self) -> "HydrologicalAnalyzer":
        """Compute slope (degrees) from filled DEM."""
        logger.info("Computing slope …")
        with rasterio.open(self.dtm_path) as src:
            dem_arr  = src.read(1).astype(np.float32)
            res      = src.res[0]           # metres per pixel

        # Central difference gradient
        dy, dx   = np.gradient(dem_arr, res)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        self.slope = np.degrees(slope_rad).astype(np.float32)
        logger.success(
            f"Slope: mean {self.slope.mean():.2f}°, "
            f"max {self.slope.max():.2f}°"
        )
        return self

    def compute_twi(self) -> "HydrologicalAnalyzer":
        """
        Topographic Wetness Index  TWI = ln(α / tan β)
        where α = specific catchment area, β = local slope.

        High TWI → terrain tends to accumulate and retain water
        → primary input feature for waterlogging prediction.
        """
        if self.acc is None:
            self.compute_flow_accumulation()
        if self.slope is None:
            self.compute_slope()

        logger.info("Computing TWI …")
        with rasterio.open(self.dtm_path) as src:
            res = src.res[0]

        acc_arr  = np.array(self.acc, dtype=np.float64)
        sca      = (acc_arr + 1) * res * res   # specific catchment area (m²)
        slope_r  = np.radians(self.slope.astype(np.float64))
        tan_beta = np.tan(np.where(slope_r > 0.001, slope_r, 0.001))

        self.twi = np.log(sca / tan_beta).astype(np.float32)
        logger.success(
            f"TWI: mean {self.twi.mean():.2f}, "
            f"max {self.twi.max():.2f}"
        )
        return self

    # ── Stream Extraction ────────────────────────────────────────────────

    def extract_streams(
        self,
        threshold: int = 1000,
        output_gpkg: Optional[str | Path] = None,
    ) -> gpd.GeoDataFrame:
        """
        Extract stream network as vector GeoDataFrame.

        Parameters
        ----------
        threshold     : minimum flow accumulation (cells) to define a stream
        output_gpkg   : optional path to save as GeoPackage layer

        Returns
        -------
        GeoDataFrame with LineString geometries (drainage channels)
        """
        if self.acc is None:
            self.compute_flow_accumulation()

        logger.info(f"Extracting streams (threshold = {threshold} cells) …")

        acc_arr = np.array(self.acc, dtype=np.float32)
        stream_mask = (acc_arr >= threshold).astype(np.uint8)

        with rasterio.open(self.dtm_path) as src:
            transform = src.transform
            crs_obj   = src.crs or CRS.from_epsg(32643)

        if stream_mask.sum() == 0:
            logger.warning("No streams found – lower threshold?")
            return gpd.GeoDataFrame(columns=["geometry", "acc_value", "order"])

        # Trace stream network by following D8 flow directions – O(N)
        fdir_arr = np.array(self.fdir, dtype=np.int32)
        stream_lines = self._raster_streams_to_lines(
            stream_mask, fdir_arr, acc_arr, transform, crs_obj
        )

        n = len(stream_lines)
        logger.success(f"Extracted {n} stream channel segments")

        if output_gpkg:
            self._save_gpkg(stream_lines, output_gpkg, layer="drainage_channels")

        return stream_lines

    def extract_depressions(
        self,
        min_depth: float = 0.1,
        output_gpkg: Optional[str | Path] = None,
    ) -> gpd.GeoDataFrame:
        """
        Identify and vectorise topographic depressions (sinks).
        These are primary waterlogging risk zones.

        Parameters
        ----------
        min_depth   : minimum depression depth in metres to include
        """
        with rasterio.open(self.dtm_path) as src:
            dem_arr   = src.read(1).astype(np.float32)
            dem_arr   = np.where(dem_arr == NODATA, np.nan, dem_arr)
            transform = src.transform
            crs_obj   = src.crs or CRS.from_epsg(32643)

        filled_arr = np.array(self.dem_filled, dtype=np.float32) if self.dem_filled is not None else dem_arr.copy()
        depth = filled_arr - dem_arr
        depth_mask = ((depth >= min_depth) & np.isfinite(depth)).astype(np.uint8)

        geoms = [
            {"geometry": geom, "depth_m": float(val)}
            for geom, val in shapes(depth * depth_mask, transform=transform)
            if val >= min_depth
        ]

        if not geoms:
            logger.warning("No depressions found with depth >= {min_depth} m")
            return gpd.GeoDataFrame(columns=["geometry", "depth_m"])

        dep_gdf = gpd.GeoDataFrame(
            [{"depth_m": g["depth_m"], "geometry": shape(g["geometry"])} for g in geoms],
            crs=crs_obj,
        )
        dep_gdf["area_m2"] = dep_gdf.geometry.area
        dep_gdf = dep_gdf.sort_values("depth_m", ascending=False).reset_index(drop=True)

        logger.success(f"Found {len(dep_gdf)} depressions (depth ≥ {min_depth} m)")

        if output_gpkg:
            self._save_gpkg(dep_gdf, output_gpkg, layer="depression_polygons")

        return dep_gdf

    def delineate_catchments(
        self,
        outlet_points: Optional[gpd.GeoDataFrame] = None,
        output_gpkg: Optional[str | Path] = None,
    ) -> gpd.GeoDataFrame:
        """
        Delineate sub-catchments for each drainage outlet.
        Uses pysheds catchment function.
        """
        if self.fdir is None:
            self.compute_flow_direction()

        with rasterio.open(self.dtm_path) as src:
            transform = src.transform
            crs_obj   = src.crs or CRS.from_epsg(32643)

        if outlet_points is None:
            # Auto-detect outlets as points of highest flow accumulation
            outlet_points = self._auto_detect_outlets(n=5)

        catchments = []
        for idx, row in outlet_points.iterrows():
            x, y = row.geometry.x, row.geometry.y
            # Snap to nearest grid cell
            col = int((x - transform.c) / transform.a)
            r   = int((transform.f - y) / abs(transform.e))
            try:
                catch = self.grid.catchment(
                    x=col, y=r, fdir=self.fdir, xytype="index"
                )
                catch_arr = np.array(catch, dtype=np.uint8)
                geoms = [
                    shape(g)
                    for g, v in shapes(catch_arr, transform=transform)
                    if v == 1
                ]
                if geoms:
                    from shapely.ops import unary_union
                    catchments.append({
                        "outlet_id": idx,
                        "geometry": unary_union(geoms),
                        "area_m2": unary_union(geoms).area,
                    })
            except Exception as e:
                logger.warning(f"Catchment delineation failed for outlet {idx}: {e}")

        gdf = gpd.GeoDataFrame(catchments, crs=crs_obj)
        logger.success(f"Delineated {len(gdf)} catchments")

        if output_gpkg:
            self._save_gpkg(gdf, output_gpkg, layer="catchment_boundaries")

        return gdf

    # ── Export All ───────────────────────────────────────────────────────

    def export_all(
        self,
        output_dir: str | Path,
        stream_threshold: int = 1000,
        crs: str = "EPSG:32643",
    ) -> Dict[str, Path]:
        """
        Run full pipeline and export all raster (COG) and vector (GPKG) layers.

        Returns dict mapping layer name → output path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}

        with rasterio.open(self.dtm_path) as src:
            transform = src.transform
            _crs      = src.crs or CRS.from_epsg(32643)
            shape2d   = (src.height, src.width)

        def _save_raster(arr, name, band_name):
            tmp  = output_dir / f"_{name}_tmp.tif"
            cog  = output_dir / f"{name}.tif"
            write_geotiff(arr.astype(np.float32), transform, tmp, crs=_crs, band_name=band_name)
            return convert_to_cog(tmp, cog_path=cog)

        self.fill_depressions()
        self.compute_flow_direction()
        self.compute_flow_accumulation()
        self.compute_slope()
        self.compute_twi()

        paths["slope"]            = _save_raster(self.slope,                "slope",            "slope_degrees")
        paths["flow_direction"]   = _save_raster(np.array(self.fdir),       "flow_direction",   "D8_flow_direction")
        paths["flow_accumulation"]= _save_raster(np.log1p(self.acc),        "flow_accumulation","log_flow_accumulation")
        paths["twi"]              = _save_raster(self.twi,                  "twi",              "topographic_wetness_index")

        # Vector outputs → single GeoPackage
        gpkg_path = output_dir / "drainage_network.gpkg"
        streams   = self.extract_streams(threshold=stream_threshold, output_gpkg=gpkg_path)
        depressions = self.extract_depressions(output_gpkg=gpkg_path)
        catchments  = self.delineate_catchments(output_gpkg=gpkg_path)

        paths["gpkg"] = gpkg_path
        logger.success(f"All hydrological layers exported to {output_dir}")
        return paths

    # ── Private Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _strahler_order(acc_vals: np.ndarray) -> np.ndarray:
        """Approximate Strahler stream order from flow accumulation."""
        order = np.ones_like(acc_vals, dtype=int)
        order[acc_vals > 5000]  = 2
        order[acc_vals > 20000] = 3
        order[acc_vals > 80000] = 4
        return order

    @staticmethod
    def _raster_streams_to_lines(
        stream_mask: np.ndarray,
        fdir_arr: np.ndarray,
        acc_arr: np.ndarray,
        transform: Affine,
        crs_obj,
    ) -> gpd.GeoDataFrame:
        """
        Convert a binary stream raster to LineString geometries by tracing D8
        flow directions.  O(N) in stream-pixel count; replaces the former O(N²)
        greedy nearest-neighbour approach.

        Uses pysheds default dirmap (64=N, 128=NE, 1=E, 2=SE, 4=S, 8=SW,
        16=W, 32=NW).
        """
        D8 = {
            64: (-1,  0), 128: (-1,  1),
             1: ( 0,  1),   2: ( 1,  1),
             4: ( 1,  0),   8: ( 1, -1),
            16: ( 0, -1),  32: (-1, -1),
        }
        pixel_size = abs(transform.a)

        rows_s, cols_s = np.where(stream_mask == 1)
        if len(rows_s) == 0:
            return gpd.GeoDataFrame(
                {"geometry": [], "acc_value": [], "order": []}, crs=crs_obj
            )

        nrows, ncols = stream_mask.shape
        stream_cells: set[tuple[int, int]] = set(
            zip(rows_s.tolist(), cols_s.tolist())
        )

        # Build downstream map: each cell → its immediate downstream stream cell
        downstream: dict[tuple[int, int], tuple[int, int] | None] = {}
        for r, c in stream_cells:
            d = int(fdir_arr[r, c])
            if d in D8:
                dr, dc = D8[d]
                nb = (r + dr, c + dc)
                downstream[(r, c)] = nb if nb in stream_cells else None
            else:
                downstream[(r, c)] = None

        # Headwaters = cells that are never anyone else's downstream target
        downstream_targets: set[tuple[int, int]] = {
            v for v in downstream.values() if v is not None
        }
        headwaters = [c for c in stream_cells if c not in downstream_targets]

        def to_xy(r: int, c: int) -> tuple[float, float]:
            x = transform.c + c * transform.a + transform.a / 2
            y = transform.f + r * transform.e - abs(transform.e) / 2
            return (x, y)

        def strahler(v: float) -> int:
            if v > 80_000: return 4
            if v > 20_000: return 3
            if v >  5_000: return 2
            return 1

        lines: list[dict] = []
        visited: set[tuple[int, int]] = set()

        def _trace(start: tuple[int, int]) -> list[tuple[int, int]]:
            seg: list[tuple[int, int]] = []
            cur: tuple[int, int] | None = start
            # Safety cap: never visit the same cell twice in one trace
            while cur is not None and cur not in visited:
                seg.append(cur)
                visited.add(cur)
                cur = downstream.get(cur)
            # Include junction point as shared endpoint
            if cur is not None and cur in visited and seg:
                seg.append(cur)
            return seg

        def _seg_to_line(seg: list[tuple[int, int]]) -> None:
            if len(seg) < 2:
                return
            coords = [to_xy(r, c) for r, c in seg]
            seg_acc = float(np.mean([acc_arr[r, c] for r, c in seg
                                     if 0 <= r < nrows and 0 <= c < ncols]))
            line: LineString = LineString(coords)
            if pixel_size > 0:
                line = line.simplify(pixel_size * 0.5, preserve_topology=False)
            if not line.is_empty:
                lines.append({
                    "geometry": line,
                    "acc_value": seg_acc,
                    "order": strahler(seg_acc),
                })

        for hw in headwaters:
            if hw not in visited:
                _seg_to_line(_trace(hw))

        # Catch any unvisited cells (e.g. isolated loops – rare in filled D8)
        unvisited = stream_cells - visited
        while unvisited:
            start = next(iter(unvisited))
            seg = _trace(start)
            unvisited -= set(seg)
            _seg_to_line(seg)

        if not lines:
            return gpd.GeoDataFrame(
                {"geometry": [], "acc_value": [], "order": []}, crs=crs_obj
            )
        return gpd.GeoDataFrame(lines, crs=crs_obj)

    def _auto_detect_outlets(self, n: int = 5) -> gpd.GeoDataFrame:
        """Find top-N flow accumulation maxima as candidate outlets."""
        if self.acc is None:
            self.compute_flow_accumulation()

        with rasterio.open(self.dtm_path) as src:
            transform = src.transform
            crs_obj   = src.crs or CRS.from_epsg(32643)

        acc_arr = np.array(self.acc)
        flat    = acc_arr.ravel()
        top_n   = np.argsort(flat)[-n:][::-1]
        rows, cols = np.unravel_index(top_n, acc_arr.shape)

        xs = transform.c + cols * transform.a + transform.a / 2
        ys = transform.f + rows * transform.e - abs(transform.e) / 2

        return gpd.GeoDataFrame(
            {"acc_value": flat[top_n]},
            geometry=gpd.points_from_xy(xs, ys),
            crs=crs_obj,
        )

    @staticmethod
    def _save_gpkg(gdf: gpd.GeoDataFrame, gpkg_path: str | Path, layer: str):
        """Append/write a layer to a GeoPackage file."""
        gpkg_path = Path(gpkg_path)
        gpkg_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if gpkg_path.exists() else "w"
        gdf.to_file(str(gpkg_path), layer=layer, driver="GPKG")
        logger.info(f"Layer '{layer}' → {gpkg_path.name}")
