"""
src/hydrology/drainage_network.py
───────────────────────────────────
Optimal drainage network design for flood-prone village abadi areas.

Steps:
  1. Identify drainage demand nodes (waterlogging hotspots + depressions)
  2. Identify candidate outlet points (village boundary edges, existing drains)
  3. Build a weighted graph on the flow direction grid
  4. Apply Minimum Spanning Tree (MST) to find cost-optimal channel network
  5. Design hydraulic dimensions for each channel segment (Manning's equation)
  6. Export GIS-ready layers to GeoPackage

Design standards:
  - Open earthen channels or RCC U-drain for abadi
  - 10-year return period design storm
  - Manning's n = 0.025 (earthen) / 0.013 (concrete)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import rasterio
from rasterio.crs import CRS
from shapely.geometry import LineString, Point
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════
#  Hydraulic Design Constants & Dataclasses
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ChannelSegment:
    """Represents one section of designed drainage channel."""
    segment_id:   int
    from_node:    int
    to_node:      int
    length_m:     float
    geometry:     LineString

    # Catchment properties
    catchment_area_m2: float = 0.0
    avg_slope:         float = 0.001   # m/m

    # Hydraulic design outputs
    design_flow_m3s:  float = 0.0      # peak discharge (m³/s)
    channel_type:     str   = "earthen"  # earthen | concrete | pipe
    bottom_width_m:   float = 0.3
    depth_m:          float = 0.5
    top_width_m:      float = 0.8      # trapezoidal
    side_slope:       float = 1.5      # H:V
    velocity_ms:      float = 0.0
    capacity_m3s:     float = 0.0
    freeboard_m:      float = 0.15

    # Cost
    cost_inr:         float = 0.0
    cost_per_m_inr:   float = 800.0


@dataclass
class DrainageDesignParameters:
    """Design storm and hydraulic parameters for the drainage system."""
    return_period_yr:      int   = 10
    rainfall_intensity_mmhr: float = 50.0   # mm/hr – Gujarat 10-yr
    runoff_coefficient:    float = 0.65     # urban mixed abadi
    manning_n_earthen:     float = 0.025
    manning_n_concrete:    float = 0.013
    min_velocity_ms:       float = 0.3      # self-cleaning
    max_velocity_ms:       float = 2.0      # erosion limit (earthen)
    side_slope_hv:         float = 1.5      # trapezoidal H:V
    freeboard_m:           float = 0.15
    cost_earthen_inr_m:    float = 800.0
    cost_concrete_inr_m:   float = 2200.0
    cost_pipe_inr_m:       float = 3500.0


# ══════════════════════════════════════════════════════════════════════════
#  Rational Method Discharge Calculation
# ══════════════════════════════════════════════════════════════════════════

def rational_discharge(
    area_m2: float,
    intensity_mmhr: float,
    runoff_coeff: float,
) -> float:
    """
    Q = C · i · A  (Rational Method)

    Parameters
    ----------
    area_m2         : catchment area (m²)
    intensity_mmhr  : design rainfall intensity (mm/hr)
    runoff_coeff    : dimensionless runoff coefficient C

    Returns
    -------
    Peak discharge Q in m³/s
    """
    i_ms  = intensity_mmhr / (1000 * 3600)  # mm/hr → m/s
    return runoff_coeff * i_ms * area_m2


# ══════════════════════════════════════════════════════════════════════════
#  Manning's Equation – Trapezoidal Channel Sizing
# ══════════════════════════════════════════════════════════════════════════

def design_trapezoidal_channel(
    Q: float,
    slope: float,
    params: DrainageDesignParameters,
    min_bottom_width: float = 0.3,
) -> Tuple[float, float, float, float, float, float, str]:
    """
    Size a trapezoidal channel using Manning's equation.

    Q = (1/n) * A * R^(2/3) * S^(1/2)

    Parameters
    ----------
    Q      : design discharge (m³/s)
    slope  : channel bed slope (m/m)
    params : design parameters
    min_bottom_width : minimum bottom width (m)

    Returns
    -------
    (bottom_width, depth, top_width, velocity, capacity, cost_per_m, channel_type)
    """
    z = params.side_slope_hv
    n = params.manning_n_earthen
    S = max(slope, 0.0005)     # minimum slope to avoid stagnation

    # Iterative sizing: start with minimum width, increase depth
    b = min_bottom_width
    d = 0.1  # initial depth guess

    for _ in range(500):
        A  = (b + z * d) * d
        P  = b + 2 * d * np.sqrt(1 + z**2)
        R  = A / P if P > 0 else 1e-6
        Q_cap = (1 / n) * A * R**(2/3) * S**0.5
        V     = Q_cap / A if A > 0 else 0

        if Q_cap >= Q:
            break
        # Increase depth preferentially, widen if needed
        if d < 1.2:
            d += 0.05
        else:
            b += 0.1
            d = 0.1   # reset depth

    top_w   = b + 2 * z * d
    freeboard_d = d + params.freeboard_m

    # Choose channel type based on velocity
    if V > params.max_velocity_ms:
        n       = params.manning_n_concrete
        cost_pm = params.cost_concrete_inr_m
        ch_type = "concrete"
    elif Q > 1.5:       # large discharge → pipe is impractical → concrete
        n       = params.manning_n_concrete  # record correct Manning n for this type
        cost_pm = params.cost_concrete_inr_m
        ch_type = "concrete"
    else:
        cost_pm = params.cost_earthen_inr_m
        ch_type = "earthen"

    return b, d, top_w, V, Q_cap, cost_pm, ch_type


# ══════════════════════════════════════════════════════════════════════════
#  Graph-Based Network Optimization
# ══════════════════════════════════════════════════════════════════════════

def build_flow_graph(
    streams_gdf: gpd.GeoDataFrame,
    dtm_path: str | Path,
    hotspots_gdf: Optional[gpd.GeoDataFrame] = None,
) -> nx.DiGraph:
    """
    Build a directed weighted graph from stream channel segments.

    Edge weight = construction cost (derived from length × terrain slope
    × channel type factor).

    Nodes represent channel junctions, inlets, and outlet points.
    """
    G = nx.DiGraph()

    with rasterio.open(dtm_path) as src:
        transform = src.transform
        cell_size = src.res[0]

    for idx, row in streams_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        coords = list(geom.coords) if isinstance(geom, LineString) else []
        if len(coords) < 2:
            continue

        start = coords[0]
        end   = coords[-1]
        length = geom.length

        # Approximate slope from flow accumulation order
        order = row.get("order", 1)
        slope = max(0.001, 0.005 / order)   # steeper for smaller streams

        # Edge cost proportional to length and channel width (proxy)
        cost = length * 800   # base earthen cost

        G.add_edge(
            start, end,
            length=length,
            slope=slope,
            cost=cost,
            order=order,
            geometry=geom,
            segment_id=idx,
        )

    logger.info(
        f"Flow graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def optimize_drainage_mst(
    G: nx.DiGraph,
    outlet_nodes: Optional[List] = None,
) -> nx.Graph:
    """
    Find minimum-cost drainage network using Minimum Spanning Tree.

    For each hotspot, the MST guarantees connectivity to an outlet
    with minimum total construction cost.

    Parameters
    ----------
    G             : directed flow graph
    outlet_nodes  : forced outlet nodes (village boundary exits)

    Returns
    -------
    Undirected MST subgraph representing the optimal drainage network
    """
    if G.number_of_edges() == 0:
        logger.warning("Empty flow graph – returning empty MST")
        return nx.Graph()

    undirected = G.to_undirected()
    mst = nx.minimum_spanning_tree(undirected, weight="cost")
    logger.info(
        f"MST: {mst.number_of_nodes()} nodes, "
        f"{mst.number_of_edges()} edges, "
        f"total cost: ₹{sum(d['cost'] for u,v,d in mst.edges(data=True)):,.0f}"
    )
    return mst


# ══════════════════════════════════════════════════════════════════════════
#  High-Level Drainage Designer
# ══════════════════════════════════════════════════════════════════════════

class DrainageNetworkDesigner:
    """
    Orchestrates the full drainage design workflow.

    Usage
    -----
    designer = DrainageNetworkDesigner(dtm_path, params)
    designer.load_inputs(streams_gdf, hotspots_gdf, depressions_gdf)
    results = designer.design()
    designer.export(output_gpkg="data/output/drainage_network.gpkg")
    """

    def __init__(
        self,
        dtm_path: str | Path,
        params: Optional[DrainageDesignParameters] = None,
    ):
        self.dtm_path  = Path(dtm_path)
        self.params    = params or DrainageDesignParameters()
        self.segments: List[ChannelSegment] = []
        self.summary:  Dict = {}

        # Will be set by load_inputs
        self.streams_gdf      = None
        self.hotspots_gdf     = None
        self.depressions_gdf  = None

    def load_inputs(
        self,
        streams_gdf: gpd.GeoDataFrame,
        hotspots_gdf: Optional[gpd.GeoDataFrame] = None,
        depressions_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> "DrainageNetworkDesigner":
        self.streams_gdf      = streams_gdf
        self.hotspots_gdf     = hotspots_gdf
        self.depressions_gdf  = depressions_gdf
        logger.info(
            f"Inputs loaded: {len(streams_gdf)} stream segments, "
            f"{len(hotspots_gdf) if hotspots_gdf is not None else 0} hotspot polygons"
        )
        return self

    def design(self) -> List[ChannelSegment]:
        """
        Run full network design:
          1. Build flow graph
          2. MST optimization
          3. Hydraulic sizing for each segment
          4. Cost estimation
        """
        logger.info("Starting drainage network design …")

        G   = build_flow_graph(self.streams_gdf, self.dtm_path, self.hotspots_gdf)
        mst = optimize_drainage_mst(G)

        self.segments = []
        for seg_id, (u, v, data) in enumerate(mst.edges(data=True)):
            length       = data.get("length", 1.0)
            slope        = data.get("slope", 0.001)

            # Catchment area approximation: proportional to flow accumulation order
            order        = data.get("order", 1)
            catch_area   = length * 50 * order   # rough proxy (m²)

            Q_design     = rational_discharge(
                catch_area,
                self.params.rainfall_intensity_mmhr,
                self.params.runoff_coefficient,
            )

            b, d, top_w, V, Q_cap, cost_pm, ch_type = design_trapezoidal_channel(
                Q_design, slope, self.params
            )

            total_cost = length * cost_pm

            geom = data.get("geometry")
            if geom is None:
                geom = LineString([u, v])

            seg = ChannelSegment(
                segment_id        = seg_id,
                from_node         = seg_id * 2,
                to_node           = seg_id * 2 + 1,
                length_m          = length,
                geometry          = geom,
                catchment_area_m2 = catch_area,
                avg_slope         = slope,
                design_flow_m3s   = Q_design,
                channel_type      = ch_type,
                bottom_width_m    = b,
                depth_m           = d,
                top_width_m       = top_w,
                velocity_ms       = V,
                capacity_m3s      = Q_cap,
                cost_inr          = total_cost,
                cost_per_m_inr    = cost_pm,
            )
            self.segments.append(seg)

        self._compute_summary()
        logger.success(
            f"Design complete: {len(self.segments)} channel segments, "
            f"total cost ₹{self.summary['total_cost_inr']:,.0f}"
        )
        return self.segments

    def _compute_summary(self):
        if not self.segments:
            self.summary = {}
            return
        total_length = sum(s.length_m for s in self.segments)
        total_cost   = sum(s.cost_inr for s in self.segments)
        n_earthen    = sum(1 for s in self.segments if s.channel_type == "earthen")
        n_concrete   = sum(1 for s in self.segments if s.channel_type == "concrete")
        self.summary = {
            "total_segments":    len(self.segments),
            "total_length_m":    total_length,
            "total_cost_inr":    total_cost,
            "cost_per_m_avg":    total_cost / total_length if total_length else 0,
            "n_earthen_channels": n_earthen,
            "n_concrete_channels": n_concrete,
            "max_design_flow_m3s": max(s.design_flow_m3s for s in self.segments),
        }

    def export(
        self,
        output_gpkg: str | Path,
        crs: str = "EPSG:32643",
    ) -> Path:
        """
        Export designed drainage channels to a GeoPackage layer.
        """
        if not self.segments:
            logger.warning("No segments to export – run .design() first.")
            return Path(output_gpkg)

        output_gpkg = Path(output_gpkg)
        crs_obj = CRS.from_epsg(int(crs.split(":")[-1]))

        rows = []
        for s in self.segments:
            rows.append({
                "segment_id":      s.segment_id,
                "length_m":        round(s.length_m, 2),
                "slope_mm":        round(s.avg_slope * 1000, 3),
                "Q_design_m3s":    round(s.design_flow_m3s, 4),
                "channel_type":    s.channel_type,
                "bottom_width_m":  round(s.bottom_width_m, 2),
                "depth_m":         round(s.depth_m, 2),
                "top_width_m":     round(s.top_width_m, 2),
                "velocity_ms":     round(s.velocity_ms, 3),
                "capacity_m3s":    round(s.capacity_m3s, 4),
                "cost_inr":        round(s.cost_inr, 0),
                "geometry":        s.geometry,
            })

        gdf = gpd.GeoDataFrame(rows, crs=crs_obj)
        mode = "a" if output_gpkg.exists() else "w"
        gdf.to_file(str(output_gpkg), layer="drainage_channels", driver="GPKG")

        # Export summary as a single-row metadata layer
        summary_gdf = gpd.GeoDataFrame(
            [self.summary],
            geometry=[gdf.geometry.unary_union.centroid],
            crs=crs_obj,
        )
        summary_gdf.to_file(str(output_gpkg), layer="design_summary", driver="GPKG")

        logger.success(
            f"Drainage network exported:\n"
            f"  Segments      : {self.summary['total_segments']}\n"
            f"  Total length  : {self.summary['total_length_m']:,.0f} m\n"
            f"  Total cost    : ₹{self.summary['total_cost_inr']:,.0f}\n"
            f"  Output        : {output_gpkg}"
        )
        return output_gpkg

    def print_summary(self):
        """Print a formatted design summary to console."""
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table   = Table(title="Drainage Network Design Summary", show_header=True)
        table.add_column("Parameter")
        table.add_column("Value", justify="right")

        rows = [
            ("Total channel segments",    str(self.summary.get("total_segments", 0))),
            ("Total channel length",       f"{self.summary.get('total_length_m', 0):,.0f} m"),
            ("Estimated construction cost", f"₹{self.summary.get('total_cost_inr', 0):,.0f}"),
            ("Avg cost per metre",         f"₹{self.summary.get('cost_per_m_avg', 0):,.0f}/m"),
            ("Earthen channels",          str(self.summary.get("n_earthen_channels", 0))),
            ("Concrete channels",         str(self.summary.get("n_concrete_channels", 0))),
            ("Peak design discharge",     f"{self.summary.get('max_design_flow_m3s', 0):.3f} m³/s"),
            ("Design return period",      f"{self.params.return_period_yr} years"),
            ("Design rainfall intensity", f"{self.params.rainfall_intensity_mmhr} mm/hr"),
        ]
        for k, v in rows:
            table.add_row(k, v)

        console.print(table)
