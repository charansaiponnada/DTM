"""
Generate the DTM Drainage AI pipeline architecture diagram as PNG.
"""

import graphviz
from pathlib import Path

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)

C_BG      = "#0f172a"
C_STAGE   = "#1e293b"
C_TOOL    = "#3b0764"
C_PURPOSE = "#1e1b4b"
C_INPUT   = "#166534"
C_OUTPUT  = "#7c2d12"
C_EDGE    = "#475569"
C_WHITE   = "#f8fafc"
C_GRAY    = "#94a3b8"
C_CYAN    = "#22d3ee"
C_GREEN   = "#4ade80"
C_YELLOW  = "#fbbf24"
C_ORANGE  = "#fb923c"
C_RED     = "#f87171"
C_PINK    = "#f472b6"

g = graphviz.Digraph(
    name="DTM_Drainage_AI_Pipeline",
    format="png",
    engine="dot",
)
g.attr(
    bgcolor=C_BG, pad="0.5", rankdir="TB", splines="ortho", dpi="500",
    nodesep="0.15", ranksep="0.2",
    label="DTM Drainage AI  —  Full Pipeline Architecture",
    labelloc="t", fontcolor=C_WHITE, fontname="Consolas", fontsize="18",
)
g.attr("node", fontname="Consolas", fontsize="9", shape="box",
       style="filled,rounded", color=C_EDGE, fontcolor=C_WHITE)
g.attr("edge", fontname="Consolas", fontsize="8")

STYLE_IN   = dict(fillcolor=C_INPUT)
STYLE_TOOL = dict(fillcolor=C_TOOL)
STYLE_WHY  = dict(fillcolor=C_PURPOSE)
STYLE_OUT  = dict(fillcolor=C_OUTPUT)

# ══════════════════════════════════════════════════════════════════════════
#  STAGE 1
# ══════════════════════════════════════════════════════════════════════════
with g.subgraph(name="cluster_1") as s:
    s.attr(label="Stage 1  ·  Data Inspection & Tiling",
           style="filled,rounded", fillcolor=C_STAGE, color=C_CYAN,
           fontcolor=C_CYAN, fontname="Consolas", fontsize="12")

    s.node("s1_input", "INPUT\nLAS / LAZ Point Cloud\n(.las / .laz, up to 64M pts)", **STYLE_IN)
    s.node("s1_tool", "TOOLS\nlaspy  ·  numpy  ·  tqdm\n\nReads header-only metadata\nSamples 50K pts for intensity\nSpatial tiling for memory safety", **STYLE_TOOL)
    s.node("s1_why", "WHY\nAvoid loading full 2 GB file\ninto memory. Fast metadata\nscan ensures pipeline can\nplan memory-safe processing.", **STYLE_WHY)
    s.node("s1_out", "OUTPUTS\nPointCloudMetadata\n(point_count, density, CRS,\nbounds, has_classification,\nintensity_range)\n+ tile_index.json", **STYLE_OUT)

    s.edge("s1_input", "s1_tool", color=C_EDGE)
    s.edge("s1_tool", "s1_why", color=C_EDGE)
    s.edge("s1_tool", "s1_out", color=C_EDGE, style="bold")

# ══════════════════════════════════════════════════════════════════════════
#  STAGE 2
# ══════════════════════════════════════════════════════════════════════════
with g.subgraph(name="cluster_2") as s:
    s.attr(label="Stage 2  ·  Ground Classification",
           style="filled,rounded", fillcolor=C_STAGE, color=C_GREEN,
           fontcolor=C_GREEN, fontname="Consolas", fontsize="12")

    s.node("s2_in", "INPUT\nTile LAS files\n(from Stage 1)", **STYLE_IN)
    s.node("s2_smrf", "STEP 1: SMRF Filter\npdal.exe (filters.smrf)\n\nSlope=0.15  Window=18.0\nThreshold=0.5  Scalar=1.25\n\nMorphological filter:\nrolling ball segments ground\nfrom non-ground returns.", **STYLE_TOOL)
    s.node("s2_rf", "STEP 2: ML Refinement (optional)\nRandomForestClassifier (scikit-learn)\n\n12 per-point geometric features:\nPCA eigenvalues (planarity,\nlinearity, sphericity, anisotropy,\nownivariance, surface_variation,\nverticality) + density + z-range", **STYLE_TOOL)
    s.node("s2_why", "WHY\nSMRF is fast on flat Gujarat\nterrain (<5 deg slopes). Random\nForest fixes SMRF boundary\nerrors using per-point PCA\nfeatures for higher accuracy.", **STYLE_WHY)
    s.node("s2_out", "OUTPUTS\nclassified_ground.las\n(LAS 1.4, class 2 = ground)\n+ optional RF model .joblib", **STYLE_OUT)

    s.edge("s2_in", "s2_smrf", color=C_EDGE)
    s.edge("s2_smrf", "s2_rf", color=C_EDGE, label="optional", fontcolor=C_YELLOW)
    s.edge("s2_smrf", "s2_why", color=C_EDGE)
    s.edge("s2_rf", "s2_out", color=C_EDGE, style="bold")

# ══════════════════════════════════════════════════════════════════════════
#  STAGE 3
# ══════════════════════════════════════════════════════════════════════════
with g.subgraph(name="cluster_3") as s:
    s.attr(label="Stage 3  ·  DTM Generation & Terrain Derivatives",
           style="filled,rounded", fillcolor=C_STAGE, color=C_YELLOW,
           fontcolor=C_YELLOW, fontname="Consolas", fontsize="12")

    s.node("s3_in", "INPUT\nclassified_ground.las\n(ground points only, class 2)", **STYLE_IN)
    s.node("s3_idw", "STEP 1: IDW Interpolation\nscipy.spatial.cKDTree\n\nk=16 nearest neighbours\nRadius=5 m  Power=2\nBatch size=100K cells\nGaussian smoothing (sigma=1.5)", **STYLE_TOOL)
    s.node("s3_cog", "STEP 2: COG Conversion\nrio-cogeo (cog_translate)\nrasterio\n\nOverview levels: 2,4,8,16\nDEFLATE compression\nCRS: EPSG:32643 (UTM 43N)", **STYLE_TOOL)
    s.node("s3_deriv", "STEP 3: Terrain Derivatives\nnumpy.gradient  +  scipy.ndimage\nsrc.features (Evans curvature)\n\nSlope (gradient+arctan)\nAspect (arctan2)\nCurvature (Evans 1979 method)\nTPI (uniform_filter w=15&51)\nRoughness  +  Hillshade", **STYLE_TOOL)
    s.node("s3_why", "WHY\nIDW is simple, fast, and works\nwell on dense ground points.\nCOG format enables cloud-\noptimised streaming. Derivatives\nfeed hydrology + ML models.", **STYLE_WHY)
    s.node("s3_out", "OUTPUTS\ndtm.tif (COG, 0.5 m)\nslope.tif  +  aspect.tif\nplan_curvature.tif\nprofile_curvature.tif\ntpi_15.tif  +  tpi_51.tif\nroughness.tif  +  hillshade.tif", **STYLE_OUT)

    s.edge("s3_in", "s3_idw", color=C_EDGE)
    s.edge("s3_idw", "s3_cog", color=C_EDGE)
    s.edge("s3_cog", "s3_deriv", color=C_EDGE)
    s.edge("s3_idw", "s3_why", color=C_EDGE)
    s.edge("s3_deriv", "s3_out", color=C_EDGE, style="bold")

# ══════════════════════════════════════════════════════════════════════════
#  STAGE 4
# ══════════════════════════════════════════════════════════════════════════
with g.subgraph(name="cluster_4") as s:
    s.attr(label="Stage 4  ·  Hydrological Analysis",
           style="filled,rounded", fillcolor=C_STAGE, color=C_ORANGE,
           fontcolor=C_ORANGE, fontname="Consolas", fontsize="12")

    s.node("s4_in", "INPUT\ndtm.tif (COG)  +  slope.tif", **STYLE_IN)
    s.node("s4_fill", "STEP 1: Fill Depressions\npysheds.Grid\n\nWang & Liu algorithm\nfill_depressions() + resolve_flats()\n\nRemoves artificial sinks so\nwater can flow continuously\nto catchment outlet.", **STYLE_TOOL)
    s.node("s4_flow", "STEP 2: Flow Direction & Accumulation\npysheds.Grid\n\nD8 single-flow direction\nflowdir() -> accumulation()\n\nEach cell flows to steepest\ndownhill neighbour. Accumulation\ncounts upstream contributing cells.", **STYLE_TOOL)
    s.node("s4_twi", "STEP 3: TWI Computation\nnumpy\n\nTWI = ln(alpha / tan(beta))\nalpha = specific catchment area\nbeta = slope in radians\n\nHigh TWI = convergence-prone\n= potential waterlogging areas.", **STYLE_TOOL)
    s.node("s4_strm", "STEP 4: Stream Extraction\nnumpy + geopandas + shapely\n\nThreshold accumulation >1000\nD8 tracing -> LineStrings\nVectorise depressions &\ncatchment boundaries to GPKG.", **STYLE_TOOL)
    s.node("s4_why", "WHY\nPysheds is pure-Python and\nfast for moderate DEMs. D8 is\nstandard for flat terrain.\nTWI is best single predictor\nof water accumulation.", **STYLE_WHY)
    s.node("s4_out", "OUTPUTS\nflow_direction.tif (COG)\nflow_accumulation.tif (COG)\ntwi.tif (COG)\ndrainage_network.gpkg layers:\n  - drainage_channels\n  - depression_polygons\n  - catchment_boundaries", **STYLE_OUT)

    s.edge("s4_in", "s4_fill", color=C_EDGE)
    s.edge("s4_fill", "s4_flow", color=C_EDGE)
    s.edge("s4_flow", "s4_twi", color=C_EDGE)
    s.edge("s4_twi", "s4_strm", color=C_EDGE)
    s.edge("s4_flow", "s4_why", color=C_EDGE)
    s.edge("s4_strm", "s4_out", color=C_EDGE, style="bold")

# ══════════════════════════════════════════════════════════════════════════
#  STAGE 5
# ══════════════════════════════════════════════════════════════════════════
with g.subgraph(name="cluster_5") as s:
    s.attr(label="Stage 5  ·  Waterlogging Prediction (XGBoost)",
           style="filled,rounded", fillcolor=C_STAGE, color=C_RED,
           fontcolor=C_RED, fontname="Consolas", fontsize="12")

    s.node("s5_in", "INPUTS\ndtm.tif  +  twi.tif\nflow_accumulation.tif\nslope.tif", **STYLE_IN)
    s.node("s5_feat", "STEP 1: Feature Stack\nsrc.features + scipy.ndimage\n\n10 features per pixel:\nElevation_norm, Slope_deg,\nAspect (sin+cos), TWI, TPI,\nLog(flow_accumulation),\nPlan+Profile curvature,\nDepression_depth, Stream_dist\n-> array shape (H, W, 10)", **STYLE_TOOL)
    s.node("s5_label", "STEP 2: Pseudo-Labels\nnumpy\n\nRule-based heuristic:\n(TWI>=8 and TPI<=-0.3\nand Slope<=2 deg) OR\nFlow_acc >= 85th percentile\n\nNo surveyed ground truth\nlabels available.", **STYLE_TOOL)
    s.node("s5_xgb", "STEP 3: XGBoost Training\nxgboost (XGBClassifier)\nscikit-learn\n\nhist tree method\nEvaluation metric: AUC-PR\nscale_pos_weight=5\nRobustScaler normalisation\nStratifiedKFold CV (5 folds)", **STYLE_TOOL)
    s.node("s5_pred", "STEP 4: Predict & Export\nnumpy + rasterio + geopandas\n\npredict_proba() -> (H,W) map\nThreshold 0.3/0.5/0.7 for\nLOW/MEDIUM/HIGH risk zones\nVectorise hotspots to GPKG\nSave model to joblib.", **STYLE_TOOL)
    s.node("s5_why", "WHY\nXGBoost dominates tabular/\npixel-wise ML. RobustScaler\nhandles outlier features.\nPseudo-labels let us train\nwithout expensive surveyed\nwaterlogging data.", **STYLE_WHY)
    s.node("s5_out", "OUTPUTS\nwaterlogging_probability.tif\n  (COG, probability 0-1)\nwaterlogging_hotspots layer\n  in drainage_network.gpkg\n  (risk_level: LOW/MED/HIGH)\nmodels/waterlogging_xgb.joblib", **STYLE_OUT)

    s.edge("s5_in", "s5_feat", color=C_EDGE)
    s.edge("s5_feat", "s5_label", color=C_EDGE)
    s.edge("s5_label", "s5_xgb", color=C_EDGE)
    s.edge("s5_xgb", "s5_pred", color=C_EDGE)
    s.edge("s5_xgb", "s5_why", color=C_EDGE)
    s.edge("s5_pred", "s5_out", color=C_EDGE, style="bold")

# ══════════════════════════════════════════════════════════════════════════
#  STAGE 6
# ══════════════════════════════════════════════════════════════════════════
with g.subgraph(name="cluster_6") as s:
    s.attr(label="Stage 6  ·  Drainage Network Design",
           style="filled,rounded", fillcolor=C_STAGE, color=C_PINK,
           fontcolor=C_PINK, fontname="Consolas", fontsize="12")

    s.node("s6_in", "INPUTS\ndtm.tif + drainage_network.gpkg\n(streams + hotspots layers)", **STYLE_IN)
    s.node("s6_graph", "STEP 1: Build Flow Graph\nnetworkx.DiGraph\n\nStream segments -> directed\ngraph. Edge weight =\nlength x cost_per_metre.\nConnect new channels to\nexisting stream network.", **STYLE_TOOL)
    s.node("s6_mst", "STEP 2: MST Optimisation\nnetworkx.minimum_spanning_tree\n\nFinds minimum-cost set of\nchannels that connects all\nwaterlogging hotspots to\nnatural outlet streams.", **STYLE_TOOL)
    s.node("s6_hyd", "STEP 3: Hydraulic Design\nRational Method + Manning's eqn\n\nQ = C x i x A / 360  (cumecs)\nC=runoff coeff  i=rainfall mm/hr\nA=catchment area\n\nTrapezoidal channel sizing:\nQ = (1/n) A R^(2/3) S^(1/2)\nIterative solver for depth/width.", **STYLE_TOOL)
    s.node("s6_cost", "STEP 4: Cost Estimation\nnumpy + pandas\n\nEarthen: ~Rs1,200/m\nPipe: ~Rs3,500/m\nTotal project cost with\nbreakdown per segment.", **STYLE_TOOL)
    s.node("s6_why", "WHY\nMST guarantees cost-minimal\nnetwork connecting all\nhotspots. Rational Method\nis standard Indian drainage\ndesign code. Manning's\nequation sizes each channel.", **STYLE_WHY)
    s.node("s6_out", "OUTPUTS\ndrainage_network.gpkg:\n  - drainage_channels\n    (bottom_width_m, depth_m,\n     velocity_ms, cost_inr,\n     channel_type, q_design)\n  - design_summary\n    (total_length_m,\n     total_cost_inr,\n     num_segments)", **STYLE_OUT)

    s.edge("s6_in", "s6_graph", color=C_EDGE)
    s.edge("s6_graph", "s6_mst", color=C_EDGE)
    s.edge("s6_mst", "s6_hyd", color=C_EDGE)
    s.edge("s6_hyd", "s6_cost", color=C_EDGE)
    s.edge("s6_mst", "s6_why", color=C_EDGE)
    s.edge("s6_cost", "s6_out", color=C_EDGE, style="bold")

# ══════════════════════════════════════════════════════════════════════════
#  EVALUATION cluster
# ══════════════════════════════════════════════════════════════════════════
with g.subgraph(name="cluster_eval") as s:
    s.attr(label="Evaluation  ·  Accuracy Metrics",
           style="filled,rounded", fillcolor="#0c0a1d", color=C_GRAY,
           fontcolor=C_GRAY, fontname="Consolas", fontsize="12")

    s.node("eval_ground", "Ground Classification\nAccuracy + F1 score\n(heuristic proxy)", **STYLE_TOOL)
    s.node("eval_dtm", "DTM Accuracy\nRMSE + MAE + LE90\n(flat-plane check)", **STYLE_TOOL)
    s.node("eval_water", "Waterlogging Model\nAUC-ROC + F1 + Brier\n(5-fold cross-val)", **STYLE_TOOL)
    s.node("eval_drain", "Drainage Design\nChannel coverage + Cost\nHydraulic validation", **STYLE_TOOL)

g.edge("s2_out", "eval_ground", color=C_GRAY, style="dashed", label="  eval")
g.edge("s3_out", "eval_dtm", color=C_GRAY, style="dashed", label="  eval")
g.edge("s5_out", "eval_water", color=C_GRAY, style="dashed", label="  eval")
g.edge("s6_out", "eval_drain", color=C_GRAY, style="dashed", label="  eval")

# ══════════════════════════════════════════════════════════════════════════
#  Inter-stage edges
# ══════════════════════════════════════════════════════════════════════════
g.edge("s1_out", "s2_in", color=C_CYAN, style="bold", label="tiles + metadata", fontcolor=C_GRAY)
g.edge("s2_out", "s3_in", color=C_GREEN, style="bold", label="classified LAS", fontcolor=C_GRAY)
g.edge("s3_out", "s4_in", color=C_YELLOW, style="bold", label="DTM + slope COGs", fontcolor=C_GRAY)
g.edge("s4_out", "s5_in", color=C_ORANGE, style="bold", label="TWI + accum + GPKG", fontcolor=C_GRAY)
g.edge("s5_out", "s6_in", color=C_RED, style="bold", label="hotspots + streams", fontcolor=C_GRAY)

# ── Render ──────────────────────────────────────────────────────────────────
png_path = OUT / "pipeline_architecture.png"
g.render(str(png_path.with_suffix("")), cleanup=True)
print(f"Saved -> {png_path}")
print(f"Size  -> {png_path.stat().st_size:,} bytes")
