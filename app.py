"""
DTM Drainage AI - Streamlit Web App
====================================
MoPR Geospatial Intelligence Hackathon
A web interface for processing drone point cloud data to generate DTM and drainage networks.
"""

import streamlit as st
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

st.set_page_config(
    page_title="DTM Drainage AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    .card { background: #f8f9fa; border-radius: 10px; padding: 1.5rem; margin: 0.5rem 0; border: 1px solid #e9ecef; }
    .metric-card { background: white; border-radius: 8px; padding: 1rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #2c5282; }
    .metric-label { font-size: 0.85rem; color: #718096; }
    .pipeline-step { display: flex; align-items: center; padding: 0.75rem; margin: 0.5rem 0; background: white; border-radius: 8px; border-left: 4px solid #4299e1; }
    .step-number { background: #4299e1; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-right: 1rem; }
    .output-item { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: white; border-radius: 6px; margin: 0.25rem 0; border: 1px solid #e2e8f0; }
    .viz-container { background: white; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_SCRIPT = "run_pipeline.py"

STAGES = {
    1: {
        "name": "Data Inspection",
        "icon": "🔍",
        "desc": "Analyze point cloud metadata, density, CRS",
    },
    2: {
        "name": "Ground Classification",
        "icon": "🏗️",
        "desc": "SMRF + ML to separate ground from objects",
    },
    3: {
        "name": "DTM Generation",
        "icon": "📐",
        "desc": "IDW interpolation to create terrain model",
    },
    4: {
        "name": "Hydrological Analysis",
        "icon": "🌊",
        "desc": "Flow direction, accumulation, TWI",
    },
    5: {
        "name": "Waterlogging Prediction",
        "icon": "⚠️",
        "desc": "XGBoost model for flood risk zones",
    },
    6: {
        "name": "Drainage Network",
        "icon": "🛣️",
        "desc": "MST + Manning's for optimal drains",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data
def load_raster(path):
    """Load raster and return data, transform, extent."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1)
        extent = src.bounds
        crs = src.crs
        transform = src.transform
    return data, extent, crs, transform


@st.cache_data
def load_geopackage(path, layer=None):
    """Load GeoPackage and return GeoDataFrame."""
    import geopandas as gpd

    gdf = gpd.read_file(path, layer=layer)
    return gdf


def colorize_elevation(data, nodata=-9999):
    """Create RGB image from elevation data."""
    data = data.astype(np.float32)
    data[data == nodata] = np.nan

    vmin, vmax = np.nanpercentile(data, (2, 98))
    if vmax == vmin:
        vmax = vmin + 1

    normalized = (data - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Multi-color colormap: dark blue -> cyan -> green -> yellow -> red
    colors = np.array(
        [
            [20, 20, 120],  # Dark blue (low)
            [40, 100, 180],  # Blue
            [0, 180, 100],  # Cyan-green
            [100, 220, 50],  # Green
            [220, 220, 0],  # Yellow
            [255, 100, 0],  # Orange
            [180, 20, 20],  # Red (high)
        ],
        dtype=np.float32,
    )

    n_colors = len(colors)
    idx = normalized * (n_colors - 1)
    idx_floor = np.floor(idx).astype(int)
    idx_ceil = np.ceil(idx).astype(int)
    idx_floor = np.clip(idx_floor, 0, n_colors - 1)
    idx_ceil = np.clip(idx_ceil, 0, n_colors - 1)

    t = (idx - idx_floor).reshape(-1, 1)
    rgb = colors[idx_floor.flatten()] * (1 - t) + colors[idx_ceil.flatten()] * t
    rgb = rgb.reshape(data.shape + (3,))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Set nodata to transparent
    rgb[np.isnan(data)] = [0, 0, 0]

    return rgb, vmin, vmax


def colorize_risk(data, nodata=-9999, threshold=0.45):
    """Create RGB image from waterlogging probability."""
    data = data.astype(np.float32)
    data[data == nodata] = np.nan

    data = np.clip(data, 0, 1)

    # Blue gradient: low (white/light blue) -> high (dark blue)
    colors = np.array(
        [
            [255, 255, 255],  # White (0)
            [200, 230, 255],  # Light blue (0.25)
            [100, 180, 255],  # Medium blue (0.5)
            [30, 100, 220],  # Blue (0.75)
            [10, 50, 180],  # Dark blue (1.0)
        ],
        dtype=np.float32,
    )

    n_colors = len(colors)
    idx = data * (n_colors - 1)
    idx_floor = np.floor(idx).astype(int)
    idx_ceil = np.ceil(idx).astype(int)
    idx_floor = np.clip(idx_floor, 0, n_colors - 1)
    idx_ceil = np.clip(idx_ceil, 0, n_colors - 1)

    t = (idx - idx_floor).reshape(-1, 1)
    rgb = colors[idx_floor.flatten()] * (1 - t) + colors[idx_ceil.flatten()] * t
    rgb = rgb.reshape(data.shape + (3,))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    rgb[np.isnan(data)] = [240, 240, 240]  # Gray for nodata

    return rgb


def create_dtm_legend(vmin, vmax):
    """Create HTML legend for DTM."""
    return f"""
    <div style="background:white; padding:10px; border-radius:5px; border:1px solid #ccc; margin-top:10px;">
    <b>Elevation Legend</b>
    <div style="background:linear-gradient(to right, rgb(20,20,120), rgb(40,100,180), rgb(0,180,100), rgb(100,220,50), rgb(220,220,0), rgb(255,100,0), rgb(180,20,20)); height:15px; width:100%;"></div>
    <div style="display:flex; justify-content:space-between; font-size:11px; margin-top:3px;">
        <span>{vmin:.1f}m</span><span>{vmax:.1f}m</span>
    </div>
    </div>
    """


def create_risk_legend():
    return """<div style="background:white; padding:10px; border-radius:5px; border:1px solid #ccc; margin-top:10px;"><b>Waterlogging Risk</b><div style="background:linear-gradient(to right, white, rgb(200,230,255), rgb(100,180,255), rgb(30,100,220), rgb(10,50,180)); height:15px; width:100%;"></div><div style="display:flex; justify-content:space-between; font-size:11px; margin-top:3px;"><span>Low</span><span>High</span></div></div>"""


def get_centroid(gdf):
    """Get centroid of GeoDataFrame."""
    return gdf.geometry.centroid.iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    input_file: str,
    output_dir: str,
    stages: str,
    stream_threshold: int,
    resolution: float,
    use_ml: bool,
):
    cmd = [
        sys.executable,
        PIPELINE_SCRIPT,
        "--input",
        input_file,
        "--output",
        output_dir,
        "--stages",
        stages,
        "--stream-threshold",
        str(stream_threshold),
        "--resolution",
        str(resolution),
    ]
    if not use_ml:
        cmd.append("--no-ml")

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def load_metrics(output_dir: str) -> dict:
    metrics_path = Path(output_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def get_available_files():
    input_dir = Path("data/input")
    if input_dir.exists():
        return list(input_dir.glob("*.las")) + list(input_dir.glob("*.laz"))
    return []


def get_output_files(output_dir: Path):
    if not output_dir.exists():
        return []
    files = []
    for ext in ["*.tif", "*.gpkg", "*.las", "*.json"]:
        files.extend(output_dir.glob(ext))
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")

    selected_stages = st.multiselect(
        "Select Stages",
        options=list(STAGES.keys()),
        default=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: f"{STAGES[x]['icon']} {STAGES[x]['name']}",
    )

    st.markdown("### Parameters")
    stream_threshold = st.slider(
        "Stream Threshold",
        100,
        5000,
        1000,
        50,
        help="Min flow accumulation to define streams",
    )
    resolution = st.slider("DTM Resolution (m)", 0.1, 2.0, 0.5, 0.1)
    use_ml = st.checkbox(
        "Use ML Refinement", value=True, help="Better ground classification with ML"
    )

    st.markdown("---")
    st.markdown("""
    **Tips:**
    - Processing large files may take 10-30 minutes
    - Use `--no-ml` for faster processing
    - View results in QGIS for full visualization
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🗺️ DTM Drainage AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">MoPR Geospatial Intelligence Hackathon - Generate DTM and Drainage Network from Drone Point Cloud</p>',
    unsafe_allow_html=True,
)

# ─── Input Section ───
st.markdown("## 📁 Input Data")

col1, col2 = st.columns([2, 1])

with col1:
    available_files = get_available_files()

    if available_files:
        file_options = {f.name: str(f) for f in available_files}
        selected_file = st.selectbox(
            "Select Point Cloud File",
            options=list(file_options.keys()),
            help="Choose from available files",
        )
        input_path = file_options[selected_file]
        file_size = Path(input_path).stat().st_size / (1024**3)
        st.info(f"📊 File size: {file_size:.2f} GB")
    else:
        input_path = st.text_input(
            "Enter LAS/LAZ File Path",
            placeholder="C:\\path\\to\\your\\file.las",
            help="Enter full path to your point cloud file",
        )
        if input_path and Path(input_path).exists():
            file_size = Path(input_path).stat().st_size / (1024**3)
            st.info(f"📊 File size: {file_size:.2f} GB")
        else:
            st.warning("⚠️ No files found in data/input. Enter path manually.")

with col2:
    output_name = st.text_input(
        "Output Folder Name", value="output", help="Name for output folder"
    )

# ─── Run Button ───
st.markdown("")

if st.button(
    "🚀 Run Pipeline", type="primary", disabled=not input_path or not selected_stages
):
    if not input_path:
        st.error("Please select or enter an input file")
    elif not selected_stages:
        st.error("Please select at least one stage")
    else:
        output_dir = f"data/output/{output_name}"
        stages_str = ",".join(map(str, sorted(selected_stages)))

        with st.spinner("Processing... This may take several minutes for large files."):
            result = run_pipeline(
                input_path, output_dir, stages_str, stream_threshold, resolution, use_ml
            )

            if result.returncode == 0:
                st.success("✅ Pipeline completed successfully!")

                # Show results
                output_path = Path(output_dir)
                metrics = load_metrics(str(output_path))

                if metrics:
                    display_results(output_path, metrics)
            else:
                st.error("❌ Pipeline failed")
                st.code(
                    result.stderr[-2000:]
                    if len(result.stderr) > 2000
                    else result.stderr
                )

# ─── Demo Results Section ───
st.markdown("---")
st.markdown("## 📊 Visualization & Results")

# Find available output folders
output_base = Path("data/output")
output_folders = []

if output_base.exists():
    output_folders = [d.name for d in output_base.iterdir() if d.is_dir()]

# Output folder selector
st.markdown("### 📂 Select Output Folder")
col_sel1, col_sel2 = st.columns([2, 1])

with col_sel1:
    if output_folders:
        selected_output = st.selectbox(
            "Choose output folder to visualize",
            options=output_folders,
            index=output_folders.index("DEVDI") if "DEVDI" in output_folders else 0,
        )
    else:
        selected_output = "output"
        st.info("No output folders found. Run the pipeline first.")

with col_sel2:
    st.caption(f"Found: {len(output_folders)} folder(s)")

demo_output = output_base / selected_output

if demo_output.exists():
    metrics = load_metrics(str(demo_output))

    # Create tabs for visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🗺️ DTM",
            "🌊 Waterlogging",
            "🛣️ Drainage Network",
            "📈 Terrain Analysis",
            "📋 Metrics",
        ]
    )

    # ─── Tab 1: DTM ───
    with tab1:
        st.markdown("### Digital Terrain Model")

        dtm_path = demo_output / "dtm.tif"
        if dtm_path.exists():
            try:
                data, bounds, crs, transform = load_raster(str(dtm_path))
                rgb, vmin, vmax = colorize_elevation(data)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(rgb, caption="DTM Elevation Map", use_container_width=True)
                with col2:
                    st.markdown(create_dtm_legend(vmin, vmax), unsafe_allow_html=True)
                    st.markdown(
                        f"""
                    <div class="card">
                    <b>Statistics</b><br>
                    Min: {vmin:.2f} m<br>
                    Max: {vmax:.2f} m<br>
                    Relief: {vmax - vmin:.2f} m
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error loading DTM: {e}")
        else:
            st.warning("DTM not found. Run pipeline with stages 1-3.")

    # ─── Tab 2: Waterlogging ───
    with tab2:
        st.markdown("### Waterlogging Risk Prediction")

        wl_path = demo_output / "waterlogging_probability.tif"
        if wl_path.exists():
            try:
                data, bounds, crs, transform = load_raster(str(wl_path))
                rgb = colorize_risk(data)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(
                        rgb,
                        caption="Waterlogging Probability Map",
                        use_container_width=True,
                    )
                with col2:
                    st.markdown(create_risk_legend(), unsafe_allow_html=True)

                    # Statistics
                    valid_data = data[data > -9999]
                    high_risk = (valid_data >= 0.45).sum() / len(valid_data) * 100
                    st.markdown(
                        f"""
                    <div class="card">
                    <b>Risk Statistics</b><br>
                    Mean: {np.mean(valid_data):.2f}<br>
                    High Risk (≥45%): {high_risk:.1f}%<br>
                    Max: {np.max(valid_data):.2f}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error loading waterlogging map: {e}")
        else:
            st.warning("Waterlogging map not found. Run pipeline with stage 5.")

    # ─── Tab 3: Drainage Network ───
    with tab3:
        st.markdown("### Drainage Network")

        drainage_path = demo_output / "drainage_network.gpkg"
        if drainage_path.exists():
            try:
                import folium
                from streamlit_folium import st_folium

                # Load drainage_channels layer (not default polygon layer)
                gdf = load_geopackage(str(drainage_path), layer="drainage_channels")

                # Get centroid for map center
                centroid = gdf.geometry.centroid.iloc[0]

                # Create map
                m = folium.Map(
                    location=[centroid.y, centroid.x],
                    zoom_start=14,
                    tiles="cartodbpositron",
                )

                # Add drainage network
                folium.GeoJson(
                    gdf,
                    style_function=lambda x: {
                        "color": "#2c5282",
                        "weight": 3,
                        "opacity": 0.8,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=[
                            "segment_id",
                            "length_m",
                            "slope_mm",
                            "depth_m",
                            "bottom_width_m",
                            "capacity_m3s",
                            "cost_inr",
                        ],
                        aliases=[
                            "Segment:",
                            "Length (m):",
                            "Slope (‰):",
                            "Depth (m):",
                            "Width (m):",
                            "Capacity (m³/s):",
                            "Cost (₹):",
                        ],
                        localize=True,
                    ),
                ).add_to(m)

                # Add legend
                legend_html = """
                <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; border: 1px solid #ccc;">
                <b>Legend</b><br>
                <span style="color: #2c5282; font-weight: bold;">━━</span> Drainage Channel
                </div>
                """
                m.get_root().html.add_child(folium.Element(legend_html))

                st_folium(m, height=500, width="100%")

                # Network statistics
                if metrics and "drainage" in metrics:
                    dr = metrics["drainage"]
                    st.markdown(
                        """
                    <div style="display: flex; gap: 20px; margin-top: 20px;">
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Channels</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{:.1f} km</div>
                        <div class="metric-label">Total Length</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">₹{:.0f} L</div>
                        <div class="metric-label">Estimated Cost</div>
                    </div>
                    </div>
                    """.format(
                            dr.get("channel_count", 0),
                            dr.get("total_length_m", 0) / 1000,
                            dr.get("total_cost_inr", 0) / 100000,
                        ),
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Error loading drainage network: {e}")
                st.info("Note: Large drainage networks may take time to render.")
        else:
            st.warning("Drainage network not found. Run pipeline with stage 6.")

    # ─── Tab 4: Terrain Analysis ───
    with tab4:
        st.markdown("### Terrain Derivatives")

        terrain_files = {
            "slope.tif": "Slope",
            "aspect.tif": "Aspect",
            "hillshade.tif": "Hillshade",
            "twi.tif": "Topographic Wetness Index",
            "roughness.tif": "Roughness",
        }

        cols = st.columns(3)
        for i, (fname, label) in enumerate(terrain_files.items()):
            fpath = demo_output / fname
            if fpath.exists():
                try:
                    data, bounds, crs, transform = load_raster(str(fpath))
                    data = data.astype(np.float32)
                    data[data == -9999] = np.nan

                    # Normalize for display
                    vmin, vmax = np.nanpercentile(data, (2, 98))
                    if vmax > vmin:
                        normalized = ((data - vmin) / (vmax - vmin) * 255).astype(
                            np.uint8
                        )
                    else:
                        normalized = np.zeros_like(data, dtype=np.uint8)

                    with cols[i % 3]:
                        st.image(normalized, caption=label, clamp=True)
                except Exception as e:
                    with cols[i % 3]:
                        st.warning(f"Error loading {fname}")
            else:
                with cols[i % 3]:
                    st.caption(f"{label}: Not available")

    # ─── Tab 5: Metrics ───
    with tab5:
        st.markdown("### Performance Metrics")

        if metrics:
            cols = st.columns(4)

            if "ground_classification" in metrics:
                gc = metrics["ground_classification"]
                with cols[0]:
                    st.metric("Ground F1", f"{gc.get('f1_score', 0):.3f}")
                with cols[1]:
                    st.metric("Ground Recall", f"{gc.get('recall', 0) * 100:.1f}%")
                with cols[2]:
                    st.metric(
                        "Ground Precision", f"{gc.get('precision', 0) * 100:.1f}%"
                    )
                with cols[3]:
                    st.metric("IoU", f"{gc.get('iou', 0):.3f}")

            cols2 = st.columns(4)
            if "dtm" in metrics:
                dtm = metrics["dtm"]
                with cols2[0]:
                    st.metric("DTM RMSE", f"{dtm.get('rmse_m', 0):.3f} m")
                with cols2[1]:
                    st.metric("DTM MAE", f"{dtm.get('mae_m', 0):.3f} m")
                with cols2[2]:
                    st.metric("LE90", f"{dtm.get('le90_m', 0):.3f} m")
                with cols2[3]:
                    st.metric("NMAD", f"{dtm.get('nmad_m', 0):.3f} m")

            if "waterlogging" in metrics:
                wl = metrics["waterlogging"]
                st.markdown("#### Waterlogging Model")
                cols3 = st.columns(4)
                with cols3[0]:
                    st.metric(
                        "ROC AUC", f"{wl.get('mean_metrics', {}).get('roc_auc', 0):.3f}"
                    )
                with cols3[1]:
                    st.metric(
                        "F1 Score", f"{wl.get('mean_metrics', {}).get('f1', 0):.3f}"
                    )
                with cols3[2]:
                    st.metric(
                        "Precision",
                        f"{wl.get('mean_metrics', {}).get('precision', 0):.3f}",
                    )
                with cols3[3]:
                    st.metric(
                        "Recall", f"{wl.get('mean_metrics', {}).get('recall', 0):.3f}"
                    )

                st.markdown("##### Feature Importances")
                fi_data = wl.get("feature_importances", [])
                if fi_data:
                    fi_df = [
                        {"Feature": f["feature"], "Importance": f["importance"]}
                        for f in fi_data
                    ]
                    st.bar_chart(
                        [f["importance"] for f in fi_data[:5]], x_label="Feature"
                    )
                    st.caption(
                        ", ".join(
                            [
                                f"{f['feature']}: {f['importance']:.3f}"
                                for f in fi_data[:5]
                            ]
                        )
                    )

            if "drainage" in metrics:
                dr = metrics["drainage"]
                st.markdown("#### Drainage Network")
                cols4 = st.columns(4)
                with cols4[0]:
                    st.metric("Channels", dr.get("channel_count", 0))
                with cols4[1]:
                    st.metric(
                        "Total Length", f"{dr.get('total_length_m', 0) / 1000:.1f} km"
                    )
                with cols4[2]:
                    st.metric(
                        "Est. Cost", f"₹{dr.get('total_cost_inr_lakhs', 0):.1f} L"
                    )
                with cols4[3]:
                    st.metric("Avg Velocity", f"{dr.get('avg_velocity_ms', 0):.2f} m/s")
        else:
            st.info("No metrics available. Run pipeline with --evaluate flag.")

        # Download section
        st.markdown("---")
        st.markdown("### 📥 Download Outputs")

        output_files = get_output_files(demo_output)

        cols = st.columns(3)
        for i, f in enumerate(output_files):
            with cols[i % 3]:
                file_size_mb = f.stat().st_size / (1024**2)
                st.download_button(
                    label=f"📥 {f.name} ({file_size_mb:.1f} MB)",
                    data=f.read_bytes(),
                    file_name=f.name,
                )

else:
    st.info("No demo data found. Run the pipeline to generate results.")

# ─── Footer ───
st.markdown("---")
st.caption("""
**DTM Drainage AI** - MoPR Geospatial Intelligence Hackathon  
Output Format: OGC-compliant (GeoPackage, Cloud-Optimized GeoTIFF, LAS 1.4)
""")


# ─────────────────────────────────────────────────────────────────────────────
# Function to display results after pipeline run
# ─────────────────────────────────────────────────────────────────────────────


def display_results(output_path, metrics):
    """Display results after pipeline completion."""
    st.markdown("---")
    st.markdown("## 📊 Results")

    if metrics:
        cols = st.columns(4)

        if "ground_classification" in metrics:
            gc = metrics["ground_classification"]
            with cols[0]:
                st.metric("Ground F1", f"{gc.get('f1_score', 0):.2f}")

        if "dtm" in metrics:
            dtm = metrics["dtm"]
            with cols[1]:
                st.metric("DTM RMSE", f"{dtm.get('rmse_m', 0):.2f}m")

        if "waterlogging" in metrics:
            wl = metrics["waterlogging"]
            with cols[2]:
                st.metric(
                    "Waterlog AUC",
                    f"{wl.get('mean_metrics', {}).get('roc_auc', 0):.2f}",
                )

        if "drainage" in metrics:
            dr = metrics["drainage"]
            with cols[3]:
                st.metric("Drainage Cost", f"₹{dr.get('total_cost_inr', 0):,.0f}")

    st.info(
        "💡 Switch to the Visualization tabs above to see maps and interactive views."
    )
