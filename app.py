"""
DTM Drainage AI - Streamlit Web App
====================================
MoPR Geospatial Intelligence Hackathon
A robust, production-ready web interface for processing drone point cloud data.
"""

import streamlit as st
import subprocess
import sys
import json
import os
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import traceback

st.set_page_config(
    page_title="DTM Drainage AI - MoPR Hackathon",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "DTM Drainage AI - Ministry of Panchayati Raj Geospatial Hackathon",
    },
)

# ══════════════════════════════════════════════════════════════════════════
# Custom CSS - Professional Hackathon Look
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
    /* Main styling */
    .block-container { padding-top: 1rem; }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #a0aec0;
        margin-top: 0.5rem;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4299e1;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #2c5282; }
    .metric-label { font-size: 0.8rem; color: #718096; text-transform: uppercase; }
    
    /* Pipeline step indicators */
    .step-pending { color: #a0aec0; }
    .step-running { color: #ed8936; }
    .step-success { color: #38a169; }
    .step-error { color: #e53e3e; }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4299e1, #38a169);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: #2c5282;
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .info-box-blue { background: #ebf8ff; border-left: 4px solid #4299e1; }
    .info-box-green { background: #f0fff4; border-left: 4px solid #38a169; }
    .info-box-yellow { background: #fffaf0; border-left: 4px solid #ed8936; }
    .info-box-red { background: #fff5f5; border-left: 4px solid #e53e3e; }
    
    /* Download button */
    .download-btn {
        background: #2c5282;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        text-decoration: none;
    }
    
    /* Sidebar sections */
    .sidebar-section {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

PIPELINE_SCRIPT = "run_pipeline.py"

STAGES = {
    1: {
        "name": "Data Inspection",
        "icon": "1",
        "desc": "Analyze point cloud metadata, density, CRS",
    },
    2: {
        "name": "Ground Classification",
        "icon": "2",
        "desc": "SMRF + ML to separate ground from objects",
    },
    3: {
        "name": "DTM Generation",
        "icon": "3",
        "desc": "IDW interpolation to create terrain model",
    },
    4: {
        "name": "Hydrological Analysis",
        "icon": "4",
        "desc": "Flow direction, accumulation, TWI",
    },
    5: {
        "name": "Waterlogging Prediction",
        "icon": "5",
        "desc": "XGBoost model for flood risk zones",
    },
    6: {
        "name": "Drainage Network",
        "icon": "6",
        "desc": "MST + Manning's for optimal drains",
    },
}

# ══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════


def set_page_config():
    """Initialize page configuration."""
    st.set_page_config(
        page_title="DTM Drainage AI - MoPR Hackathon",
        page_icon="🗺️",
        layout="wide",
    )


def run_pipeline(input_file, output_dir, stages, stream_threshold, resolution, use_ml):
    """Run the pipeline with proper encoding for Windows."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable,
        PIPELINE_SCRIPT,
        "--input",
        str(input_file),
        "--output",
        str(output_dir),
        "--stages",
        stages,
        "--stream-threshold",
        str(stream_threshold),
        "--resolution",
        str(resolution),
    ]
    if not use_ml:
        cmd.append("--no-ml")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,  # 1 hour max
        )
        return result
    except subprocess.TimeoutExpired:
        return type(
            "obj",
            (object,),
            {"returncode": -1, "stderr": "Pipeline timed out after 1 hour"},
        )()
    except Exception as e:
        return type("obj", (object,), {"returncode": -1, "stderr": str(e)})()


def load_metrics(output_dir):
    """Load metrics.json if available."""
    metrics_path = Path(output_dir) / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def get_available_files():
    """Get list of available LAS/LAZ files."""
    input_dir = Path("data/input")
    if input_dir.exists():
        files = list(input_dir.glob("*.las")) + list(input_dir.glob("*.laz"))
        return [(f.name, str(f)) for f in files]
    return []


def get_output_folders():
    """Get list of output folders."""
    output_dir = Path("data/output")
    if output_dir.exists():
        return [d.name for d in output_dir.iterdir() if d.is_dir()]
    return []


def load_raster(path):
    """Load raster with caching."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1)
        bounds = src.bounds
        crs = src.crs
    return data, bounds, crs


def colorize_dtm(data, nodata=-9999):
    """Create RGB from elevation data."""
    data = data.astype(np.float32)
    data[data == nodata] = np.nan

    # Handle all NaN case
    if np.all(np.isnan(data)):
        return np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8), 0, 1

    vmin, vmax = np.nanpercentile(data, (2, 98))
    if vmax == vmin:
        vmax = vmin + 1

    normalized = (data - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.5)

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
    data = np.nan_to_num(data, nan=0.5)

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


def create_download_zip(output_dir):
    """Create a ZIP file of all outputs."""
    zip_path = output_dir / "outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.rglob("*.*"):
            if f.is_file() and not f.name.startswith(".") and "_tiles" not in str(f):
                zf.write(f, f.name)
    return zip_path


# ══════════════════════════════════════════════════════════════════════════
# Session State Management
# ══════════════════════════════════════════════════════════════════════════

if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = "idle"  # idle, running, success, error
if "current_output" not in st.session_state:
    st.session_state.current_output = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# ══════════════════════════════════════════════════════════════════════════
# Sidebar - Settings
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Pipeline Configuration")

    # Stage selection
    selected_stages = st.multiselect(
        "Select Stages (1-6)",
        options=list(STAGES.keys()),
        default=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: f"Stage {x}: {STAGES[x]['name']}",
    )

    st.markdown("### Parameters")

    # Technical parameters
    col1, col2 = st.columns(2)
    with col1:
        stream_threshold = st.slider("Stream Threshold", 100, 5000, 1000, 50)
    with col2:
        resolution = st.slider("Resolution (m)", 0.1, 2.0, 0.5, 0.1)

    use_ml = st.checkbox(
        "Use ML Refinement", value=True, help="Better ground classification using ML"
    )

    st.markdown("---")
    st.markdown("### About")
    st.caption("""
    **DTM Drainage AI**
    
    MoPR Geospatial Intelligence Hackathon
    
    Input: LAS/LAZ Point Cloud
    Output: DTM, Drainage Network, Waterlogging Risk Map
    """)

    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("- [QGIS Download](https://qgis.org)")
    st.markdown("- [Input Data Format](docs/)")

# ══════════════════════════════════════════════════════════════════════════
# Main Content
# ══════════════════════════════════════════════════════════════════════════

# Header
st.markdown(
    """
<div class="header-container">
    <h1 class="header-title">DTM Drainage AI</h1>
    <p class="header-subtitle">Ministry of Panchayati Raj - Geospatial Intelligence Hackathon</p>
</div>
""",
    unsafe_allow_html=True,
)

# Quick info
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.markdown(
        '<div class="info-box info-box-blue"><b>Input Format:</b> LAS/LAZ Point Cloud</div>',
        unsafe_allow_html=True,
    )
with col_info2:
    st.markdown(
        '<div class="info-box info-box-green"><b>Output:</b> GeoPackage, GeoTIFF (OGC)</div>',
        unsafe_allow_html=True,
    )
with col_info3:
    st.markdown(
        '<div class="info-box info-box-yellow"><b>Processing:</b> 6-Stage Pipeline</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# Input Section
# ══════════════════════════════════════════════════════════════════════════

st.markdown("## 1. Input Data Selection")

col_input1, col_input2 = st.columns([3, 1])

with col_input1:
    # Get available files
    available_files = get_available_files()

    if available_files:
        file_options = {name: path for name, path in available_files}
        selected_file = st.selectbox(
            "Select Point Cloud File",
            options=list(file_options.keys()),
            help="Choose from available files in data/input folder",
        )
        input_path = file_options[selected_file]

        # Show file info
        if input_path and Path(input_path).exists():
            file_size_gb = Path(input_path).stat().st_size / (1024**3)
            st.info(f"File: {selected_file} | Size: {file_size_gb:.2f} GB")
    else:
        # Manual input for external files
        input_path = st.text_input(
            "Enter LAS/LAZ File Path",
            placeholder="C:\\data\\village_point_cloud.las",
            help="Enter full path to your point cloud file",
        )

        if input_path and Path(input_path).exists():
            file_size_gb = Path(input_path).stat().st_size / (1024**3)
            st.info(f"File: {Path(input_path).name} | Size: {file_size_gb:.2f} GB")
        else:
            st.warning("No files in data/input. Enter path above or copy files there.")

with col_input2:
    output_name = st.text_input(
        "Output Folder", value="village_output", help="Name for output folder"
    )

# ══════════════════════════════════════════════════════════════════════════
# Pipeline Execution
# ══════════════════════════════════════════════════════════════════════════

st.markdown("## 2. Run Pipeline")

col_run1, col_run2 = st.columns([3, 1])

with col_run1:
    run_button = st.button(
        "🚀 Run Processing Pipeline",
        type="primary",
        disabled=not input_path or not selected_stages,
        width="stretch",
    )

with col_run2:
    st.caption(f"Selected stages: {', '.join(map(str, selected_stages))}")

if run_button:
    if not input_path:
        st.error("Please select or enter an input file")
    elif not selected_stages:
        st.error("Please select at least one stage")
    else:
        output_dir = Path("data/output") / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        stages_str = ",".join(map(str, sorted(selected_stages)))

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Starting pipeline...")
        progress_bar.progress(10)

        # Run pipeline
        result = run_pipeline(
            input_path, output_dir, stages_str, stream_threshold, resolution, use_ml
        )

        progress_bar.progress(100)

        if result.returncode == 0:
            status_text.text("Pipeline completed successfully!")
            st.session_state.pipeline_status = "success"
            st.session_state.current_output = str(output_dir)

            st.markdown(
                '<div class="info-box info-box-green"><b>Success!</b> Processing completed. View results below.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.session_state.pipeline_status = "error"
            st.markdown(
                '<div class="info-box info-box-red"><b>Error:</b> Pipeline failed. Check error message below.</div>',
                unsafe_allow_html=True,
            )

            # Show error (last 1000 chars to avoid overwhelming)
            error_msg = result.stderr[-1000:] if result.stderr else "Unknown error"
            with st.expander("Error Details"):
                st.code(error_msg)

# ══════════════════════════════════════════════════════════════════════════
# Results Visualization
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 3. Results & Visualization")

# Output folder selector
output_folders = get_output_folders()

col_res1, col_res2 = st.columns([3, 1])

with col_res1:
    if output_folders:
        selected_output = st.selectbox(
            "Select Output Folder to View",
            options=output_folders,
            index=0 if "DEVDI" in output_folders else 0,
        )
    else:
        selected_output = None
        st.info("No output folders found. Run the pipeline first.")

with col_res2:
    if selected_output:
        output_path = Path("data/output") / selected_output
        file_count = len(list(output_path.glob("*.tif"))) + len(
            list(output_path.glob("*.gpkg"))
        )
        st.metric("Files Generated", file_count)

# ══════════════════════════════════════════════════════════════════════════
# Visualization Tabs
# ══════════════════════════════════════════════════════════════════════════

if selected_output:
    output_path = Path("data/output") / selected_output
    metrics = load_metrics(str(output_path))

    # Create tabs
    tab_dtm, tab_water, tab_drain, tab_terrain, tab_metrics = st.tabs(
        [
            "1. DTM",
            "2. Waterlogging Risk",
            "3. Drainage Network",
            "4. Terrain Analysis",
            "5. Metrics & Download",
        ]
    )

    # ─────────────────────────────────────────────────────────────────────
    # Tab 1: DTM
    # ─────────────────────────────────────────────────────────────────────
    with tab_dtm:
        st.markdown("### Digital Terrain Model")

        dtm_path = output_path / "dtm.tif"
        if dtm_path.exists():
            try:
                data, bounds, crs = load_raster(str(dtm_path))
                rgb, vmin, vmax = colorize_dtm(data)

                # Display
                col_dtm1, col_dtm2 = st.columns([3, 1])

                with col_dtm1:
                    st.image(rgb, caption="DTM Elevation Map", width="stretch")

                with col_dtm2:
                    st.markdown("#### Statistics")
                    st.metric("Min Elevation", f"{vmin:.1f} m")
                    st.metric("Max Elevation", f"{vmax:.1f} m")
                    st.metric("Relief", f"{vmax - vmin:.1f} m")

                    # Legend
                    st.markdown(
                        """
                    <div style="background:white; padding:10px; border-radius:5px; border:1px solid #ccc; margin-top:20px;">
                    <b>Elevation (m)</b>
                    <div style="background:linear-gradient(to right, rgb(20,20,120), rgb(40,100,180), rgb(0,180,100), rgb(100,220,50), rgb(220,220,0), rgb(255,100,0), rgb(180,20,20)); height:15px; width:100%;"></div>
                    <div style="display:flex; justify-content:space-between; font-size:11px;">
                        <span>{}</span><span>{}</span>
                    </div>
                    </div>
                    """.format(f"{vmin:.0f}", f"{vmax:.0f}"),
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error loading DTM: {e}")
        else:
            st.warning("DTM not found. Run pipeline with stages 1-3.")

    # ─────────────────────────────────────────────────────────────────────
    # Tab 2: Waterlogging Risk
    # ─────────────────────────────────────────────────────────────────────
    with tab_water:
        st.markdown("### Waterlogging Risk Prediction")

        wl_path = output_path / "waterlogging_probability.tif"
        if wl_path.exists():
            try:
                data, bounds, crs = load_raster(str(wl_path))
                rgb = colorize_risk(data)

                col_wl1, col_wl2 = st.columns([3, 1])

                with col_wl1:
                    st.image(
                        rgb,
                        caption="Waterlogging Probability Map",
                        width="stretch",
                    )

                with col_wl2:
                    # Statistics
                    valid_data = data[data > -9999]
                    high_risk = (valid_data >= 0.45).sum() / len(valid_data) * 100
                    mean_prob = np.mean(valid_data)

                    st.metric("Mean Probability", f"{mean_prob:.2f}")
                    st.metric("High Risk Area", f"{high_risk:.1f}%")
                    st.metric("Max Probability", f"{np.max(valid_data):.2f}")

                    # Legend
                    st.markdown(
                        """
                    <div style="background:white; padding:10px; border-radius:5px; border:1px solid #ccc; margin-top:20px;">
                    <b>Risk Level</b>
                    <div style="background:linear-gradient(to right, white, rgb(200,230,255), rgb(100,180,255), rgb(30,100,220), rgb(10,50,180)); height:15px; width:100%;"></div>
                    <div style="display:flex; justify-content:space-between; font-size:11px;">
                        <span>Low</span><span>High</span>
                    </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Waterlogging map not found. Run pipeline with stage 5.")

    # ─────────────────────────────────────────────────────────────────────
    # Tab 3: Drainage Network
    # ─────────────────────────────────────────────────────────────────────
    with tab_drain:
        st.markdown("### Drainage Network Plan View")

        drainage_path = output_path / "drainage_network.gpkg"
        dtm_path = output_path / "dtm.tif"

        if drainage_path.exists() and dtm_path.exists():
            try:
                import matplotlib.pyplot as plt
                from matplotlib.colors import LinearSegmentedColormap
                import geopandas as gpd
                import rasterio

                # Load data
                gdf = gpd.read_file(str(drainage_path), layer="drainage_channels")

                with rasterio.open(str(dtm_path)) as src:
                    dtm_data = src.read(1)
                    dtm_bounds = src.bounds

                # Create figure
                fig, ax = plt.subplots(figsize=(12, 10))

                # DTM background
                dtm_data = dtm_data.astype(np.float32)
                dtm_data[dtm_data == -9999] = np.nan
                vmin, vmax = np.nanpercentile(dtm_data, (2, 98))

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
                    dtm_data,
                    cmap=cmap,
                    norm=plt.Normalize(vmin=vmin, vmax=vmax),
                    extent=[
                        dtm_bounds.left,
                        dtm_bounds.right,
                        dtm_bounds.bottom,
                        dtm_bounds.top,
                    ],
                    origin="upper",
                    alpha=0.8,
                )

                # Drainage overlay
                gdf.plot(ax=ax, color="#2c5282", linewidth=1.5, alpha=0.9)

                ax.set_title(
                    "Drainage Network Plan View", fontsize=14, fontweight="bold"
                )
                ax.set_xlabel("Easting (m)")
                ax.set_ylabel("Northing (m)")

                # Stats
                total_length = gdf["length_m"].sum()
                num_channels = len(gdf)
                total_cost = gdf["cost_inr"].sum()

                stats_text = f"Channels: {num_channels}\nLength: {total_length / 1000:.1f} km\nCost: ₹{total_cost / 100000:.1f}L"
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

                st.pyplot(fig)

                # Channel details
                st.markdown("#### Channel Details")
                cols = st.columns(3)
                cols[0].metric("Total Channels", num_channels)
                cols[1].metric("Total Length", f"{total_length / 1000:.1f} km")
                cols[2].metric("Estimated Cost", f"₹{total_cost / 100000:.1f} L")

                st.dataframe(
                    gdf[
                        [
                            "segment_id",
                            "length_m",
                            "slope_mm",
                            "depth_m",
                            "bottom_width_m",
                            "cost_inr",
                        ]
                    ].head(15),
                    width="stretch",
                )

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning(
                "Drainage network or DTM not found. Run pipeline with stages 3 and 6."
            )

    # ─────────────────────────────────────────────────────────────────────
    # Tab 4: Terrain Analysis
    # ─────────────────────────────────────────────────────────────────────
    with tab_terrain:
        st.markdown("### Terrain Derivatives")

        terrain_files = {
            "slope.tif": "Slope (degrees)",
            "aspect.tif": "Aspect (degrees)",
            "hillshade.tif": "Hillshade",
            "twi.tif": "Topographic Wetness Index",
            "roughness.tif": "Surface Roughness",
        }

        cols = st.columns(3)
        for i, (fname, label) in enumerate(terrain_files.items()):
            fpath = output_path / fname
            if fpath.exists():
                try:
                    data, _, _ = load_raster(str(fpath))
                    data = data.astype(np.float32)
                    data[data == -9999] = np.nan

                    vmin, vmax = np.nanpercentile(data, (2, 98))
                    if vmax > vmin:
                        normalized = (data - vmin) / (vmax - vmin) * 255
                        normalized = np.nan_to_num(normalized, nan=128).astype(np.uint8)
                    else:
                        normalized = np.zeros_like(data, dtype=np.uint8)

                    with cols[i % 3]:
                        st.image(normalized, caption=label, clamp=True)
                except Exception:
                    with cols[i % 3]:
                        st.caption(f"{label}: Error")
            else:
                with cols[i % 3]:
                    st.caption(f"{label}: Not available")

    # ─────────────────────────────────────────────────────────────────────
    # Tab 5: Metrics & Download
    # ─────────────────────────────────────────────────────────────────────
    with tab_metrics:
        st.markdown("### Performance Metrics")

        if metrics:
            # Ground Classification
            if "ground_classification" in metrics:
                gc = metrics["ground_classification"]
                st.markdown("#### Ground Classification")
                cols = st.columns(4)
                cols[0].metric("F1 Score", f"{gc.get('f1_score', 0):.3f}")
                cols[1].metric("Recall", f"{gc.get('recall', 0) * 100:.1f}%")
                cols[2].metric("Precision", f"{gc.get('precision', 0) * 100:.1f}%")
                cols[3].metric("IoU", f"{gc.get('iou', 0):.3f}")

            # DTM
            if "dtm" in metrics:
                dtm = metrics["dtm"]
                st.markdown("#### DTM Accuracy")
                cols = st.columns(4)
                cols[0].metric("RMSE", f"{dtm.get('rmse_m', 0):.3f} m")
                cols[1].metric("MAE", f"{dtm.get('mae_m', 0):.3f} m")
                cols[2].metric("LE90", f"{dtm.get('le90_m', 0):.3f} m")
                cols[3].metric("Resolution", f"{dtm.get('dtm_resolution_m', 0.5)} m")

            # Waterlogging
            if "waterlogging" in metrics:
                wl = metrics["waterlogging"]
                st.markdown("#### Waterlogging Prediction")
                cols = st.columns(4)
                cols[0].metric(
                    "ROC AUC", f"{wl.get('mean_metrics', {}).get('roc_auc', 0):.3f}"
                )
                cols[1].metric(
                    "F1 Score", f"{wl.get('mean_metrics', {}).get('f1', 0):.3f}"
                )
                cols[2].metric(
                    "Precision", f"{wl.get('mean_metrics', {}).get('precision', 0):.3f}"
                )
                cols[3].metric(
                    "Recall", f"{wl.get('mean_metrics', {}).get('recall', 0):.3f}"
                )

            # Drainage
            if "drainage" in metrics:
                dr = metrics["drainage"]
                st.markdown("#### Drainage Network")
                cols = st.columns(4)
                cols[0].metric("Channels", dr.get("channel_count", 0))
                cols[1].metric("Length", f"{dr.get('total_length_m', 0) / 1000:.1f} km")
                cols[2].metric("Cost", f"₹{dr.get('total_cost_inr_lakhs', 0):.1f} L")
                cols[3].metric("Velocity", f"{dr.get('avg_velocity_ms', 0):.2f} m/s")
        else:
            st.info("No metrics available. Run pipeline with --evaluate.")

        # Download section
        st.markdown("---")
        st.markdown("### Download Outputs")

        # Get all output files
        output_files = []
        if output_path.exists():
            for ext in ["*.tif", "*.gpkg", "*.las"]:
                output_files.extend(output_path.glob(ext))

        if output_files:
            # Create ZIP
            with st.spinner("Creating ZIP archive..."):
                zip_path = create_download_zip(output_path)

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                zip_size_mb = zip_path.stat().st_size / (1024**2)
                st.download_button(
                    label=f"📥 Download All ({zip_size_mb:.1f} MB)",
                    data=zip_path.read_bytes(),
                    file_name=f"{selected_output}_outputs.zip",
                    width="stretch",
                )

            with col_dl2:
                st.caption(f"Total files: {len(output_files)}")

            # Individual file downloads
            st.markdown("#### Individual Files")
            cols = st.columns(3)
            for i, f in enumerate(output_files[:12]):  # Show first 12
                with cols[i % 3]:
                    size_mb = f.stat().st_size / (1024**2)
                    st.download_button(
                        label=f"{f.name} ({size_mb:.1f} MB)",
                        data=f.read_bytes(),
                        file_name=f.name,
                    )
        else:
            st.warning("No output files found.")

# ══════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption("""
**DTM Drainage AI** - MoPR Geospatial Intelligence Hackathon  

Output Format: OGC-compliant (GeoPackage, Cloud-Optimized GeoTIFF, LAS 1.4)  
For full GIS visualization, use QGIS with the generated GeoPackage and GeoTIFF files.
""")
