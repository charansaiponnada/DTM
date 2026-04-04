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

st.set_page_config(
    page_title="DTM Drainage AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for better UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c5282;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
    }
    
    /* Pipeline steps */
    .pipeline-step {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: white;
        border-radius: 8px;
        border-left: 4px solid #4299e1;
    }
    .step-number {
        background: #4299e1;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-right: 1rem;
    }
    
    /* Status indicators */
    .status-success {
        color: #38a169;
        font-weight: 600;
    }
    .status-error {
        color: #e53e3e;
        font-weight: 600;
    }
    .status-pending {
        color: #718096;
    }
    
    /* Output file list */
    .output-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        background: white;
        border-radius: 6px;
        margin: 0.25rem 0;
        border: 1px solid #e2e8f0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_SCRIPT = "run_pipeline.py"
CONFIG_FILE = "config/config.yaml"

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

OUTPUT_FILES = {
    "dtm.tif": "Digital Terrain Model",
    "waterlogging_probability.tif": "Waterlogging Risk Map",
    "drainage_network.gpkg": "Drainage Network (GeoPackage)",
    "flow_direction.tif": "Flow Direction",
    "flow_accumulation.tif": "Flow Accumulation",
    "twi.tif": "Topographic Wetness Index",
    "slope.tif": "Slope",
    "aspect.tif": "Aspect",
    "hillshade.tif": "Hillshade",
    "classified_ground.las": "Classified Point Cloud",
}

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
    """Run the pipeline and return output."""
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
    """Load metrics.json if available."""
    metrics_path = Path(output_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def get_available_files():
    """Get list of available LAS/LAZ files in data/input folder."""
    input_dir = Path("data/input")
    if input_dir.exists():
        return list(input_dir.glob("*.las")) + list(input_dir.glob("*.laz"))
    return []


def get_output_files(output_dir: Path):
    """Get list of generated output files."""
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

    # Stage selection
    selected_stages = st.multiselect(
        "Select Stages",
        options=list(STAGES.keys()),
        default=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: f"{STAGES[x]['icon']} {STAGES[x]['name']}",
    )

    # Parameters
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

    # Quick info
    st.markdown("---")
    st.markdown("""
    **Tips:**
    - Processing 1.7GB files may take 10-30 minutes
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
    # Check for available files
    available_files = get_available_files()

    if available_files:
        file_options = {f.name: str(f) for f in available_files}
        selected_file = st.selectbox(
            "Select Point Cloud File",
            options=list(file_options.keys()),
            help="Choose from available files in data/input folder",
        )
        input_path = file_options[selected_file]
        file_size = Path(input_path).stat().st_size / (1024**3)  # GB
        st.info(f"📊 File size: {file_size:.2f} GB")
    else:
        # Show path input for large files
        input_path = st.text_input(
            "Enter LAS/LAZ File Path",
            placeholder="C:\\path\\to\\your\\file.las",
            help="Enter full path to your large point cloud file",
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
        # Create output directory path
        output_dir = f"data/output/{output_name}"
        stages_str = ",".join(map(str, sorted(selected_stages)))

        # Progress container
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Processing...")

            # Show selected stages
            for stage in sorted(selected_stages):
                st.markdown(
                    f"""
                <div class="pipeline-step">
                    <span class="step-number">{stage}</span>
                    <span>{STAGES[stage]["icon"]} {STAGES[stage]["name"]}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown(f"**Input:** `{Path(input_path).name}`")
            st.markdown(f"**Output:** `{output_dir}`")

            # Progress bar
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
                status_text.text("✅ Pipeline completed!")
                st.success("Processing completed successfully!")
            else:
                status_text.text("❌ Pipeline failed")
                st.error("Processing failed. Check the error below:")
                st.code(
                    result.stderr[-2000:]
                    if len(result.stderr) > 2000
                    else result.stderr
                )

# ─── Demo Data Section ───
st.markdown("---")
st.markdown("## 📊 Demo Results")

# Check for existing output
demo_output = Path("data/output/DEVDI")
if demo_output.exists():
    metrics = load_metrics(str(demo_output))

    # Metrics display
    if metrics:
        st.markdown("### Performance Metrics")

        cols = st.columns(4)

        if "ground_classification" in metrics:
            gc = metrics["ground_classification"]
            with cols[0]:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{gc.get("f1_score", 0):.2f}</div>
                    <div class="metric-label">Ground F1 Score</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{gc.get("recall", 0) * 100:.0f}%</div>
                    <div class="metric-label">Ground Recall</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        if "dtm" in metrics:
            dtm = metrics["dtm"]
            with cols[2]:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{dtm.get("rmse_m", 0):.2f}m</div>
                    <div class="metric-label">DTM RMSE</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        if "waterlogging" in metrics:
            wl = metrics["waterlogging"]
            with cols[3]:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{wl.get("mean_metrics", {}).get("roc_auc", 0):.2f}</div>
                    <div class="metric-label">Waterlog AUC</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        if "drainage" in metrics:
            dr = metrics["drainage"]
            st.markdown("### Drainage Network")
            dcols = st.columns(3)
            with dcols[0]:
                st.metric("Channels", dr.get("channel_count", 0))
            with dcols[1]:
                st.metric(
                    "Total Length", f"{dr.get('total_length_m', 0) / 1000:.1f} km"
                )
            with dcols[2]:
                st.metric(
                    "Est. Cost", f"₹{dr.get('total_cost_inr', 0) / 10000000:.2f} Cr"
                )

    # Output files
    st.markdown("### Generated Files")
    output_files = get_output_files(demo_output)

    for f in output_files:
        if f.name in OUTPUT_FILES:
            desc = OUTPUT_FILES[f.name]
        else:
            desc = f.name

        file_size_mb = f.stat().st_size / (1024**2)
        st.markdown(
            f"""
        <div class="output-item">
            <span>📄 <b>{f.name}</b> - {desc}</span>
            <span style="color:#718096">{file_size_mb:.1f} MB</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.info(
        "💡 For full visualization, open the GeoPackage (.gpkg) and GeoTIFF (.tif) files in QGIS."
    )

else:
    st.info("No demo data found. Run the pipeline to generate results.")

# ─── Pipeline Overview ───
st.markdown("---")
st.markdown("## 🔬 Pipeline Overview")

cols = st.columns(3)
for i, (num, info) in enumerate(STAGES.items()):
    with cols[i % 3]:
        st.markdown(
            f"""
        <div class="card">
            <h4>{info["icon"]} Stage {num}: {info["name"]}</h4>
            <p style="color:#666">{info["desc"]}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ─── Footer ───
st.markdown("---")
st.caption(f"""
**DTM Drainage AI** - MoPR Geospatial Intelligence Hackathon  
Output Format: OGC-compliant (GeoPackage, Cloud-Optimized GeoTIFF, LAS 1.4)
""")
