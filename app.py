"""
Streamlit App for DTM Drainage AI
=================================
A web interface for the MoPR Hackathon DTM + Drainage Network pipeline.

Usage:
    streamlit run app.py

Requirements:
    - All pipeline dependencies
    - streamlit
    - folium (for maps)
    - rasterio, matplotlib (for visualization)
"""

import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import tempfile
import shutil
from pathlib import Path
import json
import subprocess
import sys

st.set_page_config(
    page_title="DTM Drainage AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_SCRIPT = "run_pipeline.py"
CONFIG_FILE = "config/config.yaml"

# Pipeline stage descriptions
STAGE_INFO = {
    1: {
        "name": "Data Inspection",
        "desc": "Analyze point cloud metadata, density, CRS",
    },
    2: {
        "name": "Ground Classification",
        "desc": "SMRF + ML to separate ground from objects",
    },
    3: {"name": "DTM Generation", "desc": "IDW interpolation to create terrain model"},
    4: {"name": "Hydrological Analysis", "desc": "Flow direction, accumulation, TWI"},
    5: {
        "name": "Waterlogging Prediction",
        "desc": "XGBoost model for flood risk zones",
    },
    6: {
        "name": "Drainage Network Design",
        "desc": "MST + Manning's equation for optimal drains",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(input_file: Path, output_dir: Path, stages: str = "1,2,3,4,5,6"):
    """Run the pipeline and return output."""
    cmd = [
        sys.executable,
        PIPELINE_SCRIPT,
        "--input",
        str(input_file),
        "--output",
        str(output_dir),
        "--stages",
        stages,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def load_metrics(output_dir: Path) -> dict:
    """Load metrics.json if available."""
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def display_dtm_legend():
    """Display DTM color legend."""
    legend_html = """
    <div style="background:white; padding:10px; border-radius:5px; border:1px solid #ccc;">
    <b>Elevation (m)</b><br>
    <div style="background:linear-gradient(to right, #2b2b2b, #4f8cff, #00ff00, #ffff00, #ff0000); height:20px; width:100%;"></div>
    <div style="display:flex; justify-content:space-between; font-size:12px;">
        <span>Low</span><span>High</span>
    </div>
    </div>
    """
    return legend_html


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌊 DTM Drainage AI - MoPR Hackathon")
st.markdown("""
**Digital Terrain Model generation and Drainage Network design from Drone Point Cloud Data**

Upload a LAS/LAZ file to generate:
- DTM (Digital Terrain Model)
- Hydrological analysis layers
- Waterlogging risk prediction
- Optimized drainage network
""")

# ─── Sidebar: Configuration ───
with st.sidebar:
    st.header("⚙️ Pipeline Settings")

    stages_to_run = st.multiselect(
        "Stages to run",
        options=list(STAGE_INFO.keys()),
        default=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: f"{x}. {STAGE_INFO[x]['name']}",
    )

    stream_threshold = st.slider(
        "Stream Threshold",
        100,
        5000,
        1000,
        help="Min flow accumulation to define streams",
    )

    resolution = st.slider("DTM Resolution (m)", 0.1, 2.0, 0.5, 0.1)

    use_ml = st.checkbox(
        "Use ML Refinement", value=True, help="Use ML for better ground classification"
    )

# ─── Main Area ───
uploaded_file = st.file_uploader(
    "📁 Upload Point Cloud (LAS/LAZ)", type=[".las", ".laz"]
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.getvalue())

        # Output directory
        output_name = uploaded_file.name.replace(".laz", "").replace(".las", "")
        output_dir = Path(tmpdir) / output_name

        # Build stages string
        stages_str = ",".join(map(str, sorted(stages_to_run)))

        # Run button
        if st.button("🚀 Run Pipeline", type="primary"):
            with st.spinner("Running DTM Drainage AI pipeline..."):
                result = run_pipeline(tmp_path, output_dir, stages_str)

                if result.returncode == 0:
                    st.success("✅ Pipeline completed successfully!")

                    # Load metrics
                    metrics = load_metrics(output_dir)

                    # ─── Results Section ───
                    st.divider()
                    st.header("📊 Results")

                    # Metrics overview
                    if metrics:
                        col1, col2, col3, col4 = st.columns(4)

                        if "ground_classification" in metrics:
                            gc = metrics["ground_classification"]
                            col1.metric("Ground F1", f"{gc.get('f1_score', 0):.2f}")
                            col2.metric("Ground Recall", f"{gc.get('recall', 0):.1%}")

                        if "dtm" in metrics:
                            dtm = metrics["dtm"]
                            col3.metric("DTM RMSE", f"{dtm.get('rmse_m', 0):.2f}m")

                        if "waterlogging" in metrics:
                            wl = metrics["waterlogging"]
                            col4.metric(
                                "Waterlog AUC",
                                f"{wl.get('mean_metrics', {}).get('roc_auc', 0):.2f}",
                            )

                        if "drainage" in metrics:
                            dr = metrics["drainage"]
                            st.metric(
                                "Drainage Cost", f"₹{dr.get('total_cost_inr', 0):,.0f}"
                            )

                    # ─── Visualizations ───
                    st.divider()

                    tab1, tab2, tab3, tab4 = st.tabs(
                        [
                            "🗺️ DTM Visualization",
                            "🌊 Waterlogging Risk",
                            "🛤️ Drainage Network",
                            "📥 Download Outputs",
                        ]
                    )

                    # Tab 1: DTM
                    with tab1:
                        st.subheader("Digital Terrain Model")
                        dtm_path = output_dir / "dtm.tif"
                        if dtm_path.exists():
                            st.info(f"DTM generated: {dtm_path.name}")
                            st.info(f"Resolution: {resolution}m")

                            # Display legend
                            st.markdown(display_dtm_legend(), unsafe_allow_html=True)

                            st.caption(
                                "DTM preview would appear here. Use QGIS for full visualization."
                            )
                        else:
                            st.warning("DTM not generated (may need stages 1-3)")

                    # Tab 2: Waterlogging
                    with tab2:
                        st.subheader("Waterlogging Probability")
                        wl_path = output_dir / "waterlogging_probability.tif"
                        if wl_path.exists():
                            st.info(f"Probability map: {wl_path.name}")

                            if "waterlogging" in metrics:
                                wl = metrics["waterlogging"]
                                threshold = wl.get("threshold", "N/A")
                                positive_rate = wl.get("positive_rate", 0) * 100
                                st.metric("Risk Threshold", f"{threshold}")
                                st.metric("High Risk Area", f"{positive_rate:.1f}%")
                        else:
                            st.warning(
                                "Waterlogging prediction not generated (need stage 5)"
                            )

                    # Tab 3: Drainage Network
                    with tab3:
                        st.subheader("Drainage Network Design")
                        drainage_path = output_dir / "drainage_network.gpkg"
                        if drainage_path.exists():
                            st.info(f"Drainage network: {drainage_path.name}")

                            if "drainage" in metrics:
                                dr = metrics["drainage"]
                                col1, col2 = st.columns(2)
                                col1.metric("Channels", dr.get("channel_count", 0))
                                col2.metric(
                                    "Total Length",
                                    f"{dr.get('total_length_m', 0) / 1000:.1f} km",
                                )

                            st.info(
                                "Use QGIS to view the GeoPackage with drainage segments and properties."
                            )
                        else:
                            st.warning("Drainage network not generated (need stage 6)")

                    # Tab 4: Downloads
                    with tab4:
                        st.subheader("Download Outputs")

                        output_files = (
                            list(output_dir.glob("*.tif"))
                            + list(output_dir.glob("*.gpkg"))
                            + list(output_dir.glob("*.las"))
                        )

                        if output_files:
                            for f in output_files:
                                st.download_button(
                                    label=f"📥 {f.name}",
                                    data=f.read_bytes(),
                                    file_name=f.name,
                                )
                        else:
                            st.warning("No output files generated")

                else:
                    st.error("❌ Pipeline failed")
                    st.code(result.stderr)

        # Show pipeline info
        st.divider()
        st.subheader("ℹ️ Pipeline Stages")
        for stage_num in sorted(stages_to_run):
            info = STAGE_INFO[stage_num]
            st.markdown(f"**{stage_num}. {info['name']}**")
            st.caption(info["desc"])

# ─── Footer ───
st.divider()
st.caption("""
DTM Drainage AI - MoPR Geospatial Intelligence Hackathon  
For full visualization, use QGIS with the generated GeoPackage and GeoTIFF files.
""")
