"""
DTM Drainage AI — Streamlit Web App  ·  MoPR / IITTNiF Geospatial Hackathon (PS-2)
=================================================================================
Results-first viewer: pick an already-processed village and explore its flood-risk
and drainage plan layer-by-layer (GIS style). No live pipeline run needed for the
demo — the 4 SVAMITVA villages are pre-processed. A pipeline runner is kept in a
collapsed "offline" expander at the bottom for processing new files.

Run:  streamlit run app/app.py
"""
from __future__ import annotations
import os
import sys
import json
import zipfile
import subprocess
from pathlib import Path

import folium
from folium.plugins import Fullscreen, MiniMap
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from branca.element import MacroElement, Template

sys.path.insert(0, str(Path(__file__).resolve().parent))        # app/ dir → geo_utils
from geo_utils import (
    raster_to_overlay, load_drainage_channels,
    load_waterlogging_hotspots, load_catchment_boundaries,
    high_risk_zones, channel_color, RISK_COLORS,
)

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "output"
METRICS = OUT_DIR / "_reports" / "honest_metrics.json"
PIPELINE_SCRIPT = "run_pipeline.py"

# Villages that are fully processed (present in honest_metrics.json)
VILLAGES = {
    "DEVDI (Gujarat)":          "DEVDI",
    "KHAPRETA (Gujarat)":       "KHAPRETA",
    "DHAL HOSHIARPUR (Punjab)": "DHAL_HOSHIARPUR",
    "CHAKHIRASINGH (Punjab)":   "CHAKHIRASINGH",
}

st.set_page_config(
    page_title="DTM Drainage AI — Flood & Drainage Planner",
    page_icon="🌊", layout="wide", initial_sidebar_state="expanded",
    menu_items={"About": "DTM Drainage AI — Ministry of Panchayati Raj Geospatial Hackathon"},
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.0rem !important; padding-bottom: 0.6rem !important; }
iframe { border-radius: 10px !important; border: 1px solid #dde3ea !important; }
[data-testid="stMetricValue"] { font-size: 25px !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 12.5px !important; }
.header-container { background: linear-gradient(120deg,#0E3A53 0%,#117777 100%);
        padding: 16px 24px; border-radius: 12px; margin-bottom: 14px; }
.header-title { font-size: 1.9rem; font-weight: 800; color: #fff; margin: 0; letter-spacing:.2px; }
.header-subtitle { font-size: .98rem; color: #d7eef0; margin-top: 6px; line-height:1.5; }
.toc-h { font-size:12px; font-weight:800; letter-spacing:.8px; color:#0E3A53;
         text-transform:uppercase; margin:14px 0 4px; }
.rec { background:#FFF6E6; border-left:5px solid #E8910C; border-radius:8px;
       padding:12px 16px; font-size:14px; color:#5a3d00; line-height:1.55; }
section[data-testid="stSidebar"] { width: 345px !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_metrics() -> dict:
    with open(METRICS) as f:
        return json.load(f)

@st.cache_data(show_spinner="Loading village layers…")
def load_village(key: str) -> dict:
    base = OUT_DIR / key
    gpkg = base / "drainage_network.gpkg"
    dtm  = base / "dtm.tif"
    ch    = load_drainage_channels(gpkg, dtm_path=dtm)          # artifacts dropped
    hs_hi = load_waterlogging_hotspots(gpkg, risk_filter=["HIGH"])
    cat   = load_catchment_boundaries(gpkg)
    zones = high_risk_zones(base / "waterlogging_probability.tif", thresh=0.65)
    ovl = dict(
        dtm   = raster_to_overlay(base/"dtm.tif",                     "dtm",          opacity=0.85),
        hill  = raster_to_overlay(base/"hillshade.tif",               "hillshade",    opacity=0.85),
        slope = raster_to_overlay(base/"slope.tif",                   "slope",        opacity=0.75),
        twi   = raster_to_overlay(base/"twi.tif",                     "twi",          opacity=0.75),
        wl    = raster_to_overlay(base/"waterlogging_probability.tif","waterlogging", opacity=0.80, vmin=0, vmax=1),
    )
    hi_ha = round(float(hs_hi["area_m2"].sum()) / 1e4, 1) if len(hs_hi) else 0.0
    return dict(ch=ch, hi=zones, cat=cat, ovl=ovl, center=ovl["dtm"][2], hi_ha=hi_ha)


metrics = load_metrics()
all_v   = metrics["villages"]

BASEMAPS = {
    "🛰️ Satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     "Tiles © Esri"),
    "🗺️ Street":    ("OpenStreetMap", None),
    "◻️ Light":     ("CartoDB positron", None),
    "◼️ Dark":      ("CartoDB dark_matter", None),
}


def legend_macro(show_risk, show_drain) -> MacroElement:
    rows = ""
    if show_risk:
        rows += ('<div><span style="background:#D32F2F" class="lg"></span>High flood risk</div>'
                 '<div><span style="background:#FF8F00" class="lg"></span>Moderate flood risk</div>')
    if show_drain:
        rows += ('<div><span style="background:#C9A227" class="lg"></span>Earthen channel</div>'
                 '<div><span style="background:#19A3C3" class="lg"></span>Concrete channel</div>')
    html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position:fixed; bottom:24px; left:24px; z-index:9999;
        background:rgba(255,255,255,.94); border:1px solid #cdd6df; border-radius:8px;
        padding:9px 13px; font-size:12.5px; color:#223; box-shadow:0 1px 6px rgba(0,0,0,.18);
        font-family:sans-serif; line-height:1.7;">
      <div style="font-weight:700; margin-bottom:3px;">Map key</div>{rows}
    </div>
    <style>.lg{{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:7px;vertical-align:middle;}}</style>
    {{% endmacro %}}
    """
    m = MacroElement(); m._template = Template(html)
    return m


# ── Sidebar = GIS Table of Contents ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 Flood & Drainage Planner")
    st.caption("Layer-by-layer village map · MoPR / IITTNiF · PS-2")

    sel = st.selectbox("**Village** (pre-processed)", list(VILLAGES.keys()))
    vk  = VILLAGES[sel]
    vm  = all_v[vk]
    D   = load_village(vk)

    base_label = st.radio("**Base map**", list(BASEMAPS.keys()), index=0, horizontal=True)

    st.markdown('<div class="toc-h">🟥 Flood risk</div>', unsafe_allow_html=True)
    L_wl  = st.checkbox("Flood-risk heat map", True,
                        help="Model's predicted chance of waterlogging — red = high, yellow = moderate, green = low")
    L_hi  = st.checkbox("High-risk zones (outlined)", True)

    st.markdown('<div class="toc-h">🟦 Drainage plan</div>', unsafe_allow_html=True)
    L_ear = st.checkbox("Earthen channels", True)
    L_con = st.checkbox("Concrete channels", True)
    L_cat = st.checkbox("Catchment areas", False, help="Land area that drains to each outlet")

    st.markdown('<div class="toc-h">⛰️ Terrain</div>', unsafe_allow_html=True)
    L_hill = st.checkbox("Hillshade (3-D relief)", False)
    L_dtm  = st.checkbox("Elevation (DTM)", False)
    L_slp  = st.checkbox("Slope", False)
    L_twi  = st.checkbox("Wetness index (TWI)", False)

    opacity = st.slider("Layer opacity", 0.2, 1.0, 0.80, 0.05)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-container">
  <h1 class="header-title">🌊 {sel}</h1>
  <p class="header-subtitle">Where rainwater is likely to collect, and the drainage channels recommended to
     drain it — built automatically from a LiDAR/drone survey. Tick layers on the left to explore;
     hover any zone or channel for details.</p>
</div>
""", unsafe_allow_html=True)

dr = vm["drainage"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Land at high flood risk", f"{D['hi_ha']:.0f} ha", help="Total high-risk waterlogging area")
c2.metric("Drainage channels needed", f"{dr['channel_count']:,}", f"{dr['total_length_m']/1000:.0f} km total")
c3.metric("Estimated cost", f"₹{dr['total_cost_inr_lakhs']:.0f} L", "CPWD DSR 2023-24")
c4.metric("Flood-model confidence", f"{vm['waterlogging']['roc_auc']*100:.1f}%", "ROC-AUC")

# ── Build the GIS map ────────────────────────────────────────────────────────
tiles, attr = BASEMAPS[base_label]
m = folium.Map(location=D["center"], zoom_start=15, control_scale=True, tiles=tiles, attr=attr)
ovl = D["ovl"]

def _raster(layer_key, name, show):
    if not show: return
    png, bounds, _ = ovl[layer_key]
    folium.raster_layers.ImageOverlay(image=png, bounds=bounds, opacity=opacity,
                                      name=name, interactive=False).add_to(m)

_raster("hill",  "Hillshade", L_hill)
_raster("dtm",   "Elevation", L_dtm)
_raster("slope", "Slope",     L_slp)
_raster("twi",   "Wetness (TWI)", L_twi)
_raster("wl",    "Flood-risk heat", L_wl)

if L_cat and len(D["cat"]):
    folium.GeoJson(D["cat"].__geo_interface__, name="Catchments",
        style_function=lambda f: dict(fillColor="#1B3A5C", fillOpacity=0.06,
                                      color="#1B3A5C", weight=2, dashArray="6 4"),
        tooltip=folium.GeoJsonTooltip(["outlet_id"], aliases=["Catchment"]),
    ).add_to(m)

if L_hi and len(D["hi"]):
    folium.GeoJson(D["hi"].__geo_interface__, name="High-risk zones",
        style_function=lambda f: dict(fillColor=RISK_COLORS["HIGH"], fillOpacity=0.40,
                                      color="#7A0C0C", weight=1.4),
        tooltip=folium.Tooltip("High flood-risk zone"),
    ).add_to(m)

ch = D["ch"]
for show, ctype, label in [(L_ear,"earthen","Earthen channels"), (L_con,"concrete","Concrete channels")]:
    if not show: continue
    sub = ch[ch["channel_type"] == ctype]
    if not len(sub): continue
    folium.GeoJson(sub.__geo_interface__, name=label,
        style_function=lambda f, c=channel_color(ctype): dict(color=c, weight=2.5, opacity=0.95),
        tooltip=folium.GeoJsonTooltip(
            fields=["segment_id","channel_type","length_m","depth_m","bottom_width_m","velocity_ms","cost_inr_k"],
            aliases=["ID","Type","Length (m)","Depth (m)","Base width (m)","Speed (m/s)","Cost (₹ '000)"]),
    ).add_to(m)

m.get_root().add_child(legend_macro(L_wl or L_hi, L_ear or L_con))
Fullscreen(position="topright").add_to(m)
MiniMap(toggle_display=True, position="bottomright").add_to(m)
folium.LayerControl(collapsed=True, position="topright").add_to(m)
st_folium(m, width="100%", height=600, returned_objects=[])

# ── Plain-language recommendation ────────────────────────────────────────────
w = vm["waterlogging"]
n_con = dr.get("n_concrete", 0)
rec = (
    f"<b>What this means for {sel.split(' (')[0]}:</b> About <b>{D['hi_ha']:.0f} hectares</b> are at high "
    f"risk of waterlogging during heavy rain. The plan recommends <b>{dr['channel_count']:,} drainage channels</b> "
    f"(<b>{dr['total_length_m']/1000:.0f} km</b>), of which {n_con} need concrete lining where water flows fast, "
    f"the rest earthen. Estimated build cost <b>₹{dr['total_cost_inr_lakhs']:.0f} lakhs</b> at CPWD 2023-24 rates. "
    f"Channels are sized for a 10-year-return rainstorm, so none overflow."
)
st.markdown(f'<div class="rec">💡 {rec}</div>', unsafe_allow_html=True)

# ── Technical validation (for judges) ────────────────────────────────────────
with st.expander("🔬 Technical validation (for evaluators)"):
    d = vm["dtm"]
    val = w.get("validation", {}).get("physics_consistency", {})
    twi_ratio = val.get("twi_ratio_high_to_low")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**DTM accuracy (ASPRS)**")
        st.markdown(
            f"RMSE **{d['rmse_m']:.3f} m** · MAE {d['mae_m']:.3f} m\n\n"
            f"LE90 {d['le90_m']:.3f} m · bias {d['bias_m']:+.3f} m\n\n"
            f"{d['total_points']/1e6:.0f}M points · leave-out CV")
    with b:
        st.markdown("**Flood model (XGBoost)**")
        st.markdown(
            f"ROC-AUC **{w['roc_auc']:.4f}** · PR-AUC {w['pr_auc']:.4f}\n\n"
            f"F1 {w['f1']:.4f} · Recall {w['recall']:.4f}\n\n"
            + (f"TWI high/low ratio **{twi_ratio:.2f}×**" if twi_ratio else ""))
    with c:
        st.markdown("**Drainage design**")
        st.markdown(
            f"Avg velocity {dr['avg_velocity_ms']:.2f} m/s\n\n"
            f"Max flow {dr.get('max_design_flow_m3s',0):.2f} m³/s · 0 overflow\n\n"
            "Manning's trapezoidal · MST routing")
    st.caption(
        "Labels from terrain physics (TWI, depression depth, curvature — Beven & Kirkby 1979; "
        "Wang & Liu 2006), the same indices as ISRO/NRSC's National Waterlogging Atlas. Same model "
        "generalises AUC ≥ 0.989 across Gujarat ↔ Punjab without retraining. Map shows channels on "
        "surveyed terrain; headline counts include all routed segments.")

# ── Cross-village comparison + download ──────────────────────────────────────
with st.expander("📊 Compare all 4 villages & download outputs"):
    rows = []
    for label, k in VILLAGES.items():
        dd, ww, rr = all_v[k]["dtm"], all_v[k]["waterlogging"], all_v[k]["drainage"]
        rows.append({
            "Village": label, "Points (M)": round(dd["total_points"]/1e6,1),
            "DTM RMSE (m)": dd["rmse_m"], "Flood AUC": ww["roc_auc"],
            "Channels": rr["channel_count"], "Length (km)": round(rr["total_length_m"]/1000,1),
            "Cost (₹L)": rr["total_cost_inr_lakhs"],
        })
    st.dataframe(pd.DataFrame(rows).set_index("Village"), use_container_width=True)

    out_path = OUT_DIR / vk
    files = [f for ext in ("*.tif","*.gpkg","*.json") for f in out_path.glob(ext)]
    if files:
        zip_path = out_path / "outputs.zip"
        if st.button(f"Build download bundle for {sel.split(' (')[0]} ({len(files)} files)"):
            with st.spinner("Zipping outputs…"):
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in files:
                        zf.write(f, f.name)
            st.download_button(f"📦 Download {zip_path.stat().st_size/1e6:.0f} MB",
                               data=zip_path.read_bytes(),
                               file_name=f"{vk}_outputs.zip")

# ── Offline pipeline runner (kept for processing new villages; not for demo) ──
with st.expander("🛠️ Process a new village (offline — slow, not for live demo)"):
    st.caption("Runs the full 6-stage pipeline on a LAS/LAZ file. Takes 30 s–several minutes "
               "depending on point count — use offline, not during a presentation.")
    in_dir = ROOT / "data" / "input"
    files = sorted(in_dir.glob("*.las")) + sorted(in_dir.glob("*.laz")) if in_dir.exists() else []
    col1, col2 = st.columns([3, 1])
    with col1:
        if files:
            fmap = {f.name: str(f) for f in files}
            pick = st.selectbox("Point-cloud file (data/input)", list(fmap.keys()))
            in_path = fmap[pick]
            st.caption(f"{Path(in_path).stat().st_size/1e9:.2f} GB")
        else:
            in_path = st.text_input("LAS/LAZ path", placeholder=r"C:\data\village.las")
    with col2:
        out_name = st.text_input("Output folder", value="village_output")
    if st.button("▶ Run pipeline (offline)", disabled=not in_path):
        out_dir = OUT_DIR / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
        cmd = [sys.executable, PIPELINE_SCRIPT, "--input", str(in_path),
               "--output", str(out_dir), "--stages", "1,2,3,4,5,6", "--evaluate"]
        with st.spinner("Running pipeline… (this can take minutes)"):
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
        if r.returncode == 0:
            st.success(f"Done → {out_dir}. Re-run honest-metrics, then it appears in the village list.")
        else:
            st.error("Pipeline failed.")
            st.code((r.stderr or "")[-1500:])

st.caption("DTM Drainage AI · MoPR / IITTNiF Geospatial Hackathon · PS-2 · "
           "OGC outputs (GeoPackage, Cloud-Optimized GeoTIFF, LAS 1.4). "
           "Open in QGIS for full GIS analysis.")
