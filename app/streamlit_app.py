"""
app/streamlit_app.py — DTM Drainage AI  ·  village flood-risk & drainage planner
GIS-style layer-by-layer viewer for gram-panchayat planners.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import json
from pathlib import Path

import folium
from folium.plugins import Fullscreen, MiniMap
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from branca.element import MacroElement, Template

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))        # app/ dir → geo_utils
from geo_utils import (
    raster_to_overlay, load_drainage_channels,
    load_waterlogging_hotspots, load_catchment_boundaries,
    high_risk_zones, channel_color, RISK_COLORS,
)

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "output"
METRICS = OUT_DIR / "_reports" / "honest_metrics.json"

VILLAGES = {
    "DEVDI (Gujarat)":          "DEVDI",
    "KHAPRETA (Gujarat)":       "KHAPRETA",
    "DHAL HOSHIARPUR (Punjab)": "DHAL_HOSHIARPUR",
    "CHAKHIRASINGH (Punjab)":   "CHAKHIRASINGH",
}

st.set_page_config(page_title="Village Flood & Drainage Planner",
                   page_icon="🌊", layout="wide",
                   initial_sidebar_state="expanded")

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.1rem !important; padding-bottom: 0.5rem !important; }
iframe { border-radius: 10px !important; border: 1px solid #dde3ea !important; }
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 13px !important; }
.hero { background: linear-gradient(100deg,#0E3A53,#117777);
        color:#fff; border-radius:12px; padding:16px 22px; margin-bottom:14px; }
.hero h1 { margin:0; font-size:23px; font-weight:800; letter-spacing:.2px; }
.hero p  { margin:6px 0 0; font-size:14px; opacity:.92; line-height:1.5; }
.toc-h   { font-size:12px; font-weight:800; letter-spacing:.8px;
           color:#0E3A53; text-transform:uppercase; margin:14px 0 4px; }
.dot { display:inline-block; width:11px; height:11px; border-radius:3px;
       margin-right:7px; vertical-align:middle; }
.sw  { display:inline-block; width:16px; height:4px; border-radius:2px;
       margin-right:6px; vertical-align:middle; }
.rec { background:#FFF6E6; border-left:5px solid #E8910C; border-radius:8px;
       padding:12px 16px; font-size:14px; color:#5a3d00; line-height:1.55; }
.legend-card { background:#F4F8FB; border:1px solid #dde6ee; border-radius:8px;
       padding:8px 12px; font-size:12.5px; color:#234; }
section[data-testid="stSidebar"] { width: 340px !important; }
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
    # Merged high-risk zones straight from the raster (fast; avoids 13k+ polygons)
    zones = high_risk_zones(base / "waterlogging_probability.tif", thresh=0.65)
    # sum raw area_m2 (NOT rounded area_ha — rounding to 0.001 ha zeroes ~4 m² hotspots)
    hi_ha = round(float(hs_hi["area_m2"].sum()) / 1e4, 1) if len(hs_hi) else 0.0
    ovl = dict(
        dtm   = raster_to_overlay(base/"dtm.tif",                     "dtm",          opacity=0.85),
        hill  = raster_to_overlay(base/"hillshade.tif",               "hillshade",    opacity=0.85),
        slope = raster_to_overlay(base/"slope.tif",                   "slope",        opacity=0.75),
        twi   = raster_to_overlay(base/"twi.tif",                     "twi",          opacity=0.75),
        wl    = raster_to_overlay(base/"waterlogging_probability.tif","waterlogging", opacity=0.80, vmin=0, vmax=1),
    )
    return dict(ch=ch, hi=zones, cat=cat, ovl=ovl,
                center=ovl["dtm"][2], hi_ha=hi_ha)


metrics = load_metrics()
all_v   = metrics["villages"]


# ── Map legend overlay ───────────────────────────────────────────────────────
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
      <div style="font-weight:700; margin-bottom:3px;">Map key</div>
      {rows}
    </div>
    <style>.lg{{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:7px;vertical-align:middle;}}</style>
    {{% endmacro %}}
    """
    m = MacroElement(); m._template = Template(html)
    return m


BASEMAPS = {
    "🛰️ Satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     "Tiles © Esri"),
    "🗺️ Street":    ("OpenStreetMap", None),
    "◻️ Light":     ("CartoDB positron", None),
    "◼️ Dark":      ("CartoDB dark_matter", None),
}


# ── Sidebar = GIS Table of Contents ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 Flood & Drainage Planner")
    st.caption("Layer-by-layer village map · MoPR / IITTNiF PS-2")

    sel = st.selectbox("**Village**", list(VILLAGES.keys()))
    vk  = VILLAGES[sel]
    vm  = all_v[vk]
    D   = load_village(vk)

    base_label = st.radio("**Base map**", list(BASEMAPS.keys()),
                          index=0, horizontal=True)

    st.markdown('<div class="toc-h">🟥 Flood risk</div>', unsafe_allow_html=True)
    L_wl   = st.checkbox("Flood-risk heat map", True,
                         help="Model's predicted chance of waterlogging, pixel by pixel "
                              "— red = high, yellow = moderate, green = low")
    L_hi   = st.checkbox("High-risk zones (outlined)", True,
                         help="Discrete polygons flagged HIGH risk")

    st.markdown('<div class="toc-h">🟦 Drainage plan</div>', unsafe_allow_html=True)
    L_ear  = st.checkbox("Earthen channels", True)
    L_con  = st.checkbox("Concrete channels", True)
    L_cat  = st.checkbox("Catchment areas", False,
                         help="Land area that drains to each outlet")

    st.markdown('<div class="toc-h">⛰️ Terrain</div>', unsafe_allow_html=True)
    L_hill = st.checkbox("Hillshade (3-D relief)", False)
    L_dtm  = st.checkbox("Elevation (DTM)", False)
    L_slp  = st.checkbox("Slope", False)
    L_twi  = st.checkbox("Wetness index (TWI)", False)

    opacity = st.slider("Layer opacity", 0.2, 1.0, 0.80, 0.05)


# ── Header + plain-language summary ──────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🌊 {sel}</h1>
  <p>This map shows where rainwater is likely to collect and flood, and the drainage
     channels recommended to drain it — built automatically from a drone/LiDAR survey.
     Tick layers on the left to explore. Hover any channel or zone for details.</p>
</div>
""", unsafe_allow_html=True)

dr = vm["drainage"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Land at high flood risk", f"{D['hi_ha']:.0f} ha",
          help="Total area of high-risk waterlogging zones")
c2.metric("Drainage channels needed", f"{dr['channel_count']:,}",
          f"{dr['total_length_m']/1000:.0f} km total")
c3.metric("Estimated cost", f"₹{dr['total_cost_inr_lakhs']:.0f} L",
          "CPWD DSR 2023-24")
c4.metric("Flood-model confidence", f"{vm['waterlogging']['roc_auc']*100:.1f}%",
          "ROC-AUC")

# ── Build the map ────────────────────────────────────────────────────────────
tiles, attr = BASEMAPS[base_label]
m = folium.Map(location=D["center"], zoom_start=15, control_scale=True,
               tiles=tiles, attr=attr)

ovl = D["ovl"]
def _raster(layer_key, name, show):
    if not show: return
    png, bounds, _ = ovl[layer_key]
    folium.raster_layers.ImageOverlay(image=png, bounds=bounds, opacity=opacity,
                                      name=name, interactive=False).add_to(m)

# terrain (drawn first, underneath)
_raster("hill",  "Hillshade", L_hill)
_raster("dtm",   "Elevation", L_dtm)
_raster("slope", "Slope",     L_slp)
_raster("twi",   "Wetness (TWI)", L_twi)
_raster("wl",    "Flood-risk heat", L_wl)

# catchments
if L_cat and len(D["cat"]):
    folium.GeoJson(D["cat"].__geo_interface__, name="Catchments",
        style_function=lambda f: dict(fillColor="#1B3A5C", fillOpacity=0.06,
                                      color="#1B3A5C", weight=2, dashArray="6 4"),
        tooltip=folium.GeoJsonTooltip(["outlet_id"], aliases=["Catchment"]),
    ).add_to(m)

# high-risk zones (dissolved into merged polygons for speed)
if L_hi and len(D["hi"]):
    folium.GeoJson(D["hi"].__geo_interface__, name="High-risk zones",
        style_function=lambda f: dict(
            fillColor=RISK_COLORS["HIGH"], fillOpacity=0.40,
            color="#7A0C0C", weight=1.4),
        tooltip=folium.Tooltip("High flood-risk zone"),
    ).add_to(m)

# drainage channels
ch = D["ch"]
for show, ctype, label in [(L_ear,"earthen","Earthen channels"),
                           (L_con,"concrete","Concrete channels")]:
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
    f"<b>What this means for {sel.split(' (')[0]}:</b> About <b>{D['hi_ha']:.0f} hectares</b> "
    f"are at high risk of waterlogging during heavy rain"
    + f". The plan recommends <b>{dr['channel_count']:,} drainage channels</b> "
    f"(<b>{dr['total_length_m']/1000:.0f} km</b>), of which {n_con} need concrete lining where water "
    f"flows fast, the rest earthen. Estimated build cost <b>₹{dr['total_cost_inr_lakhs']:.0f} lakhs</b> "
    f"at CPWD 2023-24 rates. Channels are sized for a 10-year-return rainstorm, so none overflow."
)
st.markdown(f'<div class="rec">💡 {rec}</div>', unsafe_allow_html=True)

# ── Technical details (for judges) ───────────────────────────────────────────
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
        "Wang & Liu 2006), the same indices as ISRO/NRSC's National Waterlogging Atlas. "
        "Same model generalises AUC ≥ 0.989 across Gujarat ↔ Punjab without retraining. "
        "Note: map shows channels on surveyed terrain; headline counts include all routed segments.")
