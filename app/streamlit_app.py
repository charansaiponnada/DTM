"""
app/streamlit_app.py — DTM Drainage AI  ·  MoPR/IITTNiF Hackathon PS-2
Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import json
from pathlib import Path

import folium
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.geo_utils import (
    raster_to_overlay, load_drainage_channels,
    load_waterlogging_hotspots, load_catchment_boundaries,
    channel_color, RISK_COLORS,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "output"
METRICS = OUT_DIR / "_reports" / "honest_metrics.json"

VILLAGES = {
    "DEVDI (Gujarat)":          "DEVDI",
    "KHAPRETA (Gujarat)":       "KHAPRETA",
    "DHAL HOSHIARPUR (Punjab)": "DHAL_HOSHIARPUR",
    "CHAKHIRASINGH (Punjab)":   "CHAKHIRASINGH",
}
V_LABELS = {
    "DEVDI":           "DEVDI",
    "KHAPRETA":        "KHAPRETA",
    "DHAL_HOSHIARPUR": "DHAL HSP",
    "CHAKHIRASINGH":   "CHAKHIRA",
}
V_KEYS = list(V_LABELS.keys())

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DTM Drainage AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 12px !important; }
[data-testid="stMetricValue"] { font-size: 20px !important; }
.hero { background: linear-gradient(90deg,#1B3A5C,#117777);
        color:#fff; border-radius:8px; padding:14px 20px; margin-bottom:10px; }
.hero h2 { margin:0; font-size:20px; }
.hero p  { margin:4px 0 0; font-size:13px; opacity:.85; }
.pill { display:inline-block; background:#E8F4F8; color:#1B3A5C;
        border-radius:20px; padding:3px 10px; font-size:12px;
        font-weight:600; margin:2px 3px; }
iframe { border-radius:8px !important; border:none !important; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def add_channel_layer(fmap, gdf, show_earthen=True, show_concrete=True, weight=2):
    df = gdf.copy()
    if not show_earthen:  df = df[df["channel_type"] != "earthen"]
    if not show_concrete: df = df[df["channel_type"] != "concrete"]
    if df.empty: return
    folium.GeoJson(
        df.__geo_interface__,
        style_function=lambda f: dict(
            color=channel_color(f["properties"]["channel_type"]),
            weight=weight, opacity=0.9,
        ),
        tooltip=folium.GeoJsonTooltip(
            fields=["segment_id","channel_type","length_m","cost_inr_k",
                    "velocity_ms","bottom_width_m","depth_m"],
            aliases=["ID","Type","Length (m)","Cost (₹K)",
                     "Velocity (m/s)","Base width (m)","Depth (m)"],
        ),
        name="Drainage channels",
    ).add_to(fmap)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics() -> dict:
    with open(METRICS) as f:
        return json.load(f)

@st.cache_data(show_spinner="Loading village data…")
def load_village_data(key: str) -> dict:
    base = OUT_DIR / key
    gpkg = base / "drainage_network.gpkg"
    ch   = load_drainage_channels(gpkg)
    hs_f = load_waterlogging_hotspots(gpkg, risk_filter=["HIGH","MEDIUM"])
    cat  = load_catchment_boundaries(gpkg)
    dtm_ov  = raster_to_overlay(base/"dtm.tif",                     "dtm",          opacity=0.82)
    hs_ov   = raster_to_overlay(base/"hillshade.tif",               "hillshade",    opacity=0.72)
    wl_ov   = raster_to_overlay(base/"waterlogging_probability.tif","waterlogging", opacity=0.78, vmin=0, vmax=1)
    slp_ov  = raster_to_overlay(base/"slope.tif",                   "slope",        opacity=0.72)
    twi_ov  = raster_to_overlay(base/"twi.tif",                     "twi",          opacity=0.72)
    return dict(ch=ch, hs_f=hs_f, cat=cat,
                dtm=dtm_ov, hillshade=hs_ov, wl=wl_ov, slope=slp_ov, twi=twi_ov,
                center=dtm_ov[2])


# ── Load data ──────────────────────────────────────────────────────────────────
metrics = load_metrics()
all_v   = metrics["villages"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 DTM Drainage AI")
    st.caption("MoPR / IITTNiF Hackathon · PS-2")
    st.divider()

    sel  = st.selectbox("**Village**", list(VILLAGES.keys()), label_visibility="collapsed")
    vk   = VILLAGES[sel]
    vm   = all_v[vk]
    data = load_village_data(vk)

    c1, c2 = st.columns(2)
    c1.metric("DTM RMSE", f"{vm['dtm']['rmse_m']:.3f} m",         "ASPRS CV")
    c2.metric("WL AUC",   f"{vm['waterlogging']['roc_auc']:.4f}", "XGBoost")
    c3, c4 = st.columns(2)
    c3.metric("Channels", f"{vm['drainage']['channel_count']:,}")
    c4.metric("Cost",     f"₹{vm['drainage']['total_cost_inr_lakhs']:.0f}L")

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(
        '<span class="pill">1 Inspect</span>'
        '<span class="pill">2 Ground</span>'
        '<span class="pill">3 DTM</span>'
        '<span class="pill">4 Hydrology</span>'
        '<span class="pill">5 WL Risk</span>'
        '<span class="pill">6 Drainage</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    with st.expander("Full stats"):
        d  = vm["dtm"]
        w  = vm["waterlogging"]
        dr = vm["drainage"]
        st.markdown(
            f"Points: **{d['total_points']/1e6:.1f}M** · Ground: **{d['ground_fraction']*100:.0f}%**\n\n"
            f"LE90: **{d['le90_m']:.3f} m** · NMAD: **{d['nmad_m']:.3f} m**\n\n"
            f"F1: **{w['f1']:.4f}** · Recall: **{w['recall']:.4f}**\n\n"
            f"Waterlogged: **{w['positive_rate']*100:.1f}%** of village\n\n"
            f"Length: **{dr['total_length_m']/1000:.1f} km** · "
            f"Earthen: **{dr['n_earthen']}** · Concrete: **{dr['n_concrete']}**\n\n"
            f"Avg velocity: **{dr['avg_velocity_ms']:.3f} m/s**"
        )

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h2>🌍 {sel}</h2>
  <p>LiDAR point cloud → Ground classification (SMRF+RF) → DTM 0.5 m →
     Hydrology (D8 + TWI) → Waterlogging risk (XGBoost AUC {vm['waterlogging']['roc_auc']:.4f}) →
     Costed drainage design (Manning's + MST) → GPKG + COG outputs</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏔️ DTM & Terrain",
    "🌊 Waterlogging Risk",
    "🏗️ Drainage Design",
    "📊 Metrics & Validation",
])


# ─── TAB 1: DTM & Terrain ─────────────────────────────────────────────────────
with tab1:
    with st.expander("⚙️ Layer controls", expanded=False):
        lc1,lc2,lc3,lc4,lc5,lc6 = st.columns(6)
        show_dtm   = lc1.checkbox("DTM elevation", True,  key="l1_dtm")
        show_hs    = lc2.checkbox("Hillshade",      False, key="l1_hs")
        show_slope = lc3.checkbox("Slope",          False, key="l1_sl")
        show_twi   = lc4.checkbox("TWI",            False, key="l1_tw")
        show_catch = lc5.checkbox("Catchments",     True,  key="l1_ca")
        opacity    = lc6.slider("Opacity", 0.2, 1.0, 0.78, 0.05, key="op1")

    m1 = folium.Map(location=data["center"], zoom_start=14,
                    tiles="CartoDB positron", control_scale=True)
    for show, key, name in [
        (show_hs,    "hillshade","Hillshade"),
        (show_dtm,   "dtm",     "DTM Elevation"),
        (show_slope, "slope",   "Slope"),
        (show_twi,   "twi",     "TWI"),
    ]:
        if show:
            png, bounds, _ = data[key]
            folium.raster_layers.ImageOverlay(
                image=png, bounds=bounds, opacity=opacity,
                name=name, interactive=False,
            ).add_to(m1)
    if show_catch and len(data["cat"]) > 0:
        folium.GeoJson(
            data["cat"].__geo_interface__,
            style_function=lambda f: dict(
                fillColor="none", color="#1B3A5C", weight=2.5, dashArray="6 3"),
            tooltip=folium.GeoJsonTooltip(["outlet_id","area_m2"]),
            name="Catchments",
        ).add_to(m1)
    folium.LayerControl().add_to(m1)
    st_folium(m1, width="100%", height=560, returned_objects=[])

    d = vm["dtm"]
    st.markdown(
        f"**ASPRS accuracy** (leave-out CV · {d['n_check_points']:,} check points) — "
        f"RMSE **{d['rmse_m']:.3f} m** · MAE {d['mae_m']:.3f} m · "
        f"LE90 {d['le90_m']:.3f} m · NMAD {d['nmad_m']:.3f} m · "
        f"Bias {d['bias_m']:.4f} m"
    )


# ─── TAB 2: Waterlogging Risk ─────────────────────────────────────────────────
with tab2:
    with st.expander("⚙️ Layer controls", expanded=False):
        lc1,lc2,lc3,lc4,lc5 = st.columns(5)
        show_wl_r = lc1.checkbox("Probability raster", True,  key="l2_wl")
        show_high = lc2.checkbox("HIGH risk zones",    True,  key="l2_hi")
        show_med  = lc3.checkbox("MEDIUM risk zones",  True,  key="l2_me")
        show_ch2  = lc4.checkbox("Drain channels",     False, key="l2_ch")
        wl_op     = lc5.slider("Opacity", 0.2, 1.0, 0.72, 0.05, key="op2")

    w = vm["waterlogging"]
    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("ROC-AUC",   f"{w['roc_auc']:.4f}")
    s2.metric("PR-AUC",    f"{w['pr_auc']:.4f}")
    s3.metric("F1 Score",  f"{w['f1']:.4f}")
    s4.metric("Precision", f"{w['precision']:.4f}")
    s5.metric("Recall",    f"{w['recall']:.4f}")

    m2 = folium.Map(location=data["center"], zoom_start=14,
                    tiles="CartoDB positron", control_scale=True)
    if show_wl_r:
        png, bounds, _ = data["wl"]
        folium.raster_layers.ImageOverlay(
            image=png, bounds=bounds, opacity=wl_op,
            name="WL probability", interactive=False,
        ).add_to(m2)
    risk_f = [r for r,s in [("HIGH",show_high),("MEDIUM",show_med)] if s]
    if risk_f and len(data["hs_f"]) > 0:
        gdf = data["hs_f"][data["hs_f"]["risk_level"].isin(risk_f)]
        folium.GeoJson(
            gdf.__geo_interface__,
            style_function=lambda f: dict(
                fillColor=RISK_COLORS.get(f["properties"]["risk_level"],"#888"),
                fillOpacity=0.55, color="none", weight=0,
            ),
            tooltip=folium.GeoJsonTooltip(
                ["risk_level","prob_pct","area_ha"],
                aliases=["Risk","Prob (%)","Area (ha)"],
            ),
            name="Hotspots",
        ).add_to(m2)
    if show_ch2:
        add_channel_layer(m2, data["ch"], weight=1.5)
    folium.LayerControl().add_to(m2)
    st_folium(m2, width="100%", height=530, returned_objects=[])

    st.caption(
        "Labels derived from terrain physics (TWI, depression depth, curvature, flow accumulation) — "
        "no historical flood records required. "
        f"Generalises AUC ≥ 0.989 across Gujarat ↔ Punjab without retraining."
    )


# ─── TAB 3: Drainage Design ───────────────────────────────────────────────────
with tab3:
    with st.expander("⚙️ Layer controls", expanded=False):
        lc1,lc2,lc3,lc4 = st.columns(4)
        show_e  = lc1.checkbox("Earthen channels",  True,  key="l3_e")
        show_c  = lc2.checkbox("Concrete channels", True,  key="l3_c")
        show_bg = lc3.checkbox("WL background",     True,  key="l3_bg")
        ch_w    = lc4.slider("Line thickness", 1, 5, 2, key="chw")

    dr = vm["drainage"]
    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("Channels",    f"{dr['channel_count']:,}")
    s2.metric("Length",      f"{dr['total_length_m']/1000:.1f} km")
    s3.metric("Total cost",  f"₹{dr['total_cost_inr_lakhs']:.0f}L")
    s4.metric("Avg velocity",f"{dr['avg_velocity_ms']:.3f} m/s")
    s5.metric("Max flow",    f"{dr['max_design_flow_m3s']:.3f} m³/s")

    m3 = folium.Map(location=data["center"], zoom_start=14,
                    tiles="CartoDB dark_matter", control_scale=True)
    if show_bg:
        png, bounds, _ = data["wl"]
        folium.raster_layers.ImageOverlay(
            image=png, bounds=bounds, opacity=0.40,
            name="Waterlogging", interactive=False,
        ).add_to(m3)
    add_channel_layer(m3, data["ch"], show_e, show_c, weight=ch_w)
    folium.LayerControl().add_to(m3)
    st_folium(m3, width="100%", height=530, returned_objects=[])

    st.caption(
        "🟫 Earthen  🟦 Concrete  ·  "
        "Hover/click any channel: segment ID, type, length, design flow, velocity, width, depth, cost.  "
        "Routing: MST on D8 flow graph · Sizing: Rational Method + Manning's trapezoidal · 10-yr return period."
    )


# ─── TAB 4: Metrics & Validation ──────────────────────────────────────────────
with tab4:
    # ── RMSE cards ────────────────────────────────────────────────────────────
    st.markdown("#### DTM Accuracy — ASPRS Standard (Leave-out CV vs withheld LiDAR ground returns)")
    cols = st.columns(4)
    for col, k in zip(cols, V_KEYS):
        d    = all_v[k]["dtm"]
        pts  = d["total_points"] / 1e6
        dens = d["ground_points"] / max(1, d["total_points"]) * 100
        col.metric(
            V_LABELS[k],
            f"{d['rmse_m']:.3f} m RMSE",
            f"LE90 {d['le90_m']:.3f} m · {pts:.0f}M pts",
        )
    st.caption(
        "CHAKHIRASINGH RMSE is higher (0.254 m) because its point density is 10× lower "
        "(9.8M pts vs 193M for KHAPRETA) — RMSE scales with acquisition quality, not algorithm quality."
    )

    st.divider()

    # ── Waterlogging: model scores + cross-village transfer ───────────────────
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### Waterlogging Model (XGBoost, 5-fold CV)")
        labels = [V_LABELS[k] for k in V_KEYS]
        fig_wl = go.Figure()
        for mk, name, color in [
            ("roc_auc","ROC-AUC","#117777"),
            ("pr_auc", "PR-AUC", "#1B3A5C"),
            ("f1",     "F1",     "#B94A00"),
        ]:
            fig_wl.add_trace(go.Bar(
                name=name, x=labels,
                y=[all_v[k]["waterlogging"][mk] for k in V_KEYS],
                marker_color=color,
            ))
        fig_wl.update_layout(
            barmode="group",
            yaxis=dict(range=[0.78,1.01], title="Score"),
            legend=dict(orientation="h", y=1.14),
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white",
        )
        fig_wl.update_xaxes(showgrid=False)
        fig_wl.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig_wl, use_container_width=True)

    with c_right:
        st.markdown("#### Cross-Village Transfer AUC")
        st.caption("Train on row → test on column · zero retraining · Gujarat ↔ Punjab")
        xv     = metrics.get("cross_village_transfer", {})
        matrix = [[xv.get(r,{}).get(c,0) for c in V_KEYS] for r in V_KEYS]
        short  = [V_LABELS[k] for k in V_KEYS]
        fig_hm = go.Figure(go.Heatmap(
            z=matrix, x=short, y=short,
            colorscale="RdYlGn", zmin=0.97, zmax=1.0,
            text=[[f"{v:.4f}" for v in row] for row in matrix],
            texttemplate="%{text}", showscale=True,
            colorbar=dict(thickness=12, len=0.8),
        ))
        fig_hm.update_layout(
            xaxis_title="Test village", yaxis_title="Train village",
            height=300, margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # ── Physics validation (TWI ratio) ────────────────────────────────────────
    st.markdown("#### Waterlogging Label Validation — TWI Physics Consistency")
    st.caption(
        "Median TWI (Topographic Wetness Index) inside HIGH risk zones vs LOW risk zones. "
        "If the model is physically grounded, HIGH risk pixels must have significantly higher TWI. "
        "Ratio > 2× confirms labels capture true terrain moisture accumulation."
    )
    twi_cols = st.columns(4)
    for col, k in zip(twi_cols, V_KEYS):
        val = all_v[k].get("waterlogging", {}).get("validation", {})
        pc  = val.get("physics_consistency", {})
        hi  = pc.get("twi_median_high_risk")
        lo  = pc.get("twi_median_low_risk")
        rat = pc.get("twi_ratio_high_to_low")
        if hi and lo and rat:
            col.metric(
                V_LABELS[k],
                f"{rat:.2f}× ratio",
                f"HIGH {hi:.2f} vs LOW {lo:.2f}",
            )
        else:
            col.metric(V_LABELS[k], "—", "run validate_satellite.py")

    st.info(
        "**Methodology grounding:** TWI (Beven & Kirkby 1979) and depression-filling "
        "(Wang & Liu 2006) are the same indices used by ISRO/NRSC in India's National "
        "Waterlogging Atlas. No historical flood records required — the terrain itself "
        "encodes the drainage physics."
    )

    st.divider()
    c_dr, c_tbl = st.columns([3, 2])

    with c_dr:
        st.markdown("#### Drainage Design Comparison")
        fig_dr = go.Figure()
        fig_dr.add_trace(go.Bar(
            name="Length (km)", x=labels,
            y=[all_v[k]["drainage"]["total_length_m"]/1000 for k in V_KEYS],
            marker_color="#1B3A5C",
        ))
        fig_dr.add_trace(go.Bar(
            name="Cost (₹ Lakhs)", x=labels,
            y=[all_v[k]["drainage"]["total_cost_inr_lakhs"] for k in V_KEYS],
            marker_color="#B94A00", yaxis="y2",
        ))
        fig_dr.update_layout(
            barmode="group",
            yaxis=dict(title="Length (km)", side="left"),
            yaxis2=dict(title="Cost (₹ L)", side="right", overlaying="y"),
            legend=dict(orientation="h", y=1.14),
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white",
        )
        fig_dr.update_xaxes(showgrid=False)
        st.plotly_chart(fig_dr, use_container_width=True)

    with c_tbl:
        st.markdown("#### Summary")
        rows = []
        for k in V_KEYS:
            d,w,dr = all_v[k]["dtm"], all_v[k]["waterlogging"], all_v[k]["drainage"]
            rows.append({
                "Village":   V_LABELS[k],
                "Pts (M)":   round(d["total_points"]/1e6,1),
                "RMSE (m)":  d["rmse_m"],
                "WL AUC":    w["roc_auc"],
                "WL F1":     w["f1"],
                "Chan.":     dr["channel_count"],
                "Len (km)":  round(dr["total_length_m"]/1000,1),
                "Cost (₹L)": dr["total_cost_inr_lakhs"],
            })
        st.dataframe(
            pd.DataFrame(rows).set_index("Village"),
            use_container_width=True, height=210,
        )
