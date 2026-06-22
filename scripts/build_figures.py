"""
scripts/build_figures.py
────────────────────────
Generate honest, presentation-ready figures from honest_metrics.json and the
per-village rasters/vectors. Writes PNGs into docs/images/.

Figures
-------
  fig_dtm_accuracy.png   grouped bars: RMSE / MAE / LE90 / NMAD per village
  fig_wl_transfer.png    cross-village transfer AUC heatmap + per-village CV AUC
  fig_map_<village>.png   hillshade + drainage channels + waterlogging hotspots
  fig_scorecard.png      multi-village honest summary scorecard
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "output"
IMG = ROOT / "docs" / "images"
IMG.mkdir(parents=True, exist_ok=True)

NAVY = "#2C3E50"; SIENNA = "#A0522D"; BROWN = "#8B7355"
TEAL = "#178582"; AMBER = "#D35400"; CREAM = "#F5F0E8"
PALETTE = [NAVY, SIENNA, TEAL, AMBER, BROWN]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#888",
                     "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"})

NICE = {"DEVDI": "Devdi (GJ)", "KHAPRETA": "Khapreta (GJ)",
        "DHAL_HOSHIARPUR": "Dhal Hoshiarpur (PB)", "DHUNDA": "Dhunda (PB)",
        "CHAKHIRASINGH": "Chakhirasingh (PB)"}


def load():
    p = OUT / "_reports" / "honest_metrics.json"
    return json.loads(p.read_text())


# ────────────────────────────────────────────────────────────────────────
def fig_dtm_accuracy(rep):
    villages = [v for v in rep["villages"] if "rmse_m" in rep["villages"][v].get("dtm", {})]
    if not villages:
        print("  [skip] dtm_accuracy: no data"); return
    labels = [NICE.get(v, v) for v in villages]
    metrics = ["rmse_m", "mae_m", "le90_m", "nmad_m"]
    mlabels = ["RMSE", "MAE", "LE90", "NMAD"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(villages)); w = 0.2
    for i, (m, ml) in enumerate(zip(metrics, mlabels)):
        vals = [rep["villages"][v]["dtm"].get(m, 0) for v in villages]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=ml, color=PALETTE[i], edgecolor="white")
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Vertical error (m)", fontsize=11)
    ax.set_title("DTM Vertical Accuracy — Leave-Out CV vs Withheld LiDAR Ground Returns",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=12)
    ax.legend(ncol=4, frameon=False, fontsize=10, loc="upper left")
    ax.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.6)
    ax.text(len(villages)-0.5, 0.52, "0.5 m DTM cell size", fontsize=8, color="gray", ha="right")
    fig.tight_layout(); fig.savefig(IMG / "fig_dtm_accuracy.png", dpi=150); plt.close(fig)
    print("  [ok] fig_dtm_accuracy.png")


def fig_wl_transfer(rep):
    tm = rep.get("cross_village_transfer", {})
    tm = {a: r for a, r in tm.items() if isinstance(r, dict) and r}
    if not tm:
        print("  [skip] wl_transfer: no data"); return
    trains = list(tm.keys())
    tests = sorted({b for r in tm.values() for b in r})
    M = np.full((len(trains), len(tests)), np.nan)
    for i, a in enumerate(trains):
        for j, b in enumerate(tests):
            if b in tm[a]:
                M[i, j] = tm[a][b]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    cmap = mcolors.LinearSegmentedColormap.from_list("auc", ["#f2e9dc", TEAL, NAVY])
    im = ax1.imshow(M, cmap=cmap, vmin=0.95, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(len(tests))); ax1.set_xticklabels([NICE.get(t, t).split(" (")[0] for t in tests], rotation=30, ha="right", fontsize=9)
    ax1.set_yticks(range(len(trains))); ax1.set_yticklabels([NICE.get(t, t).split(" (")[0] for t in trains], fontsize=9)
    ax1.set_xlabel("Predicted on  →", fontsize=10); ax1.set_ylabel("Trained on  ↓", fontsize=10)
    for i in range(len(trains)):
        for j in range(len(tests)):
            if not np.isnan(M[i, j]):
                ax1.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                         color="white" if M[i, j] > 0.985 else "#333", fontsize=9, fontweight="bold")
    ax1.set_title("Cross-Village Transfer (ROC-AUC)\ntrain on one village → predict another",
                  fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="ROC-AUC")

    # right: per-village CV AUC + diagonal (self) vs off-diagonal (transfer)
    diag = [M[i, i] for i in range(min(len(trains), len(tests))) if not np.isnan(M[i, i])]
    off = [M[i, j] for i in range(len(trains)) for j in range(len(tests))
           if i != j and not np.isnan(M[i, j])]
    cvaucs, names = [], []
    for v in trains:
        wl = rep["villages"].get(v, {}).get("waterlogging", {})
        if "roc_auc" in wl:
            cvaucs.append(wl["roc_auc"]); names.append(NICE.get(v, v).split(" (")[0])
    xx = np.arange(len(names))
    bars = ax2.bar(xx, cvaucs, color=NAVY, edgecolor="white", width=0.6)
    for b, val in zip(bars, cvaucs):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9, color=NAVY, fontweight="bold")
    ax2.set_xticks(xx); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylim(0.5, 1.08); ax2.set_ylabel("ROC-AUC", fontsize=10)
    ax2.set_title("Within-Village 5-Fold CV Fidelity\nto physical risk index",
                  fontsize=12, fontweight="bold", color=NAVY, pad=10)
    if off:
        ax2.text(0.5, 0.985, f"mean cross-village transfer AUC = {np.mean(off):.3f}",
                 transform=ax2.transAxes, ha="center", fontsize=9, color=AMBER,
                 fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec=AMBER, alpha=0.9))
    fig.tight_layout(); fig.savefig(IMG / "fig_wl_transfer.png", dpi=150); plt.close(fig)
    print("  [ok] fig_wl_transfer.png")


def _valid_footprint(dtm_path, decim=4):
    """Polygon of the populated DTM area (to clip vector spurs in nodata corners)."""
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import from_bounds
    from shapely.geometry import shape as shp_shape
    from shapely.ops import unary_union
    with rasterio.open(dtm_path) as src:
        oh, ow = max(src.height // decim, 1), max(src.width // decim, 1)
        arr = src.read(1, out_shape=(oh, ow)).astype(float)
        nod = src.nodata
        mask = np.isfinite(arr) & (arr != 0)
        if nod is not None:
            mask &= (arr != nod)
        t = from_bounds(*src.bounds, ow, oh)
    polys = [shp_shape(g) for g, val in shapes(mask.astype("uint8"), mask=mask, transform=t) if val == 1]
    if not polys:
        return None
    u = unary_union(polys)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda p: p.area)
    return u.buffer(-1.5)


def fig_village_map(village):
    import rasterio
    import geopandas as gpd
    hs_path = OUT / village / "hillshade.tif"
    dtm_path = OUT / village / "dtm.tif"
    gpkg = OUT / village / "drainage_network.gpkg"
    base = hs_path if hs_path.exists() else dtm_path
    if not base.exists():
        print(f"  [skip] map {village}: no raster"); return
    with rasterio.open(base) as src:
        arr = src.read(1).astype(float)
        nod = src.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fp = None
    try:
        fp = _valid_footprint(dtm_path if dtm_path.exists() else base)
    except Exception as e:
        print(f"    footprint failed for {village}: {e}")

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.imshow(arr, cmap="Greys_r", extent=ext, origin="upper")
    # waterlogging hotspots
    try:
        hs = gpd.read_file(gpkg, layer="waterlogging_hotspots")
        if fp is not None and len(hs):
            hs = gpd.clip(hs, fp)
        if len(hs):
            hs.plot(ax=ax, color=AMBER, alpha=0.42, edgecolor="none", zorder=2)
    except Exception:
        pass
    # drainage channels — clipped to the populated footprint
    try:
        ch = gpd.read_file(gpkg, layer="drainage_channels")
        if fp is not None and len(ch):
            ch = gpd.clip(ch, fp)
        if len(ch):
            ch.plot(ax=ax, color="#1f6feb", linewidth=0.7, alpha=0.9, zorder=3)
    except Exception:
        pass
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"{NICE.get(village, village)} — Hillshade · Drainage Network · Waterlogging Hotspots",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlabel("Easting (m)", fontsize=9); ax.set_ylabel("Northing (m)", fontsize=9)
    ax.ticklabel_format(style="plain"); ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([0], [0], color="#1f6feb", lw=2, label="Designed drainage"),
                       Patch(facecolor=AMBER, alpha=0.5, label="Waterlogging hotspot")],
              loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(False)
    fig.tight_layout(); fig.savefig(IMG / f"fig_map_{village}.png", dpi=145); plt.close(fig)
    print(f"  [ok] fig_map_{village}.png")


def fig_drainage_hero(rep, village="DEVDI"):
    """DTM + designed drainage (coloured by channel type) with correct numbers."""
    import rasterio
    import geopandas as gpd
    d = OUT / village
    dtm_p = d / "dtm.tif"; gpkg = d / "drainage_network.gpkg"
    if not dtm_p.exists():
        print("  [skip] drainage hero: no dtm"); return
    with rasterio.open(dtm_p) as src:
        dem = src.read(1).astype(float)
        nod = src.nodata
        if nod is not None:
            dem[dem == nod] = np.nan
        dem[dem == 0] = np.nan
        ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    fp = None
    try:
        fp = _valid_footprint(dtm_p)
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ax.imshow(dem, cmap="terrain", extent=ext, origin="upper", alpha=0.92)
    try:
        ch = gpd.read_file(gpkg, layer="drainage_channels")
        if fp is not None and len(ch):
            ch = gpd.clip(ch, fp)
        if "channel_type" in ch.columns:
            ear = ch[ch.channel_type == "earthen"]; con = ch[ch.channel_type == "concrete"]
            if len(ear): ear.plot(ax=ax, color="#14366b", linewidth=0.7, alpha=0.9, zorder=3)
            if len(con): con.plot(ax=ax, color="#c0392b", linewidth=1.4, alpha=0.95, zorder=4)
        else:
            ch.plot(ax=ax, color="#14366b", linewidth=0.7, alpha=0.9, zorder=3)
    except Exception as e:
        print(f"    hero channels failed: {e}")
    dr = rep["villages"].get(village, {}).get("drainage", {})
    txt = (f"Channels: {dr.get('channel_count','–')}\n"
           f"Length: {dr.get('total_length_m',0)/1000:.1f} km\n"
           f"Cost: ₹{dr.get('total_cost_inr_lakhs',0):.0f} Lakh\n"
           f"Concrete upgrades: {dr.get('n_concrete','–')}\n"
           f"Capacity fails: {dr.get('capacity_exceeded_count','–')}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            family="DejaVu Sans", bbox=dict(boxstyle="round", fc="white", ec=NAVY, alpha=0.92))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color="#14366b",lw=2,label="Earthen channel"),
                       Line2D([0],[0],color="#c0392b",lw=2,label="Concrete (high-flow)")],
              loc="lower right", fontsize=9, framealpha=0.92)
    ax.set_title(f"{NICE.get(village,village)} — Designed Drainage Network over DTM",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlabel("Easting (m)", fontsize=9); ax.set_ylabel("Northing (m)", fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(False)
    if fp is not None:
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    fig.tight_layout(); fig.savefig(IMG / "fig_drainage.png", dpi=145); plt.close(fig)
    print("  [ok] fig_drainage.png (hero, real numbers)")


def fig_scorecard(rep):
    villages = list(rep["villages"].keys())
    rows = []
    for v in villages:
        r = rep["villages"][v]
        dt = r.get("dtm", {}); wl = r.get("waterlogging", {}); dr = r.get("drainage", {})
        st = r.get("dtm_stats", {})
        rows.append([
            NICE.get(v, v).split(" (")[0],
            f"{dt.get('ground_points', 0)/1e6:.1f}M" if dt.get("ground_points") else "—",
            f"{st.get('relief_m', float('nan')):.1f}" if st.get("relief_m") is not None else "—",
            f"{dt.get('rmse_m', float('nan')):.2f}",
            f"{dt.get('le90_m', float('nan')):.2f}",
            f"{wl.get('roc_auc', float('nan')):.3f}",
            f"{dr.get('channel_count', 0)}",
            f"{dr.get('total_length_m', 0)/1000:.1f}",
            f"{dr.get('total_cost_inr_lakhs', 0):.0f}",
        ])
    cols = ["Village", "Ground pts", "Relief (m)", "DTM RMSE (m)", "DTM LE90 (m)",
            "WL AUC", "Channels", "Drain (km)", "Cost (₹L)"]
    fig, ax = plt.subplots(figsize=(13, 0.6 + 0.5 * (len(rows) + 1)))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    w0 = 0.155
    widths = [w0] + [(1 - w0) / (len(cols) - 1)] * (len(cols) - 1)
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                   colWidths=widths, bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0EBE1")
        else:
            cell.set_facecolor("white")
        if c == 0 and r > 0:
            cell.set_text_props(fontweight="bold", color=NAVY)
    ax.set_title("Per-Village Results Scorecard — Honest Metrics",
                 fontsize=13, fontweight="bold", color=NAVY, pad=14)
    fig.tight_layout(); fig.savefig(IMG / "fig_scorecard.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    print("  [ok] fig_scorecard.png")


def main():
    rep = load()
    print("Building figures ...")
    fig_dtm_accuracy(rep)
    fig_wl_transfer(rep)
    fig_scorecard(rep)
    fig_drainage_hero(rep, "DEVDI")
    for v in rep["villages"]:
        try:
            fig_village_map(v)
        except Exception as e:
            print(f"  [skip] map {v}: {e}")
    print(f"Done → {IMG}")


if __name__ == "__main__":
    main()
