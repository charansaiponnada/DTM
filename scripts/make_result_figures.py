"""
scripts/make_result_figures.py
──────────────────────────────
Render clean, current result maps from the actual GeoTIFF / GeoPackage outputs
for use in the hackathon deck. Replaces the stale/broken figures in
data/output/figures/ (metrics_dashboard.png had old numbers, drainage_map.png
had edge-streak artifacts, cross_village_results.png was blank).

Outputs (300 dpi, transparent nodata, clipped to data extent):
  figures/result_<VILLAGE>.png       hillshade + waterlogging risk + drainage
  figures/risk_grid.png              2x2 waterlogging risk across all 4 villages
  figures/drainage_<VILLAGE>.png     drainage network coloured by channel type

Usage:  .venv/Scripts/python.exe scripts/make_result_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import rasterio
import geopandas as gpd

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "data" / "output"
FIG     = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "_reports" / "honest_metrics.json"

VILLAGES = {
    "DEVDI":           "Devdi · Gujarat",
    "KHAPRETA":        "Khapreta · Gujarat",
    "DHAL_HOSHIARPUR": "Dhal Hoshiarpur · Punjab",
    "CHAKHIRASINGH":   "Chakhirasingh · Punjab",
}

EARTHEN = "#C9A227"   # warm gold
CONCRETE = "#19C3E6"  # cyan
NAVY    = "#16263C"


def _read(path: Path):
    """Return (masked_array, extent) with nodata -> NaN."""
    with rasterio.open(path) as s:
        a  = s.read(1).astype(np.float32)
        nd = s.nodata
        b  = s.bounds
    if nd is not None:
        a = np.where(a == nd, np.nan, a)
    a = np.where(np.isfinite(a), a, np.nan)
    return a, (b.left, b.right, b.bottom, b.top)


def _hillshade_rgba(hs):
    """Grayscale hillshade as RGBA, transparent where NaN."""
    finite = hs[np.isfinite(hs)]
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
    norm = np.clip((hs - lo) / (hi - lo + 1e-9), 0, 1)
    rgba = plt.cm.gray(norm)
    rgba[..., 3] = np.where(np.isfinite(hs), 1.0, 0.0)
    return rgba


def _risk_rgba(prob, thresh=0.40):
    """Waterlogging probability >= thresh as RdYlGn_r, alpha ramps with prob."""
    cmap = matplotlib.colormaps["RdYlGn_r"]
    norm = np.clip(prob, 0, 1)
    rgba = cmap(norm)
    a = np.where((prob >= thresh) & np.isfinite(prob),
                 0.30 + 0.55 * np.clip((prob - thresh) / (1 - thresh), 0, 1), 0.0)
    rgba[..., 3] = a
    return rgba


def _load_channels(village, min_valid_frac=0.85):
    """Load channels, dropping flow-routing artifacts that cross nodata terrain
    (straight diagonal streaks routed through the rectangular DEM padding).
    Samples points evenly along each line and keeps only channels whose path
    lies almost entirely on valid DTM."""
    base = OUT / village
    g = gpd.read_file(base / "drainage_network.gpkg", layer="drainage_channels")
    with rasterio.open(base / "dtm.tif") as src:
        nd = src.nodata
        keep = np.zeros(len(g), dtype=bool)
        for i, geom in enumerate(g.geometry.values):
            n = 7
            pts = [geom.interpolate(t, normalized=True).coords[0]
                   for t in np.linspace(0, 1, n)]
            vals = np.array([v[0] for v in src.sample(pts)], dtype=float)
            valid = np.isfinite(vals) & (vals != nd)
            keep[i] = valid.mean() >= min_valid_frac
    return g[keep].copy()  # native UTM CRS — matches rasters


def result_map(village: str, title: str):
    base = OUT / village
    hs, ext = _read(base / "hillshade.tif")
    prob, _ = _read(base / "waterlogging_probability.tif")
    ch = _load_channels(village)

    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=300)
    ax.imshow(_hillshade_rgba(hs), extent=ext, origin="upper", interpolation="bilinear")
    ax.imshow(_risk_rgba(prob),    extent=ext, origin="upper", interpolation="bilinear")

    for ctype, col, lw in [("earthen", EARTHEN, 0.7), ("concrete", CONCRETE, 1.1)]:
        sub = ch[ch["channel_type"] == ctype]
        if len(sub):
            sub.plot(ax=ax, color=col, linewidth=lw, alpha=0.95)

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(NAVY); sp.set_linewidth(1.2)

    ax.set_title(title, fontsize=15, fontweight="bold", color=NAVY, pad=8)

    legend = [
        Patch(facecolor="#D32F2F", alpha=0.8, label="High waterlogging risk"),
        Patch(facecolor="#FFC107", alpha=0.8, label="Moderate risk"),
        Line2D([0], [0], color=EARTHEN, lw=2.2, label="Earthen channel"),
        Line2D([0], [0], color=CONCRETE, lw=2.2, label="Concrete channel"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8.5, framealpha=0.92,
              facecolor="white", edgecolor="#CCCCCC", ncol=2,
              columnspacing=1.0, handlelength=1.6)

    # scale bar (500 m)
    x0 = ext[0] + (ext[1]-ext[0])*0.62
    y0 = ext[2] + (ext[3]-ext[2])*0.045
    ax.plot([x0, x0+500], [y0, y0], color=NAVY, lw=3, solid_capstyle="butt")
    ax.text(x0+250, y0+(ext[3]-ext[2])*0.012, "500 m", ha="center", va="bottom",
            fontsize=8, color=NAVY, fontweight="bold")

    fig.tight_layout()
    out = FIG / f"result_{village}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out.name}")


def drainage_map(village: str, title: str):
    base = OUT / village
    hs, ext = _read(base / "hillshade.tif")
    ch = _load_channels(village)

    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=300)
    # dark terrain backdrop
    finite = hs[np.isfinite(hs)]
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
    norm = np.clip((hs - lo) / (hi - lo + 1e-9), 0, 1)
    rgba = plt.cm.bone(0.25 + 0.6*norm)
    rgba[..., 3] = np.where(np.isfinite(hs), 1.0, 0.0)
    ax.imshow(rgba, extent=ext, origin="upper", interpolation="bilinear")

    for ctype, col, lw in [("earthen", EARTHEN, 0.8), ("concrete", CONCRETE, 1.3)]:
        sub = ch[ch["channel_type"] == ctype]
        if len(sub):
            sub.plot(ax=ax, color=col, linewidth=lw, alpha=0.95)

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(NAVY); sp.set_linewidth(1.2)
    ax.set_title(title, fontsize=15, fontweight="bold", color=NAVY, pad=8)

    legend = [
        Line2D([0],[0], color=EARTHEN, lw=2.4, label="Earthen channel"),
        Line2D([0],[0], color=CONCRETE, lw=2.4, label="Concrete channel"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=9, framealpha=0.92,
              facecolor="white", edgecolor="#CCCCCC")
    fig.tight_layout()
    out = FIG / f"drainage_{village}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out.name}")


def risk_grid():
    """2x2 waterlogging risk across all villages, shared style."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2), dpi=200)
    with open(METRICS) as f:
        M = json.load(f)
    for ax, (v, title) in zip(axes.ravel(), VILLAGES.items()):
        base = OUT / v
        hs, ext   = _read(base / "hillshade.tif")
        prob, _   = _read(base / "waterlogging_probability.tif")
        ax.imshow(_hillshade_rgba(hs), extent=ext, origin="upper", interpolation="bilinear")
        ax.imshow(_risk_rgba(prob),    extent=ext, origin="upper", interpolation="bilinear")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(NAVY); sp.set_linewidth(1.0)
        auc = M["villages"][v]["waterlogging"]["roc_auc"]
        ax.set_title(f"{title}   ·   AUC {auc:.3f}", fontsize=12,
                     fontweight="bold", color=NAVY, pad=5)
    legend = [
        Patch(facecolor="#D32F2F", alpha=0.85, label="High risk"),
        Patch(facecolor="#FFC107", alpha=0.85, label="Moderate risk"),
        Patch(facecolor="#1A9850", alpha=0.85, label="Low risk"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Waterlogging Risk — Same Model, 4 Villages, 2 States (zero retraining)",
                 fontsize=15, fontweight="bold", color=NAVY, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = FIG / "risk_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out.name}")


if __name__ == "__main__":
    print("Rendering result figures …")
    for v, t in VILLAGES.items():
        result_map(v, t.split(" · ")[0] + " — Risk + Drainage")
        drainage_map(v, t.split(" · ")[0] + " — Drainage Network")
    risk_grid()
    print("Done.")
