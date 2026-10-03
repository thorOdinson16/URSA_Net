"""
URSA-Net — Severity Estimation Visualisation (per-video)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from video_config import cfg

# ===============================
# PARAMETERS
# ===============================

INPUT_CSV  = cfg.out("severity_scores.csv")
OUTPUT_DIR = cfg.plots_dir()

FONT_SIZE    = 11
TITLE_SIZE   = 13
FIGSIZE_WIDE = (12, 4)
FIGSIZE_SQ   = (6, 5)
DPI          = 180

COLOURS = {
    "longitudinal_crack": "#378ADD",
    "alligator_crack":    "#D85A30",
    "transverse_crack":   "#1D9E75",
    "pothole":            "#BA7517",
}

RISK_COLOURS = {
    "High":   "#E24B4A",
    "Medium": "#EF9F27",
    "Low":    "#639922",
    "None":   "#B4B2A9",
}

SEVERITY_LINE    = "#E24B4A"
RELIABILITY_LINE = "#85B7EB"
HIST_COLOUR      = "#378ADD"
SCATTER_COLOUR   = "#7F77DD"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          FONT_SIZE,
    "axes.titlesize":     TITLE_SIZE,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         DPI,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
})


# ===============================
# LOAD DATA
# ===============================

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run severity_module.py first.")

    df = pd.read_csv(path)

    if "frame_num" not in df.columns:
        extracted = df["frame"].astype(str).str.extract(r"(\d+)")
        df["frame_num"] = extracted.fillna(0).astype(int)

    return df.sort_values("frame_num").reset_index(drop=True)


def parse_class_counts(df):
    totals = {}
    for val in df["class_counts"].dropna():
        try:
            cc = json.loads(val)
            for k, v in cc.items():
                totals[k] = totals.get(k, 0) + v
        except Exception:
            continue
    return totals


# ===============================
# FIGURES
# ===============================

def plot_timeline(df, out):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.fill_between(df["frame_num"], df["severity_score"], alpha=0.15, color=SEVERITY_LINE)
    ax.plot(df["frame_num"], df["severity_score"], color=SEVERITY_LINE,
            linewidth=1.2, label="Severity score")
    if "reliability" in df.columns:
        ax.plot(df["frame_num"], df["reliability"], color=RELIABILITY_LINE,
                linewidth=1, linestyle="--", alpha=0.7, label="Frame reliability")
    ax.axhline(0.5, color=RISK_COLOURS["High"],   linestyle=":", linewidth=1)
    ax.axhline(0.3, color=RISK_COLOURS["Medium"], linestyle=":", linewidth=1)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Severity score and frame reliability — {cfg.video}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_risk_distribution(df, out):
    total  = max(len(df), 1)
    high   = int((df["severity_score"] > 0.5).sum())
    medium = int(((df["severity_score"] >= 0.3) & (df["severity_score"] <= 0.5)).sum())
    low    = int(((df["severity_score"] > 0) & (df["severity_score"] < 0.3)).sum())
    none   = int((df["severity_score"] == 0).sum())

    labels  = ["None", "Low", "Medium", "High"]
    counts  = [none, low, medium, high]
    colours = [RISK_COLOURS[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts, color=colours)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{count}\n({count/total*100:.1f}%)",
                ha="center")
    ax.set_ylabel("Number of frames")
    ax.set_title(f"Frame risk tier distribution — {cfg.video}")
    fig.tight_layout()
    fig.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_histogram(df, out):
    detected = df[df["num_detections"] > 0]["severity_score"]
    if len(detected) == 0:
        print("Skipped histogram: no detections")
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    n, bins, patches = ax.hist(detected, bins=22, color=HIST_COLOUR, alpha=0.8)
    for patch, left in zip(patches, bins[:-1]):
        if left >= 0.5:
            patch.set_facecolor(RISK_COLOURS["High"])
        elif left >= 0.3:
            patch.set_facecolor(RISK_COLOURS["Medium"])
    ax.axvline(detected.mean(),   color="#333", linestyle="--")
    ax.axvline(detected.median(), color="#888", linestyle=":")
    ax.set_xlabel("Severity score")
    ax.set_ylabel("Frames")
    ax.set_title(f"Severity distribution — {cfg.video}")
    fig.tight_layout()
    fig.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_class_breakdown(df, out):
    totals = parse_class_counts(df)
    if not totals:
        print("Skipped class breakdown")
        return
    labels  = list(totals.keys())
    counts  = list(totals.values())
    colours = [COLOURS.get(k, "#888") for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, counts, color=colours)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, str(count), ha="center")
    ax.set_ylabel("Detections")
    ax.set_title(f"Damage class distribution — {cfg.video}")
    fig.tight_layout()
    fig.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_severity_vs_uncertainty(df, out):
    det = df[df["num_detections"] > 0]
    if len(det) == 0:
        print("Skipped uncertainty plot")
        return
    colours = det["severity_score"].apply(
        lambda s: RISK_COLOURS["High"] if s > 0.5 else
                  RISK_COLOURS["Medium"] if s >= 0.3 else SCATTER_COLOUR
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    ax.scatter(det["mean_uncertainty"], det["severity_score"],
               c=colours, alpha=0.55, s=20)
    ax.set_xlabel("Prediction uncertainty")
    ax.set_ylabel("Severity score")
    ax.set_title(f"Severity vs uncertainty — {cfg.video}")
    if len(det) > 5:
        corr = det["mean_uncertainty"].corr(det["severity_score"])
        ax.text(0.97, 0.04, f"r = {corr:.3f}", transform=ax.transAxes, ha="right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_score_components(df, out):
    det = df[df["num_detections"] > 0]
    if len(det) == 0:
        print("Skipped component plot")
        return
    x = det["frame_num"].values
    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else s * 0
    bbox_n    = norm(det["total_bbox_area_ratio"])
    density_n = norm(det["damage_density"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ax, val, label, col in zip(
        axes,
        [bbox_n, density_n, det["severity_score"]],
        ["BBox area", "Damage density", "Severity score"],
        ["#378ADD", "#1D9E75", "#E24B4A"]
    ):
        ax.fill_between(x, val, alpha=0.15, color=col)
        ax.plot(x, val, color=col)
        ax.set_ylabel(label)
        ax.set_ylim(-0.05, 1.1)
    axes[-1].set_xlabel("Frame number")
    fig.tight_layout()
    fig.savefig(out)
    plt.close()
    print("Saved:", out)


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print("=" * 50)
    print(f"URSA-Net — Severity Plots  ({cfg.video})")
    print("=" * 50)

    df = load_data(INPUT_CSV)
    print("Frames:", len(df))
    print("Detected frames:", (df["num_detections"] > 0).sum())

    plot_timeline(              df, os.path.join(OUTPUT_DIR, "01_severity_timeline.png"))
    plot_risk_distribution(     df, os.path.join(OUTPUT_DIR, "02_risk_distribution.png"))
    plot_histogram(             df, os.path.join(OUTPUT_DIR, "03_score_histogram.png"))
    plot_class_breakdown(       df, os.path.join(OUTPUT_DIR, "04_class_breakdown.png"))
    plot_severity_vs_uncertainty(df, os.path.join(OUTPUT_DIR, "05_severity_vs_uncertainty.png"))
    plot_score_components(      df, os.path.join(OUTPUT_DIR, "06_score_components.png"))

    print("\nAll figures saved to:", OUTPUT_DIR)