# ------------------------------------------------------------
# lighting_analysis.py — URSA-Net
# Compares day vs evening/night across 3 metrics:
#   1. Frame retention rate
#   2. Mean TTA uncertainty
#   3. Mean severity score
# Outputs: boxplots (Figure 2) + Mann-Whitney p-values
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

INPUT_CSV   = "dataset_summary.csv"
OUTPUT_DIR  = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = [
    ("retention_pct",    "Frame Retention Rate (%)",   "Retention (%)"),
    ("mean_uncertainty", "Mean TTA Uncertainty",        "Uncertainty"),
    ("mean_severity",    "Mean Severity Score",         "Severity Score"),
]

# Merge evening + night → "Low-Light"
LABEL_MAP = {
    "day":     "Daytime",
    "evening": "Low-Light",
    "night":   "Low-Light",
}

COLORS = {
    "Daytime":   "#2196F3",
    "Low-Light": "#FF9800",
}

# ------------------------------------------------------------
# LOAD + PREPARE
# ------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

df["condition"] = df["time_of_day"].map(LABEL_MAP)

day_df  = df[df["condition"] == "Daytime"]
night_df = df[df["condition"] == "Low-Light"]

print(f"Daytime videos  : {len(day_df)}")
print(f"Low-Light videos: {len(night_df)}")

# ------------------------------------------------------------
# STATS TABLE
# ------------------------------------------------------------

print("\n{:<25} {:>10} {:>10} {:>12} {:>12} {:>10}".format(
    "Metric", "Day Mean", "LL Mean", "Day Median", "LL Median", "p-value"
))
print("-" * 82)

stats_records = []

for col, title, ylabel in METRICS:
    day_vals  = day_df[col].dropna().values
    night_vals = night_df[col].dropna().values

    stat, p = mannwhitneyu(day_vals, night_vals, alternative="two-sided")

    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")

    print(f"{title:<25} {day_vals.mean():>10.4f} {night_vals.mean():>10.4f} "
          f"{np.median(day_vals):>12.4f} {np.median(night_vals):>12.4f} "
          f"{p:>9.4f} {sig}")

    stats_records.append({
        "metric":        title,
        "day_mean":      round(float(day_vals.mean()),  6),
        "lowlight_mean": round(float(night_vals.mean()), 6),
        "day_median":    round(float(np.median(day_vals)),   6),
        "lowlight_median": round(float(np.median(night_vals)), 6),
        "mann_whitney_u": round(float(stat), 4),
        "p_value":       round(float(p), 6),
        "significant":   sig,
    })

stats_df = pd.DataFrame(stats_records)
stats_path = os.path.join(OUTPUT_DIR, "lighting_stats.csv")
stats_df.to_csv(stats_path, index=False)
print(f"\nStats saved → {stats_path}")

# ------------------------------------------------------------
# BOXPLOTS  (3 subplots side by side)
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle("Lighting Condition Analysis: Daytime vs Low-Light",
             fontsize=14, fontweight="bold", y=1.02)

for ax, (col, title, ylabel), rec in zip(axes, METRICS, stats_records):
    day_vals   = day_df[col].dropna().values
    night_vals = night_df[col].dropna().values

    data   = [day_vals, night_vals]
    labels = ["Daytime", "Low-Light"]
    colors = [COLORS["Daytime"], COLORS["Low-Light"]]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Scatter jitter
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(i + jitter, vals, color=color, alpha=0.55,
                   s=25, zorder=5, edgecolors="white", linewidths=0.4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # p-value annotation
    p     = rec["p_value"]
    sig   = rec["significant"]
    y_max = max(np.max(day_vals), np.max(night_vals))
    y_ann = y_max * 1.10
    ax.annotate(
        f"p = {p:.4f} ({sig})",
        xy=(1.5, y_ann),
        ha="center", fontsize=9,
        color="dimgray"
    )
    ax.set_ylim(top=y_ann * 1.15)

    # n labels
    ax.text(1, ax.get_ylim()[0], f"n={len(day_vals)}",
            ha="center", va="bottom", fontsize=8, color="gray")
    ax.text(2, ax.get_ylim()[0], f"n={len(night_vals)}",
            ha="center", va="bottom", fontsize=8, color="gray")

plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "figure2_lighting_analysis.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"Figure saved  → {fig_path}")
plt.close()

print("\nLighting analysis complete.")