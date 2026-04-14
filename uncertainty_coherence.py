# ------------------------------------------------------------
# uncertainty_coherence.py — URSA-Net
# Scatter: mean_uncertainty vs retention_pct per video
# Validates uncertainty module without ground truth:
#   "uncertainty tracks frame quality"
# Outputs: scatter + regression line (Figure 5) + Pearson r
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

INPUT_CSV  = "dataset_summary.csv"
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITION_COLORS = {
    "day":     "#2196F3",
    "evening": "#FF9800",
    "night":   "#9C27B0",
}

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["mean_uncertainty", "retention_pct"])

x = df["retention_pct"].values
y = df["mean_uncertainty"].values

# ------------------------------------------------------------
# PEARSON
# ------------------------------------------------------------

r, p = pearsonr(x, y)

print(f"Pearson r : {r:.4f}")
print(f"p-value   : {p:.4f}")
print(f"n videos  : {len(df)}")

if abs(r) >= 0.5:
    interp = "strong"
elif abs(r) >= 0.3:
    interp = "moderate"
else:
    interp = "weak"

direction = "negative" if r < 0 else "positive"
print(f"Interpretation: {interp} {direction} correlation")

# NOTE: All retention values are 100% in this dataset (reliability
# filter threshold set to 0.0 effectively, keeping all frames).
# The scatter therefore shows uncertainty variance across roads
# at uniform retention — still useful as a per-road uncertainty
# distribution summary, but the coherence claim needs a caveat.

if np.std(x) < 0.01:
    print("\nNOTE: All retention values are identical (100%).")
    print("Switching plot to: uncertainty vs mean_severity (more informative).")
    USE_SEVERITY = True
else:
    USE_SEVERITY = False

# ------------------------------------------------------------
# SCATTER PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

for condition, color in CONDITION_COLORS.items():
    sub = df[df["time_of_day"] == condition]
    if sub.empty:
        continue

    xv = sub["mean_severity"].values if USE_SEVERITY else sub["retention_pct"].values
    yv = sub["mean_uncertainty"].values

    ax.scatter(xv, yv, color=color, s=60, alpha=0.8,
               label=condition.capitalize(), edgecolors="white", linewidths=0.5)

    for _, row in sub.iterrows():
        xi = row["mean_severity"] if USE_SEVERITY else row["retention_pct"]
        ax.annotate(
            row["road_name"],
            (xi, row["mean_uncertainty"]),
            fontsize=5.5, alpha=0.55,
            xytext=(3, 3), textcoords="offset points"
        )

# Regression line
xv_all = df["mean_severity"].values if USE_SEVERITY else df["retention_pct"].values
yv_all = df["mean_uncertainty"].values

if USE_SEVERITY:
    xlabel = "Mean Severity Score (per video)"
    title  = "Uncertainty vs Severity Score per Road Segment"
    r2, p2 = pearsonr(xv_all, yv_all)
    r, p   = r2, p2
    print(f"\nPearson r (uncertainty vs severity): {r:.4f}, p={p:.4f}")
else:
    xlabel = "Frame Retention Rate (%)"
    title  = "Uncertainty Coherence: Uncertainty vs Retention Rate"

m, b    = np.polyfit(xv_all, yv_all, 1)
x_line  = np.linspace(xv_all.min(), xv_all.max(), 200)
ax.plot(x_line, m * x_line + b, color="crimson", linewidth=1.8,
        linestyle="--", label=f"Regression (r={r:.3f}, p={p:.3f})")

ax.set_xlabel(xlabel, fontsize=11)
ax.set_ylabel("Mean TTA Uncertainty (per video)", fontsize=11)
ax.set_title(title, fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(linestyle="--", alpha=0.35)

# Annotation box
textstr = f"Pearson r = {r:.3f}\np = {p:.3f}\nn = {len(df)}"
props = dict(boxstyle="round", facecolor="lightyellow", alpha=0.7)
ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", bbox=props)

plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "figure5_uncertainty_coherence.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"\nFigure saved → {fig_path}")
plt.close()

# Save stats
stats = pd.DataFrame([{
    "pearson_r": round(float(r), 6),
    "p_value":   round(float(p), 6),
    "n_videos":  len(df),
    "x_axis":    "mean_severity" if USE_SEVERITY else "retention_pct",
    "interpretation": f"{interp} {direction}"
}])
stats.to_csv(os.path.join(OUTPUT_DIR, "uncertainty_coherence_stats.csv"), index=False)

print("Uncertainty coherence analysis complete.")