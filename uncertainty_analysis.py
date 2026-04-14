import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time
import os

from video_config import cfg

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

CSV_FILE    = cfg.out("uncertainty_predictions.csv")
RAM_LIMIT   = 90
PLOT_DIR    = cfg.plots_dir()

HIST_OUTPUT  = os.path.join(PLOT_DIR, "uncertainty_histogram.png")
CONF_PLOT    = os.path.join(PLOT_DIR, "uncertainty_vs_confidence.png")
FRAME_CURVE  = os.path.join(PLOT_DIR, "uncertainty_over_frames.png")
CLASS_PLOT   = os.path.join(PLOT_DIR, "uncertainty_per_class.png")


# ------------------------------------------------
# RAM GUARDRAIL
# ------------------------------------------------

def wait_for_ram():
    while True:
        if psutil.virtual_memory().percent < RAM_LIMIT:
            break
        print(f"RAM usage high. Waiting...")
        time.sleep(3)


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

wait_for_ram()

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"uncertainty_predictions.csv not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

print(f"\nUncertainty Dataset Statistics — {cfg.video}")
print("--------------------------------")
print("Total detections:", len(df))
print("Mean uncertainty:",   round(df["uncertainty"].mean(),   4))
print("Median uncertainty:", round(df["uncertainty"].median(), 4))
print("Std uncertainty:",    round(df["uncertainty"].std(),    4))
print("Min uncertainty:",    round(df["uncertainty"].min(),    4))
print("Max uncertainty:",    round(df["uncertainty"].max(),    4))


# ------------------------------------------------
# HISTOGRAM
# ------------------------------------------------

wait_for_ram()

plt.figure(figsize=(8, 5))
plt.hist(df["uncertainty"], bins=40, edgecolor="black")
plt.title(f"Prediction Uncertainty Distribution — {cfg.video}")
plt.xlabel("Uncertainty")
plt.ylabel("Detection Count")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(HIST_OUTPUT)
plt.close()
print("Saved:", HIST_OUTPUT)


# ------------------------------------------------
# UNCERTAINTY VS CONFIDENCE
# ------------------------------------------------

wait_for_ram()

plt.figure(figsize=(8, 5))
plt.scatter(df["confidence_mean"], df["uncertainty"], alpha=0.4)
plt.title(f"Uncertainty vs Detection Confidence — {cfg.video}")
plt.xlabel("Confidence")
plt.ylabel("Uncertainty")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CONF_PLOT)
plt.close()
print("Saved:", CONF_PLOT)


# ------------------------------------------------
# UNCERTAINTY OVER FRAMES
# ------------------------------------------------

wait_for_ram()

df["frame_index"] = df["frame"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
df_sorted = df.sort_values("frame_index")
smooth    = df_sorted["uncertainty"].rolling(50).mean()

plt.figure(figsize=(10, 5))
plt.plot(df_sorted["frame_index"], df_sorted["uncertainty"], alpha=0.3)
plt.plot(df_sorted["frame_index"], smooth, linewidth=2)
plt.title(f"Uncertainty Over Frames — {cfg.video}")
plt.xlabel("Frame Index")
plt.ylabel("Uncertainty")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FRAME_CURVE)
plt.close()
print("Saved:", FRAME_CURVE)


# ------------------------------------------------
# UNCERTAINTY PER CLASS
# ------------------------------------------------

wait_for_ram()

class_means = df.groupby("class")["uncertainty"].mean()
plt.figure(figsize=(7, 5))
class_means.plot(kind="bar")
plt.title(f"Average Uncertainty per Damage Type — {cfg.video}")
plt.ylabel("Mean Uncertainty")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(CLASS_PLOT)
plt.close()
print("Saved:", CLASS_PLOT)

print("\nAnalysis complete.")