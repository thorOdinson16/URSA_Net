import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import psutil
import shutil
import time
import os

from video_config import cfg

# ==========================================
# CONFIG
# ==========================================

CSV_FILE        = cfg.out("frame_reliability.csv")
PLOT_DIR        = cfg.plots_dir()
HIST_OUTPUT     = os.path.join(PLOT_DIR, "reliability_histogram.png")
CURVE_OUTPUT    = os.path.join(PLOT_DIR, "reliability_curve.png")

MANUAL_THRESHOLD = 0.6
SMOOTH_WINDOW    = 50
RAM_LIMIT        = 90
MIN_DISK_GB      = 1


# ==========================================
# GUARDRAILS
# ==========================================

def wait_for_ram():
    while True:
        if psutil.virtual_memory().percent < RAM_LIMIT:
            break
        print(f"RAM usage high. Waiting...")
        time.sleep(3)


def check_storage():
    disk = shutil.disk_usage(".")
    free_gb = disk.free / (1024**3)
    if free_gb < MIN_DISK_GB:
        raise RuntimeError(f"Low disk space: {free_gb:.2f} GB available.")


# ==========================================
# LOAD DATA
# ==========================================

wait_for_ram()
check_storage()

df     = pd.read_csv(CSV_FILE)
scores = df["reliability"]

print(f"\nReliability statistics — {cfg.video}")
print("----------------------")
print("Frames:", len(scores))
print("Mean:", round(scores.mean(), 3))
print("Median:", round(scores.median(), 3))
print("Std:", round(scores.std(), 3))
print("Min:", round(scores.min(), 3))
print("Max:", round(scores.max(), 3))

suggested_threshold = scores.quantile(0.6)
print("\nSuggested threshold (60th percentile):", round(suggested_threshold, 3))
print("Manual threshold used:", MANUAL_THRESHOLD)


# ==========================================
# HISTOGRAM
# ==========================================

wait_for_ram()
check_storage()

plt.figure(figsize=(8, 5))
plt.hist(scores, bins=40, edgecolor="black")
plt.axvline(MANUAL_THRESHOLD, color="red", linestyle="--", linewidth=2,
            label=f"Threshold = {MANUAL_THRESHOLD}")
plt.title(f"Frame Reliability Distribution — {cfg.video}")
plt.xlabel("Reliability Score")
plt.ylabel("Frame Count")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(HIST_OUTPUT)
plt.close()
print("Saved:", HIST_OUTPUT)


# ==========================================
# RELIABILITY VS FRAME INDEX
# ==========================================

wait_for_ram()
check_storage()

frames = np.arange(len(scores))
smooth = scores.rolling(SMOOTH_WINDOW).mean()

plt.figure(figsize=(10, 5))
plt.plot(frames, scores, alpha=0.3, label="Raw reliability")
plt.plot(frames, smooth, linewidth=2, label=f"Moving Avg ({SMOOTH_WINDOW})")
plt.axhline(MANUAL_THRESHOLD, color="red", linestyle="--", label="Threshold")
plt.title(f"Frame Reliability Over Time — {cfg.video}")
plt.xlabel("Frame Index")
plt.ylabel("Reliability Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CURVE_OUTPUT)
plt.close()
print("Saved:", CURVE_OUTPUT)

print("\nAnalysis complete.")