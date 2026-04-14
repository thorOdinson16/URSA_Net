# ------------------------------------------------------------
# MC DROPOUT — DIAGNOSTIC BASELINE
# NOT part of the URSA-Net pipeline.
#
# Purpose: generate uncertainty statistics for paper comparison.
# Shows that MCDropout yields negligible uncertainty on this model,
# motivating the temporal variance approach used in the pipeline.
#
# Method:
#   Dropout layers are re-enabled at inference time (train mode).
#   The same frame is passed through the model MCD_RUNS times.
#   Uncertainty per class = variance of confidence scores across runs.
#   A high-confidence, well-trained model on augmented data will show
#   near-zero variance — that's the expected (and paper-relevant) result.
#
# Output:
#   mcd_baseline_results.csv   — per-frame per-class uncertainty
#   mcd_baseline_summary.txt   — statistics for paper table
#
# Usage:
#   python mcd_baseline.py
# ------------------------------------------------------------

import os
import time
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

MODEL_PATH     = "runs/detect/augmented_model/weights/best.pt"
from video_config import cfg
INPUT_FOLDER = cfg.filtered_frames_dir()
OUTPUT_CSV   = cfg.out("mcd_baseline_results.csv")
SUMMARY_TXT  = cfg.out("mcd_baseline_summary.txt")

MCD_RUNS       = 30            # standard in literature; enough to estimate variance
CONF_THRESHOLD = 0.25
BATCH_SIZE     = 8             # smaller — MCD_RUNS forward passes per batch
RAM_LIMIT      = 90

# Subsample frames for speed — MCD is expensive and we only need
# representative statistics, not full dataset coverage.
# Set to None to run on all frames.
MAX_FRAMES     = 500


# ------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------

CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole"
]


# ------------------------------------------------------------
# RAM GUARDRAIL
# ------------------------------------------------------------

def wait_for_ram():
    while True:
        if psutil.virtual_memory().percent < RAM_LIMIT:
            break
        print(f"RAM high ({psutil.virtual_memory().percent}%). Waiting...")
        time.sleep(3)


# ------------------------------------------------------------
# ENABLE DROPOUT AT INFERENCE TIME
# ------------------------------------------------------------

def enable_dropout(model_nn: nn.Module) -> int:
    """
    Set all Dropout layers to train() mode so they remain active
    during inference. BatchNorm layers stay in eval() mode to keep
    feature statistics stable — we only want stochasticity from
    dropout, not from batch statistics.

    Returns the count of dropout layers found.
    """
    count = 0
    for m in model_nn.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            count += 1
    return count


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("\nLoading YOLO model...")
model = YOLO(MODEL_PATH)

# Full eval first — then selectively re-enable dropout only
model.model.eval()
n_dropout_layers = enable_dropout(model.model)

print(f"Model loaded.")
print(f"Dropout layers found and re-enabled: {n_dropout_layers}")

if n_dropout_layers == 0:
    print(
        "\n[WARNING] No dropout layers found in this model.\n"
        "  YOLOv8/v11 architectures use minimal or no dropout by default.\n"
        "  MCDropout will produce zero variance — this is expected and is\n"
        "  precisely the negative result motivating the temporal variance approach.\n"
        "  The script will still run and record the near-zero uncertainty values.\n"
    )


# ------------------------------------------------------------
# LOAD FRAME LIST
# ------------------------------------------------------------

if not os.path.exists(INPUT_FOLDER):
    raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

all_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

if len(all_files) == 0:
    raise RuntimeError("No frames found.")

# Subsample evenly across the dataset so we cover all road conditions
if MAX_FRAMES is not None and len(all_files) > MAX_FRAMES:
    indices    = np.linspace(0, len(all_files) - 1, MAX_FRAMES, dtype=int)
    image_files = [all_files[i] for i in indices]
    print(f"\nSubsampled {MAX_FRAMES} frames from {len(all_files)} total (evenly spaced)")
else:
    image_files = all_files
    print(f"\nUsing all {len(image_files)} frames")

print(f"MCD runs per frame: {MCD_RUNS}")
print(f"Batch size:         {BATCH_SIZE}\n")


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

records = []


# ------------------------------------------------------------
# PROCESS BATCHES
# ------------------------------------------------------------

for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="MCD batches"):

    wait_for_ram()

    batch_files = image_files[i : i + BATCH_SIZE]
    batch_paths = [os.path.join(INPUT_FOLDER, f) for f in batch_files]

    # ── Collect MCD_RUNS predictions per batch ───────────────
    # Each run uses the same weights but different dropout masks,
    # producing stochastic confidence scores if dropout is present.
    mcd_predictions = []

    for run in range(MCD_RUNS):

        # Re-enable dropout before every run (eval() would turn it off)
        enable_dropout(model.model)

        with torch.no_grad():
            results = model.predict(
                source=batch_paths,
                conf=CONF_THRESHOLD,
                imgsz=640,
                device=0,
                batch=BATCH_SIZE,
                half=False,    # keep fp32 — dropout + fp16 can be unstable
                augment=False,
                verbose=False
            )

        mcd_predictions.append(results)

    # ── Compute per-frame per-class uncertainty ───────────────
    for img_idx, frame_name in enumerate(batch_files):

        # Gather all detections across MCD runs for this frame,
        # grouped by class.
        class_confs: dict = {}   # cls_id -> list of conf values across runs

        for run in range(MCD_RUNS):

            r = mcd_predictions[run][img_idx]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])

                if cls_id not in class_confs:
                    class_confs[cls_id] = []

                class_confs[cls_id].append(conf)

        if not class_confs:
            # No detections in any run — record a zero-uncertainty row
            records.append({
                "frame":            frame_name,
                "class":            "none",
                "mcd_runs_with_det": 0,
                "conf_mean":        0.0,
                "conf_std":         0.0,
                "conf_variance":    0.0,   # this is the MCD uncertainty metric
                "conf_range":       0.0,
            })
            continue

        for cls_id, conf_list in class_confs.items():

            conf_arr = np.array(conf_list)

            class_name = (
                CLASS_NAMES[cls_id]
                if cls_id < len(CLASS_NAMES)
                else str(cls_id)
            )

            records.append({
                "frame":             frame_name,
                "class":             class_name,
                "mcd_runs_with_det": len(conf_list),
                "conf_mean":         round(float(conf_arr.mean()), 6),
                "conf_std":          round(float(conf_arr.std()),  6),
                "conf_variance":     round(float(conf_arr.var()),  6),  # primary MCD metric
                "conf_range":        round(float(conf_arr.max() - conf_arr.min()), 6),
            })


# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved -> {OUTPUT_CSV}")
print(f"Total rows: {len(df)}")


# ------------------------------------------------------------
# SUMMARY STATISTICS (for paper table)
# ------------------------------------------------------------

detected = df[df["class"] != "none"]

lines = []
lines.append("=" * 55)
lines.append("MC DROPOUT BASELINE — SUMMARY")
lines.append("(for paper comparison table)")
lines.append("=" * 55)
lines.append(f"  Frames evaluated       : {df['frame'].nunique()}")
lines.append(f"  MCD forward passes     : {MCD_RUNS}")
lines.append(f"  Dropout layers found   : {n_dropout_layers}")
lines.append("")
lines.append("  -- Confidence variance (MCD uncertainty) --")

if len(detected) > 0:
    lines.append(f"  Mean variance          : {detected['conf_variance'].mean():.6f}")
    lines.append(f"  Median variance        : {detected['conf_variance'].median():.6f}")
    lines.append(f"  Max variance           : {detected['conf_variance'].max():.6f}")
    lines.append(f"  Mean conf_std          : {detected['conf_std'].mean():.6f}")
    lines.append(f"  Mean conf_range        : {detected['conf_range'].mean():.6f}")
    lines.append("")
    lines.append("  -- Per-class breakdown --")

    for cls_name, grp in detected.groupby("class"):
        lines.append(
            f"  {cls_name:<22} "
            f"var={grp['conf_variance'].mean():.6f}  "
            f"std={grp['conf_std'].mean():.6f}  "
            f"n={len(grp)}"
        )
else:
    lines.append("  No detections recorded.")

lines.append("")
lines.append("  -- Interpretation --")

if n_dropout_layers == 0:
    lines.append("  No dropout layers present in model architecture.")
    lines.append("  Variance is structurally zero — MCDropout inapplicable.")
elif len(detected) > 0 and detected["conf_variance"].mean() < 0.001:
    lines.append("  Variance is negligible (< 0.001).")
    lines.append("  Model is dropout-robust — MCDropout provides no")
    lines.append("  meaningful uncertainty signal for this architecture.")
else:
    lines.append("  Non-negligible variance detected — inspect results.")

lines.append("")
lines.append("=" * 55)

summary = "\n".join(lines)
print("\n" + summary)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\nSummary saved -> {SUMMARY_TXT}")
print("\nMCDropout baseline complete.")