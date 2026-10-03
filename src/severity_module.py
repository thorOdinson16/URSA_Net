import numpy as np
import cv2
import os
import pandas as pd
import psutil
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

from video_config import cfg

# ===============================
# PARAMETERS
# ===============================

DETECTIONS_CSV  = cfg.out("uncertainty_predictions.csv")
RELIABILITY_CSV = cfg.out("frame_reliability.csv")
OUTPUT_CSV      = cfg.out("severity_scores.csv")
FRAMES_DIR      = cfg.filtered_frames_dir()

CLASS_WEIGHTS = {
    "pothole":            1.0,
    "alligator_crack":    0.9,
    "transverse_crack":   0.7,
    "longitudinal_crack": 0.6,
}

DEFAULT_CLASS_WEIGHT = 0.5

W_BBOX    = 0.45
W_CRACK   = 0.35
W_DENSITY = 0.20


# ===============================
# HARDWARE OPTIMISATION
# ===============================

def get_available_workers():
    total = os.cpu_count() or 1
    workers = max(1, total - 2)
    print(f"\nCPU cores detected: {total}")
    print(f"Using workers: {workers}")
    return workers


def check_ram(limit=0.90):
    vm = psutil.virtual_memory()
    if vm.percent / 100 > limit:
        print(f"[WARNING] RAM usage high: {vm.percent:.1f}% "
              f"(free {vm.available/1e9:.2f} GB)")


def check_disk(path, min_free_gb=1.0):
    usage = psutil.disk_usage(os.path.dirname(os.path.abspath(path)) or ".")
    free_gb = usage.free / 1e9
    if free_gb < min_free_gb:
        raise SystemExit(f"[ERROR] Disk space too low ({free_gb:.2f} GB free)")


# ===============================
# IMAGE SIZE INFERENCE
# ===============================

def infer_image_size(det_df):
    max_x = det_df["bbox_x2"].max()
    max_y = det_df["bbox_y2"].max()
    w = int(np.ceil(max_x / 32) * 32)
    h = int(np.ceil(max_y / 32) * 32)
    return w, h


# ===============================
# BBOX AREA
# ===============================

def bbox_area_scores(det_df, img_area):
    if det_df.empty or "class" not in det_df.columns:
        return 0.0, 0.0, 0.0
    widths  = det_df["bbox_x2"] - det_df["bbox_x1"]
    heights = det_df["bbox_y2"] - det_df["bbox_y1"]
    areas   = np.maximum(widths * heights, 0)
    weights = det_df["class"].map(CLASS_WEIGHTS).fillna(DEFAULT_CLASS_WEIGHT)
    ratios  = (areas / img_area) * weights
    return float(ratios.sum()), float(ratios.mean()), float(ratios.max())


# ===============================
# CRACK LENGTH
# ===============================

def crack_length_scores(frame, det_df):
    if det_df.empty or "class" not in det_df.columns:
        return 0.0, 0.0
    lengths = []
    for _, row in det_df.iterrows():
        x1 = max(0, int(row["bbox_x1"]))
        y1 = max(0, int(row["bbox_y1"]))
        x2 = min(frame.shape[1], int(row["bbox_x2"]))
        y2 = min(frame.shape[0], int(row["bbox_y2"]))
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            lengths.append(0)
            continue
        if row["class"] == "pothole":
            w = x2 - x1
            h = y2 - y1
            lengths.append(np.sqrt(w*w + h*h))
            continue
        gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3)))
        try:
            import cv2.ximgproc as xip
            skeleton = xip.thinning(edges)
        except Exception:
            skeleton = edges
        lengths.append(np.count_nonzero(skeleton))
    return float(sum(lengths)), float(np.mean(lengths))


# ===============================
# DAMAGE DENSITY
# ===============================

def damage_density_score(det_df, img_area):
    if det_df.empty:
        return 0.0
    area_units = img_area / 10000
    return len(det_df) / area_units


# ===============================
# COMPOSITE SEVERITY
# ===============================

def composite_severity(total_bbox_ratio, total_crack_px, density, img_diag):
    c_bbox    = min(total_bbox_ratio, 1.0)
    max_crack = 0.05 * img_diag * 10
    c_crack   = min(total_crack_px / max(max_crack, 1), 1.0)
    c_density = min(density / 20.0, 1.0)
    return round(W_BBOX*c_bbox + W_CRACK*c_crack + W_DENSITY*c_density, 6)


# ===============================
# FRAME PROCESSOR
# ===============================

def process_frame(args):
    frame_name, det_records, IMG_W, IMG_H = args
    try:
        frame_path = os.path.join(FRAMES_DIR, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            frame = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        det_df    = pd.DataFrame(det_records)
        img_area  = IMG_W * IMG_H
        img_diag  = np.sqrt(img_area)

        total_area, mean_area, max_area = bbox_area_scores(det_df, img_area)
        total_crack, mean_crack         = crack_length_scores(frame, det_df)
        density                         = damage_density_score(det_df, img_area)
        score                           = composite_severity(total_area, total_crack, density, img_diag)

        mean_uncertainty = float(det_df["uncertainty"].mean())    if not det_df.empty else 0.0
        mean_confidence  = float(det_df["confidence_mean"].mean()) if not det_df.empty else 0.0
        class_counts     = det_df["class"].value_counts().to_dict() if "class" in det_df.columns else {}

        return {
            "frame":                 frame_name,
            "num_detections":        len(det_df),
            "total_bbox_area_ratio": round(total_area,   6),
            "mean_bbox_area_ratio":  round(mean_area,    6),
            "max_bbox_area_ratio":   round(max_area,     6),
            "total_crack_length_px": round(total_crack,  2),
            "mean_crack_length_px":  round(mean_crack,   2),
            "damage_density":        round(density,      6),
            "mean_confidence":       round(mean_confidence,  6),
            "mean_uncertainty":      round(mean_uncertainty, 6),
            "severity_score":        score,
            "class_counts":          json.dumps(class_counts),
        }
    except Exception as e:
        print(f"[WARN] Error processing {frame_name}: {e}")
        return None


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print("="*50)
    print(f"URSA-Net — Severity Estimation  ({cfg.video})")
    print("="*50)

    NUM_WORKERS = get_available_workers()
    check_disk(OUTPUT_CSV)

    print("\nLoading CSVs...")
    det_df      = pd.read_csv(DETECTIONS_CSV)
    reliability = pd.read_csv(RELIABILITY_CSV)

    IMG_W, IMG_H = infer_image_size(det_df)
    print(f"Detected resolution: {IMG_W} x {IMG_H}")
    print(f"Detections: {len(det_df)}")
    print(f"Frames: {reliability['frame'].nunique()}")

    required_cols = {"frame", "class", "confidence_mean", "uncertainty",
                     "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    missing = required_cols - set(det_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    det_dict = {
        frame: group.to_dict("records")
        for frame, group in det_df.groupby("frame")
    }

    all_frames = sorted(reliability["frame"].unique())
    print(f"Total frames to score: {len(all_frames)}")

    tasks = [(frame, det_dict.get(frame, []), IMG_W, IMG_H) for frame in all_frames]

    print("\nRunning severity estimation...\n")

    results = []
    errors  = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_frame, t) for t in tasks]
        for i, future in enumerate(
                tqdm(as_completed(futures), total=len(futures), desc="Severity"), 1):
            if i % 200 == 0:
                check_ram()
            res = future.result()
            if res is None:
                errors += 1
            else:
                results.append(res)

    severity_df = pd.DataFrame(results).sort_values("frame")

    merged = severity_df.merge(
        reliability[["frame", "blur_score", "brightness_score",
                     "motion_score", "reliability"]],
        on="frame", how="left"
    )

    merged.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved to: {OUTPUT_CSV}")
    scores = merged["severity_score"]
    print("\n---- Severity Summary ----")
    print(f"Frames scored: {len(merged)}")
    print(f"Errors: {errors}")
    print(f"Mean severity: {scores.mean():.4f}")
    print(f"Median severity: {scores.median():.4f}")
    print(f"Max severity: {scores.max():.4f}")
    print(f"High risk (>0.5): {(scores>0.5).sum()}")
    print(f"Medium risk (0.3-0.5): {((scores>=0.3)&(scores<=0.5)).sum()}")
    print(f"Low risk (<0.3): {(scores<0.3).sum()}")
    print("--------------------------")