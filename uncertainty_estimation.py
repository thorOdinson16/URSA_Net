import os
import time
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from video_config import cfg

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

MODEL_PATH    = "runs/detect/augmented_model/weights/best.pt"
INPUT_FOLDER  = cfg.filtered_frames_dir()
OUTPUT_CSV    = cfg.out("uncertainty_predictions.csv")

CONF_THRESHOLD = 0.25
BATCH_SIZE     = 48
RAM_LIMIT      = 90

# ------------------------------------------------------------
# CLASS NAMES & WEIGHTS
# ------------------------------------------------------------

CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole"
]

CLASS_WEIGHTS = {
    "pothole":            1.0,
    "alligator_crack":    0.9,
    "transverse_crack":   0.7,
    "longitudinal_crack": 0.6,
}

DEFAULT_CLASS_WEIGHT = 0.5


# ------------------------------------------------------------
# RAM GUARDRAIL
# ------------------------------------------------------------

def wait_for_ram():
    while True:
        ram_usage = psutil.virtual_memory().percent
        if ram_usage < RAM_LIMIT:
            break
        print(f"RAM usage high ({ram_usage}%). Waiting...")
        time.sleep(3)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print(f"\nLoading YOLO model for video: {cfg.video}")
model = YOLO(MODEL_PATH)
model.model.eval()
print("Model loaded successfully\n")


# ------------------------------------------------------------
# LOAD FRAME LIST
# ------------------------------------------------------------

if not os.path.exists(INPUT_FOLDER):
    raise FileNotFoundError(f"filtered_frames folder not found: {INPUT_FOLDER}")

image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

if len(image_files) == 0:
    raise RuntimeError(f"No frames found in {INPUT_FOLDER}")

print(f"Total frames: {len(image_files)}")
print(f"Batch size:   {BATCH_SIZE}\n")


# ------------------------------------------------------------
# PROCESS BATCHES
# ------------------------------------------------------------

records = []

for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing batches"):
    wait_for_ram()

    batch_files = image_files[i : i + BATCH_SIZE]
    batch_paths = [os.path.join(INPUT_FOLDER, f) for f in batch_files]

    results = model.predict(
        source=batch_paths,
        conf=CONF_THRESHOLD,
        imgsz=640,
        device=0,
        batch=BATCH_SIZE,
        half=True,
        augment=False,
        verbose=False
    )

    for img_idx, frame_name in enumerate(batch_files):
        r = results[img_idx]

        if r.boxes is None or len(r.boxes) == 0:
            continue

        scores = []
        for box in r.boxes:
            conf      = float(box.conf[0])
            cls_id    = int(box.cls[0])
            class_name = (CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES)
                          else str(cls_id))
            weight = CLASS_WEIGHTS.get(class_name, DEFAULT_CLASS_WEIGHT)
            scores.append(conf * weight)

        uncertainty = float(np.std(scores)) if len(scores) > 1 else 0.0
        n_dets      = len(scores)

        for box in r.boxes:
            conf      = float(box.conf[0])
            cls_id    = int(box.cls[0])
            xyxy      = box.xyxy[0].cpu().numpy()
            class_name = (CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES)
                          else str(cls_id))
            weight     = CLASS_WEIGHTS.get(class_name, DEFAULT_CLASS_WEIGHT)
            det_score  = conf * weight

            records.append({
                "frame":           frame_name,
                "class":           class_name,
                "confidence_mean": round(conf,        6),
                "weighted_score":  round(det_score,   6),
                "uncertainty":     round(uncertainty, 6),
                "num_detections":  n_dets,
                "bbox_x1":         xyxy[0],
                "bbox_y1":         xyxy[1],
                "bbox_x2":         xyxy[2],
                "bbox_y2":         xyxy[3],
            })


# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print("\n===================================")
print("Uncertainty estimation complete")
print(f"Saved to:         {OUTPUT_CSV}")
print(f"Total detections: {len(df)}")

if len(df) > 0:
    print(f"Mean uncertainty: {df['uncertainty'].mean():.6f}")
    print(f"Max uncertainty:  {df['uncertainty'].max():.6f}")
    unique_frames = df["frame"].nunique()
    zero_unc = (df.groupby("frame")["uncertainty"].first() == 0).sum()
    print(f"Frames processed: {unique_frames}")
    print(f"Single-detection frames (uncertainty=0): {zero_unc}")

print("===================================")