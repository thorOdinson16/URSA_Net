# ------------------------------------------------------------
# UNCERTAINTY ESTIMATION MODULE
# Test-Time Augmentation (TTA) based prediction uncertainty
#
# Uncertainty = variance of detection confidence scores
# across TTA runs with different visual perturbations.
# Measures how consistent the model is under realistic
# input variation — directly reflects visual ambiguity.
# ------------------------------------------------------------

import os
import time
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

MODEL_PATH    = "runs/detect/augmented_model/weights/best.pt"
from video_config import cfg
INPUT_FOLDER = cfg.filtered_frames_dir()
OUTPUT_CSV   = cfg.out("tta_uncertainty_predictions.csv")

TTA_RUNS       = 4              # augmented forward passes per batch
CONF_THRESHOLD = 0.25
BATCH_SIZE     = 32             # reduced vs plain inference (TTA is heavier)
RAM_LIMIT      = 90


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

print("\nLoading YOLO model...")

model = YOLO(MODEL_PATH)

# keep model in eval mode — TTA handles stochasticity
# via input augmentation, not weight perturbation
model.model.eval()

print("Model loaded successfully\n")


# ------------------------------------------------------------
# LOAD FRAME LIST
# ------------------------------------------------------------

if not os.path.exists(INPUT_FOLDER):
    raise FileNotFoundError("filtered_frames folder not found.")

image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

if len(image_files) == 0:
    raise RuntimeError("No frames found.")

print(f"Total frames:  {len(image_files)}")
print(f"TTA runs:      {TTA_RUNS}")
print(f"Batch size:    {BATCH_SIZE}\n")


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

records = []


# ------------------------------------------------------------
# PROCESS BATCHES
# ------------------------------------------------------------

for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing batches"):

    wait_for_ram()

    batch_files = image_files[i : i + BATCH_SIZE]

    batch_paths = [
        os.path.join(INPUT_FOLDER, f)
        for f in batch_files
    ]

    tta_predictions = []

    # --------------------------------------------
    # TTA RUNS
    # augment=True activates YOLO's built-in TTA:
    # flips + multi-scale, giving genuine variance
    # across runs due to different augment seeds
    # --------------------------------------------

    for run in range(TTA_RUNS):

        results = model.predict(
            source=batch_paths,
            conf=CONF_THRESHOLD,
            imgsz=640,
            device=0,
            batch=BATCH_SIZE,
            half=True,
            augment=True,       # <-- the key change: TTA enabled
            verbose=False
        )

        tta_predictions.append(results)

    # --------------------------------------------
    # COMPUTE UNCERTAINTY PER FRAME
    # --------------------------------------------

    for img_idx, frame_name in enumerate(batch_files):

        # collect all detections across TTA runs for this frame
        detections = []

        for run in range(TTA_RUNS):

            r = tta_predictions[run][img_idx]

            if r.boxes is None:
                continue

            for box in r.boxes:

                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                xyxy   = box.xyxy[0].cpu().numpy()

                detections.append({
                    "run":  run,
                    "conf": conf,
                    "cls":  cls_id,
                    "bbox": xyxy
                })

        if len(detections) == 0:
            continue

        # group detections by class across all TTA runs
        class_groups = {}

        for det in detections:

            cls = det["cls"]

            if cls not in class_groups:
                class_groups[cls] = []

            class_groups[cls].append(det["conf"])

        # compute uncertainty = variance of confidence scores
        # higher variance → model is less consistent → more uncertain
        for cls_id, conf_list in class_groups.items():

            uncertainty = float(np.var(conf_list))
            conf_mean   = float(np.mean(conf_list))

            class_name = (
                CLASS_NAMES[cls_id]
                if cls_id < len(CLASS_NAMES)
                else str(cls_id)
            )

            # take the bbox from the highest-confidence detection
            best_det = max(
                (d for d in detections if d["cls"] == cls_id),
                key=lambda d: d["conf"]
            )

            records.append({
                "frame":            frame_name,
                "class":            class_name,
                "confidence_mean":  round(conf_mean, 6),
                "uncertainty":      round(uncertainty, 6),
                "tta_runs":         len(conf_list),
                "bbox_x1":          best_det["bbox"][0],
                "bbox_y1":          best_det["bbox"][1],
                "bbox_x2":          best_det["bbox"][2],
                "bbox_y2":          best_det["bbox"][3],
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

print("===================================")