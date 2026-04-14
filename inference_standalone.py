# --------------------------------------------
# YOLO DAMAGE DETECTION INFERENCE SCRIPT
# Runs YOLO on filtered frames from reliability pipeline
# --------------------------------------------

from ultralytics import YOLO
import os
import psutil
import time

# -------------------------------------------------
# MODEL PATH
# -------------------------------------------------

MODEL_PATH = "runs/detect/augmented_model/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# -------------------------------------------------
# CLASS NAMES
# -------------------------------------------------

CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole"
]

# -------------------------------------------------
# INPUT FOLDER (FROM RELIABILITY FILTER)
# -------------------------------------------------

SOURCE_FOLDER = "filtered_frames"

if not os.path.exists(SOURCE_FOLDER):
    raise FileNotFoundError(
        "filtered_frames folder not found. Run reliability_filter.py first."
    )

# -------------------------------------------------
# RAM GUARDRAIL
# -------------------------------------------------

def wait_for_ram():
    while True:
        ram_usage = psutil.virtual_memory().percent

        if ram_usage < 90:
            break

        print(f"RAM usage high ({ram_usage}%). Waiting...")
        time.sleep(5)


# -------------------------------------------------
# RUN INFERENCE
# -------------------------------------------------

print("\nStarting YOLO inference on filtered frames...\n")

wait_for_ram()

results = model.track(
    source=SOURCE_FOLDER,
    conf=0.25,
    imgsz=640,
    device=0,
    save=True,
    stream=True,
    tracker="bytetrack.yaml"   # ByteTrack built into YOLOv8
)

# -------------------------------------------------
# PRINT DETECTIONS
# -------------------------------------------------

for r in results:

    if r.boxes is None:
        continue

    for box in r.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

        print(f"Detected {label} with confidence {conf:.2f}")

print("\nInference completed.")
print("Results saved in: runs/detect/")