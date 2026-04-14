import os
import time
import psutil
from ultralytics import YOLO

from video_config import cfg

MODEL_PATH     = "runs/detect/augmented_model/weights/best.pt"
INPUT_FOLDER   = cfg.filtered_frames_dir()
CONF_THRESHOLD = 0.25


def wait_for_ram():
    while True:
        if psutil.virtual_memory().percent < 90:
            break
        print("RAM high. Waiting...")
        time.sleep(3)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(INPUT_FOLDER):
    raise FileNotFoundError(f"filtered_frames not found: {INPUT_FOLDER}")

print(f"\nLoading YOLO model for video: {cfg.video}")
model = YOLO(MODEL_PATH)
print("Model loaded.\n")

wait_for_ram()

print("Running tracking + annotation on filtered frames...\n")

results = model.track(
    source=INPUT_FOLDER,
    conf=CONF_THRESHOLD,
    imgsz=640,
    device=0,
    save=True,
    stream=True,
    tracker="bytetrack.yaml"
)

for r in results:
    pass

actual_track_dir = str(model.predictor.save_dir)

# Save for rebuild_video.py
with open(cfg.out("track_dir.txt"), "w") as f:
    f.write(actual_track_dir)

print("\nAnnotation complete.")
print(f"Annotated frames saved to: {actual_track_dir}")