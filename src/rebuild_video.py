import cv2
import os
import shutil
import pandas as pd
from tqdm import tqdm

from video_config import cfg

# ===============================
# CONFIG
# ===============================

with open(cfg.out("track_dir.txt")) as f:
    FRAME_FOLDER = f.read().strip()
OUTPUT_VIDEO   = cfg.out("output_video.mp4")
SEVERITY_CSV   = cfg.out("severity_scores.csv")

FPS_NORMAL       = 30
FPS_DAMAGE       = 10
DAMAGE_THRESHOLD = 0.0

# Only clear the per-video temp folders, not global ones
TEMP_FOLDERS = [cfg.frames_dir(), cfg.filtered_frames_dir()]


# ===============================
# LOAD SEVERITY DATA
# ===============================

severity_df   = pd.read_csv(SEVERITY_CSV)
damage_frames = set(severity_df[severity_df["num_detections"] > 0]["frame"].tolist())

print(f"Frames with damage: {len(damage_frames)}")


# ===============================
# LOAD FRAME LIST
# ===============================

frames = sorted(
    f for f in os.listdir(FRAME_FOLDER)
    if f.lower().endswith((".jpg", ".png"))
)

if not frames:
    raise RuntimeError("No frames found.")

print(f"Found {len(frames)} frames")


# ===============================
# INIT VIDEO WRITER
# ===============================

first_path = os.path.join(FRAME_FOLDER, frames[0])
first = cv2.imread(first_path)

if first is None:
    raise RuntimeError("Could not read first frame")

h, w = first.shape[:2]

BASE_FPS      = 30
REPEAT_NORMAL = 1
REPEAT_DAMAGE = BASE_FPS // FPS_DAMAGE

video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    BASE_FPS,
    (w, h)
)


# ===============================
# BUILD VIDEO
# ===============================

for f in tqdm(frames, desc="Rebuilding video", unit="frame"):
    frame_path = os.path.join(FRAME_FOLDER, f)
    frame = cv2.imread(frame_path)
    if frame is None:
        continue
    base_name = os.path.splitext(f)[0] + ".jpg"
    is_damage = base_name in damage_frames
    repeat    = REPEAT_DAMAGE if is_damage else REPEAT_NORMAL
    for _ in range(repeat):
        video.write(frame)

video.release()

print(f"\nVideo saved as: {OUTPUT_VIDEO}")


# ===============================
# CLEAN TEMP FOLDERS
# ===============================

print("\nCleaning temporary folders...")
for folder in TEMP_FOLDERS:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted {folder}/")

print("Cleanup complete.")