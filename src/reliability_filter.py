import pandas as pd
import shutil
import os
import psutil
import time
from tqdm import tqdm

from video_config import cfg

# ==========================================
# CONFIG
# ==========================================

CSV_FILE      = cfg.out("frame_reliability.csv")
FRAME_FOLDER  = cfg.frames_dir()
OUTPUT_FOLDER = cfg.filtered_frames_dir()

THRESHOLD    = 0.35
RAM_LIMIT    = 90
CLEAN_OUTPUT = True


# ==========================================
# RAM GUARDRAIL
# ==========================================

def wait_for_ram():
    while True:
        usage = psutil.virtual_memory().percent
        if usage < RAM_LIMIT:
            break
        print(f"RAM usage high ({usage}%). Waiting...")
        time.sleep(2)


# ==========================================
# VALIDATION
# ==========================================

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"frame_reliability.csv not found: {CSV_FILE}")

if not os.path.exists(FRAME_FOLDER):
    raise FileNotFoundError(f"Frames folder not found: {FRAME_FOLDER}")


# ==========================================
# PREPARE OUTPUT FOLDER
# ==========================================

if CLEAN_OUTPUT and os.path.exists(OUTPUT_FOLDER):
    for f in os.listdir(OUTPUT_FOLDER):
        os.remove(os.path.join(OUTPUT_FOLDER, f))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# FILTERING
# ==========================================

df = pd.read_csv(CSV_FILE)
total_frames = len(df)

print("\n================================")
print(f"Frame Filtering Stage — {cfg.video}")
print("Threshold:", THRESHOLD)
print("Frames to evaluate:", total_frames)
print("================================")

kept = removed = missing = 0

for row in tqdm(df.itertuples(index=False), total=total_frames, desc="Filtering frames"):
    src = os.path.join(FRAME_FOLDER, row.frame)
    dst = os.path.join(OUTPUT_FOLDER, row.frame)

    if not os.path.exists(src):
        missing += 1
        continue

    wait_for_ram()

    if row.reliability >= THRESHOLD:
        shutil.copy(src, dst)
        kept += 1
    else:
        removed += 1

print("\nFiltering complete")
print("--------------------------------")
print("Frames kept      :", kept)
print("Frames removed   :", removed)
print("Missing frames   :", missing)
print("Total processed  :", total_frames)
if total_frames > 0:
    print("Retention rate   :", round(kept / total_frames * 100, 2), "%")
print("--------------------------------")