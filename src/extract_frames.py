import cv2
import os
import shutil
import time
import psutil
from tqdm import tqdm

from video_config import cfg

# ============================================
# CONFIG (resolved from env via video_config)
# ============================================

VIDEO_PATH   = os.path.join(cfg.gps_dir, cfg.video + ".mp4")
FRAME_FOLDER = cfg.frames_dir()
RAM_LIMIT    = 90


# ============================================
# RAM GUARDRAIL
# ============================================

def wait_for_ram():
    while True:
        ram_usage = psutil.virtual_memory().percent
        if ram_usage < RAM_LIMIT:
            break
        print(f"RAM usage high ({ram_usage}%). Waiting...")
        time.sleep(3)


# ============================================
# MAIN
# ============================================

def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    # Clear any frames from a previous run of this video
    if os.path.exists(FRAME_FOLDER):
        shutil.rmtree(FRAME_FOLDER)
        print(f"Cleared old frames: {FRAME_FOLDER}")

    os.makedirs(FRAME_FOLDER, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo:        {cfg.video}")
    print(f"Total frames: {total_frames}  |  FPS: {fps:.1f}")
    print(f"Output dir:   {FRAME_FOLDER}\n")

    frame_id  = 0
    extracted = 0

    for _ in tqdm(range(total_frames), desc=cfg.video):
        ret, frame = cap.read()
        if not ret:
            break

        wait_for_ram()

        frame_name = f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(os.path.join(FRAME_FOLDER, frame_name), frame)

        frame_id  += 1
        extracted += 1

    cap.release()

    print(f"\nExtracted {extracted} frames from {cfg.video}")
    print(f"Saved to: {FRAME_FOLDER}")
    print("=================================")


if __name__ == "__main__":
    main()