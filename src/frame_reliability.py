import cv2
import numpy as np
import os
import pandas as pd
import psutil
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from video_config import cfg

# ============================================
# CONFIG
# ============================================

IMAGE_FOLDER = cfg.frames_dir()
OUTPUT_CSV   = cfg.out("frame_reliability.csv")
RAM_LIMIT    = 90

THRESHOLD    = 0.45

W_BLUR   = 0.5
W_BRIGHT = 0.3
W_MOTION = 0.2


# ============================================
# RAM GUARDRAIL
# ============================================

def wait_for_ram():
    while True:
        usage = psutil.virtual_memory().percent
        if usage < RAM_LIMIT:
            break
        print(f"RAM usage high ({usage}%). Waiting...")
        time.sleep(2)


# ============================================
# METRICS
# ============================================

def blur_score(gray):
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(variance / 500.0, 1.0))


def brightness_score(gray):
    mean_v = float(gray.mean())
    if mean_v < 30:
        return mean_v / 30.0
    if mean_v > 240:
        return (255.0 - mean_v) / 15.0
    return 1.0


def motion_blur_score(gray):
    h, w = gray.shape
    bh, bw = h // 4, w // 4
    block_means = []
    for r in range(4):
        for c in range(4):
            block = gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            block_means.append(block.mean())
    block_std = float(np.std(block_means))
    return min(block_std / 20.0, 1.0)


# ============================================
# PROCESS SINGLE FRAME
# ============================================

def process_frame(frame_name):
    path = os.path.join(IMAGE_FOLDER, frame_name)
    img  = cv2.imread(path)
    if img is None:
        return None

    wait_for_ram()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b    = blur_score(gray)
    br   = brightness_score(gray)
    m    = motion_blur_score(gray)
    reliability = W_BLUR * b + W_BRIGHT * br + W_MOTION * m

    return [frame_name, round(b, 4), round(br, 4), round(m, 4), round(reliability, 4)]


# ============================================
# MAIN
# ============================================

def main():
    if not os.path.exists(IMAGE_FOLDER):
        raise FileNotFoundError(f"{IMAGE_FOLDER} folder not found")

    files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")])
    if not files:
        raise ValueError(f"No frames found in {IMAGE_FOLDER}")

    print(f"\nProcessing {len(files)} frames for video: {cfg.video}")

    workers = cpu_count()
    print(f"Using {workers} CPU cores\n")

    with Pool(workers) as pool:
        results = list(
            tqdm(pool.imap(process_frame, files), total=len(files),
                 desc="Computing reliability")
        )

    results = [r for r in results if r is not None]

    df = pd.DataFrame(results, columns=[
        "frame", "blur_score", "brightness_score", "motion_score", "reliability"
    ])

    df.to_csv(OUTPUT_CSV, index=False)

    passing = (df["reliability"] >= THRESHOLD).sum()

    print("\n=================================")
    print("Reliability computation complete")
    print(f"Saved          -> {OUTPUT_CSV}")
    print(f"Frames total   : {len(df)}")
    print(f"Mean reliability: {df['reliability'].mean():.3f}")
    print(f"Median          : {df['reliability'].median():.3f}")
    print(f"Frames >= {THRESHOLD} (will pass filter): {passing} "
          f"({100*passing/len(df):.1f}%)")
    print("=================================")


if __name__ == "__main__":
    main()