import os
import pandas as pd

from video_config import cfg

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SEVERITY_CSV = cfg.out("severity_scores.csv")
GEO_CSV      = "geotagged_frames.csv"   # global file produced by gps_sync.py
OUTPUT_CSV   = cfg.out("severity_scores_geo.csv")

CURRENT_VIDEO = cfg.video

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("URSA-Net — GPS Merge")
    print("=" * 55)
    print(f"\nCurrent video: {CURRENT_VIDEO}")

    # ── Load severity scores ─────────────────────────────────
    if not os.path.exists(SEVERITY_CSV):
        raise FileNotFoundError(f"Missing: {SEVERITY_CSV}")

    sev = pd.read_csv(SEVERITY_CSV)
    print(f"Severity rows loaded : {len(sev)}")

    if "video" not in sev.columns:
        print(f"  No 'video' column — setting all rows to '{CURRENT_VIDEO}'")
        sev["video"] = CURRENT_VIDEO

    # ── Load geo data ────────────────────────────────────────
    if not os.path.exists(GEO_CSV):
        raise FileNotFoundError(
            f"Missing: {GEO_CSV}  (run gps_sync.py first)"
        )

    geo = pd.read_csv(GEO_CSV)
    print(f"Geo rows loaded      : {len(geo)}")

    video_num = ''.join(filter(str.isdigit, CURRENT_VIDEO)).lstrip('0') or '0'
    geo = geo[geo["video"].astype(str) == video_num].copy()
    print(f"Geo rows for '{CURRENT_VIDEO}': {len(geo)}")

    if geo.empty:
        raise RuntimeError(
            f"No rows found for video '{CURRENT_VIDEO}' in {GEO_CSV}. "
            f"Check that gps_sync.py was run and the video stem matches."
        )

    geo = geo[["frame", "video", "lat", "lon", "ele",
               "speed_kmh", "heading", "gps_distance_m"]]
    
    sev["video"] = sev["video"].astype(str)
    geo["video"] = geo["video"].astype(str)

    merged = pd.merge(sev, geo, on=["frame", "video"], how="left")

    matched   = merged["lat"].notna().sum()
    unmatched = merged["lat"].isna().sum()

    print(f"\nMerge results:")
    print(f"  Matched (have GPS)   : {matched}")
    print(f"  Unmatched (no GPS)   : {unmatched}")

    if unmatched > 0:
        print(f"  [warn] {unmatched} frames have no GPS match — "
              "will fall back to frame-window segmentation in "
              "temporal_aggregation.py")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")
    print(f"Columns: {list(merged.columns)}")
    print("\nGPS merge complete.")