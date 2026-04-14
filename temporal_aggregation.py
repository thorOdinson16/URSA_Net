import os
import re
import json
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from video_config import cfg

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SEVERITY_CSV   = cfg.out("severity_scores_geo.csv")
OUTPUT_CSV     = cfg.out("segment_severity.csv")

SEGMENT_SIZE   = 30
SEGMENT_METRES = 15
SEV_LOW        = 0.20
SEV_HIGH       = 0.50
SEVERITY_STD_NORM = 0.3


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def check_ram(limit: float = 0.90) -> None:
    vm = psutil.virtual_memory()
    if vm.percent / 100 > limit:
        print(f"[WARNING] RAM usage high: {vm.percent:.1f}% "
              f"(free {vm.available / 1e9:.2f} GB)")


def frame_index(frame_name: str) -> int:
    nums = re.findall(r"\d+", os.path.splitext(frame_name)[0])
    return int(nums[-1]) if nums else -1


def severity_label(score: float) -> str:
    if score < SEV_LOW:
        return "low"
    if score < SEV_HIGH:
        return "medium"
    return "high"


def dominant_class(class_counts_series: pd.Series) -> str:
    totals: dict = {}
    for entry in class_counts_series.dropna():
        try:
            counts = json.loads(entry)
            for cls, cnt in counts.items():
                totals[cls] = totals.get(cls, 0) + cnt
        except (json.JSONDecodeError, TypeError):
            continue
    return max(totals, key=totals.get) if totals else "none"


# ------------------------------------------------------------
# WEIGHTING
# ------------------------------------------------------------

def compute_weights(reliability: np.ndarray,
                    uncertainty: np.ndarray) -> np.ndarray:
    inv_uncertainty = 1.0 - np.clip(uncertainty, 0.0, 1.0)
    raw   = reliability * inv_uncertainty
    total = raw.sum()
    if total == 0:
        return np.ones(len(raw)) / len(raw)
    return raw / total


# ------------------------------------------------------------
# SEGMENT AGGREGATION
# ------------------------------------------------------------

def aggregate_segment(seg_df: pd.DataFrame) -> dict:
    reliability  = seg_df["reliability"].to_numpy(dtype=float)
    uncertainty  = seg_df["mean_uncertainty"].to_numpy(dtype=float)
    severity     = seg_df["severity_score"].to_numpy(dtype=float)
    confidence   = seg_df["mean_confidence"].to_numpy(dtype=float)
    n_detections = seg_df["num_detections"].to_numpy(dtype=float)

    weights         = compute_weights(reliability, uncertainty)
    w_severity      = float(np.dot(weights, severity))
    w_confidence    = float(np.dot(weights, confidence))
    mean_detections = float(n_detections.mean())
    severity_std    = float(severity.std())
    normalized_uncertainty = float(np.clip(severity_std / SEVERITY_STD_NORM, 0.0, 1.0))

    centroid_lat = (round(float(seg_df["lat"].mean()), 8)
                    if "lat" in seg_df.columns and seg_df["lat"].notna().any()
                    else None)
    centroid_lon = (round(float(seg_df["lon"].mean()), 8)
                    if "lon" in seg_df.columns and seg_df["lon"].notna().any()
                    else None)

    return {
        "segment_id":           seg_df["segment_id"].iloc[0],
        "first_frame":          seg_df["frame"].iloc[0],
        "last_frame":           seg_df["frame"].iloc[-1],
        "num_frames":           len(seg_df),
        "weighted_severity":    round(w_severity,              6),
        "weighted_uncertainty": round(normalized_uncertainty,  6),
        "weighted_confidence":  round(w_confidence,            6),
        "mean_reliability":     round(float(reliability.mean()), 6),
        "mean_detections":      round(mean_detections, 2),
        "severity_std":         round(severity_std,    6),
        "centroid_lat":         centroid_lat,
        "centroid_lon":         centroid_lon,
        "dominant_class":       dominant_class(seg_df["class_counts"]),
        "severity_label":       severity_label(w_severity),
        "reinspection_flag":    int(normalized_uncertainty > 0.5 or mean_detections > 3),
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print(f"URSA-Net — Temporal Evidence Accumulator  ({cfg.video})")
    print("=" * 55)

    print("\nLoading severity scores...")
    if not os.path.exists(SEVERITY_CSV):
        raise FileNotFoundError(f"Missing: {SEVERITY_CSV}")

    df = pd.read_csv(SEVERITY_CSV)
    print(f"  Rows loaded: {len(df)}")

    df["frame_idx"] = df["frame"].apply(frame_index)
    df = df.sort_values("frame_idx").reset_index(drop=True)

    if "gps_distance_m" in df.columns and df["gps_distance_m"].notna().sum() > 0:
        print(f"\nUsing GPS distance-based segmentation ({SEGMENT_METRES} m/segment)")
        df["segment_id"] = (df["gps_distance_m"] // SEGMENT_METRES).astype(int)
    else:
        print(f"\nGPS data missing — falling back to frame-window segmentation "
              f"({SEGMENT_SIZE} frames/segment)")
        df["segment_id"] = (df["frame_idx"] // SEGMENT_SIZE).astype(int)

    n_segments = df["segment_id"].nunique()
    print(f"Total frames  : {len(df)}")
    print(f"Total segments: {n_segments}")

    print("\nAggregating segments...\n")
    check_ram()

    records = []
    for seg_id, group in tqdm(df.groupby("segment_id"),
                               total=n_segments, unit="seg"):
        records.append(aggregate_segment(group))

    out_df = pd.DataFrame(records)

    print("\n" + "=" * 55)
    print("SEGMENT SUMMARY")
    print("=" * 55)
    print(f"  Total segments     : {len(out_df)}")
    print(f"  Mean severity      : {out_df['weighted_severity'].mean():.4f}")
    print(f"  Mean uncertainty   : {out_df['weighted_uncertainty'].mean():.4f}")
    print(f"  Mean severity_std  : {out_df['severity_std'].mean():.4f}")
    print(f"  Reinspection flags : {out_df['reinspection_flag'].sum()}")
    geo_coverage = out_df["centroid_lat"].notna().sum()
    print(f"  Segments with GPS  : {geo_coverage} / {len(out_df)}")
    print()

    for label, count in out_df["severity_label"].value_counts().items():
        pct = 100 * count / len(out_df)
        print(f"  {label:<8} segments: {count:>4}  ({pct:.1f}%)")

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")
    print("\nTemporal aggregation complete.")