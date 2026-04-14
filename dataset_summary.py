# ------------------------------------------------------------
# dataset_summary.py — URSA-Net (FINAL FIXED VERSION)
# ------------------------------------------------------------

import os
import re
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

OUTPUTS_DIR = "outputs"
GPS_DIR     = "GPS"

OUTPUT_CSV  = "dataset_summary.csv"
OUTPUT_TXT  = "dataset_summary.txt"

GPX_NS = "http://www.topografix.com/GPX/1/0"

DAY_START = 6
DAY_END   = 18
EVE_END   = 20


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def get_gpx_start_time_ist(gpx_path: str):
    try:
        tree = ET.parse(gpx_path)
        root = tree.getroot()

        times = []

        for trkpt in root.iter(f"{{{GPX_NS}}}trkpt"):
            time_el = trkpt.find(f"{{{GPX_NS}}}time")
            if time_el is not None:
                t_utc = datetime.fromisoformat(
                    time_el.text.replace("Z", "+00:00")
                )
                times.append(t_utc)

        if not times:
            return None

        # earliest timestamp
        t_utc = min(times)

        # correct UTC → IST
        t_ist = t_utc + timedelta(hours=5, minutes=30)

        ist_hour = (
            t_ist.hour +
            t_ist.minute / 60 +
            t_ist.second / 3600
        )

        return ist_hour

    except Exception:
        return None


def time_of_day_label(ist_hour):
    if ist_hour is None:
        return "unknown"
    if DAY_START <= ist_hour < DAY_END:
        return "day"
    if DAY_END <= ist_hour < EVE_END:
        return "evening"
    return "night"


def get_track_length_km(gpx_path: str):
    import math
    try:
        tree = ET.parse(gpx_path)
        root = tree.getroot()

        pts = []
        for trkpt in root.iter(f"{{{GPX_NS}}}trkpt"):
            pts.append((
                float(trkpt.attrib["lat"]),
                float(trkpt.attrib["lon"])
            ))

        if len(pts) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(pts)):
            lat1, lon1 = pts[i - 1]
            lat2, lon2 = pts[i]

            R = 6371000
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlam = math.radians(lon2 - lon1)

            a = (
                math.sin(dphi / 2) ** 2 +
                math.cos(phi1) * math.cos(phi2) *
                math.sin(dlam / 2) ** 2
            )

            total += 2 * R * math.asin(math.sqrt(a))

        return round(total / 1000, 3)

    except Exception:
        return 0.0


def dominant_class_from_csv(path):
    try:
        df = pd.read_csv(path)
        totals = {}

        for val in df["class_counts"].dropna():
            try:
                cc = json.loads(val)
                for k, v in cc.items():
                    totals[k] = totals.get(k, 0) + v
            except:
                continue

        return max(totals, key=totals.get) if totals else "none"

    except:
        return "none"


def safe_read(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()


# ------------------------------------------------------------
# DISCOVER
# ------------------------------------------------------------

def discover_videos():
    videos = []
    for name in sorted(os.listdir(OUTPUTS_DIR)):
        full = os.path.join(OUTPUTS_DIR, name)
        if os.path.isdir(full):
            videos.append((name, full))
    return videos


# ------------------------------------------------------------
# PROCESS
# ------------------------------------------------------------

def process_video(video_stem, out_dir):
    row = {"video": video_stem}

    match = re.match(r"\d+", video_stem)
    row["road_name"] = match.group() if match else ""

    gpx_path = os.path.join(GPS_DIR, video_stem + ".gpx")

    if os.path.exists(gpx_path):
        ist_hour = get_gpx_start_time_ist(gpx_path)

        row["time_of_day"] = time_of_day_label(ist_hour)
        row["ist_hour"] = round(ist_hour, 2) if ist_hour is not None else None
        row["track_length_km"] = get_track_length_km(gpx_path)
    else:
        row["time_of_day"] = "unknown"
        row["ist_hour"] = None
        row["track_length_km"] = None

    # Frames
    rel = safe_read(os.path.join(out_dir, "frame_reliability.csv"))
    if not rel.empty:
        row["frames_extracted"] = len(rel)
        if "reliability" in rel.columns:
            kept = (rel["reliability"] >= 0.35).sum()
            row["frames_retained"] = int(kept)
            row["retention_pct"] = round(100 * kept / len(rel), 2)

    # Uncertainty
    unc = safe_read(os.path.join(out_dir, "uncertainty_predictions.csv"))
    if not unc.empty:
        row["total_detections"] = len(unc)
        if "uncertainty" in unc.columns:
            row["mean_uncertainty"] = round(float(unc["uncertainty"].mean()), 6)

    # Severity
    sev_path = os.path.join(out_dir, "severity_scores.csv")
    sev = safe_read(sev_path)
    if not sev.empty and "severity_score" in sev.columns:
        row["mean_severity"] = round(float(sev["severity_score"].mean()), 6)
        row["dominant_class"] = dominant_class_from_csv(sev_path)

    # Decision
    dec = safe_read(os.path.join(out_dir, "decision_output.csv"))
    if not dec.empty:
        row["total_segments"] = len(dec)

        if "maintenance_action" in dec.columns:
            row["urgent"] = int((dec["maintenance_action"] == "urgent").sum())
            row["schedule"] = int((dec["maintenance_action"] == "schedule").sum())
            row["monitor"] = int((dec["maintenance_action"] == "monitor").sum())

        if "reinspection_flag" in dec.columns:
            row["reinspection_flags"] = int(dec["reinspection_flag"].sum())

        if "priority_score" in dec.columns:
            row["mean_priority_score"] = round(
                float(dec["priority_score"].mean()), 6
            )

    return row


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("URSA-Net — Dataset Summary Generator (FINAL)")
    print("=" * 60)

    videos = discover_videos()
    print(f"\nFound {len(videos)} video output folders\n")

    records = []
    for stem, path in tqdm(videos, desc="Processing"):
        records.append(process_video(stem, path))

    df = pd.DataFrame(records)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")

    # ---------------- SUMMARY ----------------

    lines = []
    lines.append("=" * 60)
    lines.append("URSA-Net — DATASET SUMMARY")
    lines.append("=" * 60)

    lines.append(f"Videos processed       : {len(df)}")

    if "track_length_km" in df.columns:
        lines.append(f"Total road surveyed    : {df['track_length_km'].sum():.2f} km")

    if "frames_extracted" in df.columns:
        lines.append(f"Total frames extracted : {df['frames_extracted'].sum():,}")

    if "frames_retained" in df.columns:
        lines.append(f"Total frames retained  : {df['frames_retained'].sum():,}")

    if "retention_pct" in df.columns:
        lines.append(f"Mean retention rate    : {df['retention_pct'].mean():.2f}%")

    if "total_detections" in df.columns:
        lines.append(f"Total detections       : {df['total_detections'].sum():,}")

    if "mean_uncertainty" in df.columns:
        lines.append(f"Mean uncertainty       : {df['mean_uncertainty'].mean():.6f}")

    if "mean_severity" in df.columns:
        lines.append(f"Mean severity score    : {df['mean_severity'].mean():.6f}")

    if "total_segments" in df.columns:
        lines.append(f"Total segments         : {df['total_segments'].sum():,}")
        lines.append(f"Urgent segments        : {df['urgent'].sum():,}")
        lines.append(f"Schedule segments      : {df['schedule'].sum():,}")
        lines.append(f"Monitor segments       : {df['monitor'].sum():,}")
        lines.append(f"Reinspection flags     : {df['reinspection_flags'].sum():,}")

    lines.append("")
    lines.append("-- Time of day distribution --")
    if "time_of_day" in df.columns:
        for k, v in df["time_of_day"].value_counts().items():
            lines.append(f"{k:<10} : {v} videos")

    lines.append("")
    lines.append("-- Dominant class distribution --")
    if "dominant_class" in df.columns:
        for k, v in df["dominant_class"].value_counts().items():
            lines.append(f"{k:<25} : {v} videos")

    lines.append("=" * 60)

    summary = "\n".join(lines)

    print("\n" + summary)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\nSummary saved -> {OUTPUT_TXT}")

if __name__ == "__main__":
    main()