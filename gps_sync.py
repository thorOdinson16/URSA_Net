# ------------------------------------------------------------
# GPS SYNC MODULE — URSA-Net
# Joins video frame indices with GPX tracks to produce
# geotagged_frames.csv: frame, video, lat, lon, ele,
#                        speed_kmh, heading, gps_distance_m
#
# Frame naming is PER-VIDEO (resets to frame_000000 for each
# video). Join key downstream is (frame, video) together.
#
# Input  (all in DATA_DIR):
#   1IronShop.gpx + 1IronShop.mp4
#   2KaryaSiddhi.gpx + 2KaryaSiddhi.mp4
#   ... (32 pairs total)
#
# Output:
#   geotagged_frames.csv
#
# NOTE: Video start time is taken directly from the first GPX
# trackpoint timestamp. This assumes the GPS logger and camera
# were started simultaneously, which gives full GPS coverage
# across the entire video duration.
# ------------------------------------------------------------

import os
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import cv2
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DATA_DIR    = "GPS"
OUTPUT_CSV  = "geotagged_frames.csv"
DEFAULT_FPS = 30.0

# ------------------------------------------------------------
# GPX PARSER
# ------------------------------------------------------------

GPX_NS = "http://www.topografix.com/GPX/1/0"

def parse_gpx(gpx_path):
    tree  = ET.parse(gpx_path)
    root  = tree.getroot()
    points = []

    for trkpt in root.iter(f"{{{GPX_NS}}}trkpt"):
        lat     = float(trkpt.attrib["lat"])
        lon     = float(trkpt.attrib["lon"])
        ele_el  = trkpt.find(f"{{{GPX_NS}}}ele")
        time_el = trkpt.find(f"{{{GPX_NS}}}time")

        if time_el is None:
            continue

        ele = float(ele_el.text) if ele_el is not None else 0.0
        t   = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
        points.append({"time": t, "lat": lat, "lon": lon, "ele": ele})

    points.sort(key=lambda p: p["time"])
    return points

# ------------------------------------------------------------
# GEOMETRY
# ------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R    = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x    = math.sin(dlam) * math.cos(phi2)
    y    = (math.cos(phi1) * math.sin(phi2)
            - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

# ------------------------------------------------------------
# TRACK ENRICHMENT
# ------------------------------------------------------------

def enrich_track(points):
    """
    Adds speed_kmh, heading, dist_from_start to every trackpoint.
    dist_from_start is cumulative metres from track start (resets
    per video) — used by temporal_aggregation.py for GPS segmentation.
    Point 0 heading is forward-filled from point 1.
    """
    if len(points) >= 2:
        points[0]["heading"] = bearing_deg(
            points[0]["lat"], points[0]["lon"],
            points[1]["lat"], points[1]["lon"]
        )
    else:
        points[0]["heading"] = 0.0

    points[0]["speed_kmh"]       = 0.0
    points[0]["dist_from_start"] = 0.0

    cumulative = 0.0
    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]
        dt   = (curr["time"] - prev["time"]).total_seconds()
        dist = haversine_m(prev["lat"], prev["lon"],
                           curr["lat"], curr["lon"])
        cumulative              += dist
        curr["speed_kmh"]        = (dist / dt * 3.6) if dt > 0 else 0.0
        curr["heading"]          = bearing_deg(prev["lat"], prev["lon"],
                                               curr["lat"], curr["lon"])
        curr["dist_from_start"]  = cumulative

    return points

# ------------------------------------------------------------
# INTERPOLATION
# ------------------------------------------------------------

def interpolate_gps(points, query_time_utc):
    times = [p["time"] for p in points]

    if query_time_utc < times[0] or query_time_utc > times[-1]:
        return None

    lo, hi = 0, len(points) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if times[mid] <= query_time_utc:
            lo = mid
        else:
            hi = mid

    p0, p1 = points[lo], points[hi]
    span   = (p1["time"] - p0["time"]).total_seconds()
    t      = (query_time_utc - p0["time"]).total_seconds()
    alpha  = (t / span) if span > 0 else 0.0

    def lerp(a, b): return a + alpha * (b - a)

    return {
        "lat":             lerp(p0["lat"],             p1["lat"]),
        "lon":             lerp(p0["lon"],             p1["lon"]),
        "ele":             lerp(p0["ele"],             p1["ele"]),
        "speed_kmh":       lerp(p0["speed_kmh"],       p1["speed_kmh"]),
        "heading":         lerp(p0["heading"],          p1["heading"]),
        "dist_from_start": lerp(p0["dist_from_start"], p1["dist_from_start"]),
    }

# ------------------------------------------------------------
# FRAME COUNT + FPS
# ------------------------------------------------------------

def get_frame_count_and_fps(mp4_path):
    cap   = cv2.VideoCapture(mp4_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count, fps

# ------------------------------------------------------------
# PROCESS ONE PAIR
# ------------------------------------------------------------

def process_pair(stem, gpx_path, mp4_path):
    """
    Produces one row per frame. frame_name uses frame_idx
    (per-video, resets to 0). Join key downstream: (frame, video).

    Video start time = first GPX trackpoint timestamp.
    GPS logger and camera are assumed to have been started together.
    """
    print(f"\n  Processing: {stem}")

    points = parse_gpx(gpx_path)
    points = enrich_track(points)

    vid_start_utc = points[0]["time"]

    print(f"    GPX trackpoints  : {len(points)}")
    print(f"    GPS start (UTC)  : {points[0]['time']}")
    print(f"    GPS end   (UTC)  : {points[-1]['time']}")
    print(f"    Track length (m) : {points[-1]['dist_from_start']:.1f}")

    n_frames, fps = get_frame_count_and_fps(mp4_path)

    print(f"    Video start (UTC): {vid_start_utc}")
    print(f"    Frames / FPS     : {n_frames} / {fps:.1f}")

    rows    = []
    skipped = 0

    for frame_idx in tqdm(range(n_frames), desc=f"    {stem}", leave=False):
        query_t = vid_start_utc + timedelta(seconds=frame_idx / fps)
        geo     = interpolate_gps(points, query_t)

        if geo is None:
            skipped += 1
            continue

        rows.append({
            "frame":          f"frame_{frame_idx:06d}.jpg",
            "video":          stem,
            "frame_idx":      frame_idx,
            "timestamp_utc":  query_t.isoformat(),
            "lat":            round(geo["lat"],             8),
            "lon":            round(geo["lon"],             8),
            "ele":            round(geo["ele"],             2),
            "speed_kmh":      round(geo["speed_kmh"],       2),
            "heading":        round(geo["heading"],         2),
            "gps_distance_m": round(geo["dist_from_start"], 2),
        })

    print(f"    Geotagged : {len(rows)}  |  Outside GPS range: {skipped}")
    return rows

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("=" * 55)
    print("URSA-Net — GPS Sync")
    print("=" * 55)

    gpx_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".gpx")])

    pairs = []
    for gpx_file in gpx_files:
        stem     = gpx_file[:-4]
        mp4_path = os.path.join(DATA_DIR, stem + ".mp4")
        gpx_path = os.path.join(DATA_DIR, gpx_file)

        if not os.path.exists(mp4_path):
            print(f"[skip] No MP4 found for '{stem}'")
            continue

        pairs.append((stem, gpx_path, mp4_path))

    if not pairs:
        raise RuntimeError(f"No GPX+MP4 pairs found in '{DATA_DIR}'.")

    print(f"Found {len(pairs)} pairs: {[p[0] for p in pairs]}\n")

    all_rows = []
    for stem, gpx_path, mp4_path in pairs:
        all_rows.extend(process_pair(stem, gpx_path, mp4_path))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 55)
    print("GPS SYNC COMPLETE")
    print("=" * 55)
    print(f"  Total geotagged frames : {len(df)}")
    print(f"  Videos processed       : {df['video'].nunique()}")
    print(f"  Lat range              : {df['lat'].min():.5f} -> {df['lat'].max():.5f}")
    print(f"  Lon range              : {df['lon'].min():.5f} -> {df['lon'].max():.5f}")
    print(f"  Saved -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()