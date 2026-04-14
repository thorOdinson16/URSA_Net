import re
import glob
import folium
import numpy as np
from folium.plugins import HeatMap

# ==============================
# CONFIG
# ==============================
BASE_PATH = "outputs/*/plots/"
MARKER_FILE = "map_priority_markers.html"
HEAT_FILE = "map_severity_heatmap.html"

# ==============================
# STORAGE
# ==============================
all_markers = []
all_heat = []

# ==============================
# STEP 1: LOAD ALL FILES
# ==============================
folders = sorted(glob.glob(BASE_PATH))

for folder in folders:
    marker_path = folder + MARKER_FILE
    heat_path = folder + HEAT_FILE

    # ---------- MARKERS ----------
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            content = f.read()

            matches = re.findall(
                r"L\.circleMarker\(\s*\[\s*([\d\.]+),\s*([\d\.]+)\s*\].*?\"color\": \"(\w+)\".*?\"radius\": ([\d\.]+)",
                content,
                re.DOTALL
            )

            for lat, lon, color, radius in matches:
                all_markers.append((float(lat), float(lon), color, float(radius)))

    except FileNotFoundError:
        print(f"⚠️ Missing marker file: {marker_path}")

    # ---------- HEATMAP ----------
    try:
        with open(heat_path, "r", encoding="utf-8") as f:
            content = f.read()

            match = re.search(
                r"L\.heatLayer\(\s*(\[\[.*?\]\])",
                content,
                re.DOTALL
            )

            if match:
                data = eval(match.group(1))  # numeric only
                all_heat.extend(data)

    except FileNotFoundError:
        print(f"⚠️ Missing heatmap file: {heat_path}")

# ==============================
# STEP 2: COMPUTE BOUNDS
# ==============================
all_points = [(x[0], x[1]) for x in all_heat] + [(x[0], x[1]) for x in all_markers]

if all_points:
    lats = [p[0] for p in all_points]
    lons = [p[1] for p in all_points]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # 🔥 padding for clean borders
    lat_pad = (max_lat - min_lat) * 0.05
    lon_pad = (max_lon - min_lon) * 0.05

    bounds = [
        [min_lat - lat_pad, min_lon - lon_pad],
        [max_lat + lat_pad, max_lon + lon_pad]
    ]

    lat_mean = np.mean(lats)
    lon_mean = np.mean(lons)
else:
    bounds = None
    lat_mean, lon_mean = 12.9, 77.5

# ==============================
# STEP 3A: MARKER MAP (CLEAN PAPER STYLE)
# ==============================
marker_map = folium.Map(
    location=[lat_mean, lon_mean],
    zoom_start=15,
    tiles="cartodbpositron"  # ✅ LIGHT THEME (paper-friendly)
)

# Add markers (stronger styling)
for lat, lon, color, radius in all_markers:
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius * 0.5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        weight=2
    ).add_to(marker_map)

# Fit bounds
if bounds:
    marker_map.fit_bounds(bounds)

marker_map.save("master_markers.html")
print("master_markers.html created")

# ==============================
# STEP 3B: HEATMAP (OPTIONAL DARK)
# ==============================
heat_map = folium.Map(
    location=[lat_mean, lon_mean],
    zoom_start=15,
    tiles="CartoDB dark_matter"
)

if all_heat:
    HeatMap(
        all_heat,
        radius=14,
        blur=12,
        min_opacity=0.4
    ).add_to(heat_map)

if bounds:
    heat_map.fit_bounds(bounds)

heat_map.save("master_heatmap.html")
print("master_heatmap.html created")