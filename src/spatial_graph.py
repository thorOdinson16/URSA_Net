import os
import math
import pandas as pd
import folium
from folium.plugins import HeatMap
from tqdm import tqdm

from video_config import cfg

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SEVERITY_GEO_CSV = cfg.out("severity_scores_geo.csv")
SEGMENT_CSV      = cfg.out("segment_severity.csv")
DECISION_CSV     = cfg.out("decision_output.csv")
OUTPUT_DIR       = cfg.plots_dir()
GRAPH_CSV        = cfg.out("spatial_graph_edges.csv")

GRAPH_EDGE_DIST_M = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

def load_data():
    print("Loading data...")
    frame_df = pd.read_csv(SEVERITY_GEO_CSV)
    print(f"  Frame rows        : {len(frame_df)}")
    print(f"  Frames with GPS   : {frame_df['lat'].notna().sum()}")

    seg_df = pd.read_csv(SEGMENT_CSV)
    seg_df = seg_df.dropna(subset=["centroid_lat", "centroid_lon"])
    print(f"  Segments with GPS : {len(seg_df)}")

    return frame_df, seg_df


# ------------------------------------------------------------
# HAVERSINE
# ------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R    = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# ------------------------------------------------------------
# SPATIAL GRAPH
# ------------------------------------------------------------

def build_spatial_graph(seg_df):
    print("\nBuilding spatial graph...")
    edges = []
    segs  = seg_df.reset_index(drop=True)

    for i in tqdm(range(len(segs)), desc="Graph edges"):
        for j in range(i + 1, len(segs)):
            d = haversine_m(segs.loc[i, "centroid_lat"], segs.loc[i, "centroid_lon"],
                            segs.loc[j, "centroid_lat"], segs.loc[j, "centroid_lon"])
            if d <= GRAPH_EDGE_DIST_M:
                edges.append({
                    "seg_a": int(segs.loc[i, "segment_id"]),
                    "seg_b": int(segs.loc[j, "segment_id"]),
                    "dist_m": round(d, 2),
                })

    edge_df = pd.DataFrame(edges)
    print(f"  Edges formed: {len(edge_df)}")

    # Smooth severity using neighbour average
    if not edge_df.empty:
        sev_map  = dict(zip(segs["segment_id"], segs["weighted_severity"]))
        smoothed = {}
        for sid in segs["segment_id"]:
            neighbours = list(
                edge_df[edge_df["seg_a"] == sid]["seg_b"]
            ) + list(edge_df[edge_df["seg_b"] == sid]["seg_a"])
            all_vals = [sev_map[sid]] + [sev_map[n] for n in neighbours if n in sev_map]
            smoothed[sid] = round(float(sum(all_vals) / len(all_vals)), 6)
        seg_df = seg_df.copy()
        seg_df["smoothed_severity"] = seg_df["segment_id"].map(smoothed)
    else:
        seg_df = seg_df.copy()
        seg_df["smoothed_severity"] = seg_df["weighted_severity"]

    return edge_df, seg_df


# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------

def build_heatmap(frame_df, out_path):
    print("\nBuilding severity heatmap...")
    geo = frame_df.dropna(subset=["lat", "lon"]).copy()
    if geo.empty:
        print("  No GPS data — skipping heatmap.")
        return

    center = [geo["lat"].mean(), geo["lon"].mean()]
    m      = folium.Map(location=center, zoom_start=16,
                        tiles="CartoDB positron")

    heat_data = [
        [row["lat"], row["lon"], row["severity_score"]]
        for _, row in geo.iterrows()
        if row["severity_score"] > 0
    ]

    if heat_data:
        HeatMap(heat_data, radius=12, blur=10, max_zoom=18).add_to(m)

    m.save(out_path)
    print(f"  Saved -> {out_path}")


# ------------------------------------------------------------
# PRIORITY MAP
# ------------------------------------------------------------

ACTION_COLOUR = {"urgent": "red", "schedule": "orange", "monitor": "green"}


def build_priority_map(seg_df, out_path):
    print("\nBuilding maintenance priority map...")

    if os.path.exists(DECISION_CSV):
        dec = pd.read_csv(DECISION_CSV)[
            ["segment_id", "maintenance_action", "priority_score", "priority_rank"]
        ]
        seg = pd.merge(seg_df, dec, on="segment_id", how="left")
    else:
        seg = seg_df.copy()
        seg["maintenance_action"] = "monitor"
        seg["priority_score"]     = seg["weighted_severity"]

    seg = seg.dropna(subset=["centroid_lat", "centroid_lon"])
    if seg.empty:
        print("  No data for priority map.")
        return

    center = [seg["centroid_lat"].mean(), seg["centroid_lon"].mean()]
    m      = folium.Map(location=center, zoom_start=16, tiles="CartoDB positron")

    for _, row in seg.iterrows():
        action = row.get("maintenance_action", "monitor")
        colour = ACTION_COLOUR.get(action, "gray")
        sev    = row.get("smoothed_severity", row["weighted_severity"])
        radius = 6 + sev * 14

        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=radius,
            color=colour,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Segment {int(row['segment_id'])}</b><br>"
                f"Severity: {sev:.3f}<br>"
                f"Uncertainty: {row['weighted_uncertainty']:.3f}<br>"
                f"Action: <b>{action}</b><br>"
                f"Class: {row.get('dominant_class', 'N/A')}",
                max_width=200
            ),
        ).add_to(m)

    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                font-family:sans-serif;font-size:12px;
                box-shadow:2px 2px 6px rgba(0,0,0,.3)">
      <b>Maintenance action</b><br>
      <span style="color:red">●</span> Urgent &nbsp;
      <span style="color:orange">●</span> Schedule &nbsp;
      <span style="color:green">●</span> Monitor<br>
      <i style="color:#888">(circle size = severity)</i>
    </div>
    """))

    m.save(out_path)
    print(f"  Saved -> {out_path}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("=" * 55)
    print(f"URSA-Net — GPS Visualization + Spatial Graph  ({cfg.video})")
    print("=" * 55)

    frame_df, seg_df = load_data()

    edge_df, seg_df = build_spatial_graph(seg_df)
    edge_df.to_csv(GRAPH_CSV, index=False)
    print(f"  Graph edges saved -> {GRAPH_CSV}")

    build_heatmap(frame_df, os.path.join(OUTPUT_DIR, "map_severity_heatmap.html"))
    build_priority_map(seg_df, os.path.join(OUTPUT_DIR, "map_priority_markers.html"))

    print("\n" + "=" * 55)
    print("Done. Open the .html files in a browser.")
    print("=" * 55)


if __name__ == "__main__":
    main()