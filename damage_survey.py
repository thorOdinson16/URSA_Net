# ------------------------------------------------------------
# damage_survey.py — URSA-Net
# Bengaluru South damage survey: Pillar 2
# Outputs:
#   - Bar chart: decision distribution (Figure 3)
#   - Bar chart: dominant class distribution
#   - Reinspection flag rate summary
#   - All numbers for Table 2 in paper
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

INPUT_CSV  = "dataset_summary.csv"
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_COLORS = {
    "pothole":           "#E53935",
    "alligator_crack":   "#FB8C00",
    "longitudinal_crack":"#FDD835",
    "transverse_crack":  "#43A047",
}
DEFAULT_CLASS_COLOR = "#90A4AE"

DECISION_COLORS = {
    "urgent":   "#D32F2F",
    "schedule": "#F57C00",
    "monitor":  "#388E3C",
}

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

# ------------------------------------------------------------
# SUMMARY STATS (for paper Table 2)
# ------------------------------------------------------------

total_km       = df["track_length_km"].sum()
total_segments = df["total_segments"].sum()
total_urgent   = df["urgent"].sum()
total_schedule = df["schedule"].sum()
total_monitor  = df["monitor"].sum()
total_reinsp   = df["reinspection_flags"].sum()
reinsp_rate    = 100 * total_reinsp / total_segments

lines = []
lines.append("=" * 55)
lines.append("URSA-Net — Damage Survey Summary (Bengaluru South)")
lines.append("=" * 55)
lines.append(f"Roads surveyed          : {len(df)}")
lines.append(f"Total km surveyed       : {total_km:.2f} km")
lines.append(f"Total 15m segments      : {total_segments:,}")
lines.append(f"  Urgent                : {total_urgent:,}  ({100*total_urgent/total_segments:.1f}%)")
lines.append(f"  Schedule              : {total_schedule:,}  ({100*total_schedule/total_segments:.1f}%)")
lines.append(f"  Monitor               : {total_monitor:,}  ({100*total_monitor/total_segments:.1f}%)")
lines.append(f"Reinspection flags      : {total_reinsp:,}  ({reinsp_rate:.1f}% of segments)")
lines.append("")

# Per-class count
class_counts = df["dominant_class"].value_counts()
lines.append("Dominant class distribution:")
for cls, cnt in class_counts.items():
    lines.append(f"  {cls:<28}: {cnt} roads ({100*cnt/len(df):.1f}%)")

lines.append("")

# Top 5 worst roads by mean_priority_score
lines.append("Top 5 highest-priority roads:")
top5 = df.nlargest(5, "mean_priority_score")[
    ["road_name", "mean_priority_score", "schedule", "reinspection_flags", "dominant_class"]
]
for _, r in top5.iterrows():
    lines.append(
        f"  {r['road_name']:<28} priority={r['mean_priority_score']:.3f}  "
        f"schedule={r['schedule']}  reinsp={r['reinspection_flags']}  "
        f"class={r['dominant_class']}"
    )

lines.append("=" * 55)
summary_text = "\n".join(lines)
print(summary_text)

txt_path = os.path.join(OUTPUT_DIR, "damage_survey_summary.txt")
with open(txt_path, "w") as f:
    f.write(summary_text)
print(f"\nSummary saved → {txt_path}")

# ------------------------------------------------------------
# FIGURE 3a — Decision distribution stacked bar (per road)
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 5))

roads     = df["road_name"].tolist()
x         = np.arange(len(roads))

bar_sched = ax.bar(x, df["schedule"], color=DECISION_COLORS["schedule"],
                   label="Schedule", alpha=0.9)
bar_mon   = ax.bar(x, df["monitor"],  bottom=df["schedule"],
                   color=DECISION_COLORS["monitor"], label="Monitor", alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(roads, rotation=55, ha="right", fontsize=7)
ax.set_ylabel("Number of 15m Segments", fontsize=11)
ax.set_title("Road Damage Decision Distribution — Bengaluru South Survey",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# annotate reinspection flags
for i, (_, row) in enumerate(df.iterrows()):
    if row["reinspection_flags"] > 0:
        total = row["schedule"] + row["monitor"]
        ax.text(i, total + 0.4, f"⚑{int(row['reinspection_flags'])}",
                ha="center", fontsize=6, color="crimson")

plt.tight_layout()
fig3a_path = os.path.join(OUTPUT_DIR, "figure3a_decision_distribution.png")
fig.savefig(fig3a_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figure 3a saved → {fig3a_path}")

# ------------------------------------------------------------
# FIGURE 3b — Dominant class distribution (pie / bar)
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: pie chart
pie_labels = class_counts.index.tolist()
pie_values = class_counts.values.tolist()
pie_colors = [CLASS_COLORS.get(c, DEFAULT_CLASS_COLOR) for c in pie_labels]

axes[0].pie(pie_values, labels=pie_labels, colors=pie_colors,
            autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 9})
axes[0].set_title("Dominant Damage Class\n(% of roads)", fontsize=11, fontweight="bold")

# Right: mean_severity per dominant class
sev_by_class = df.groupby("dominant_class")["mean_severity"].mean().sort_values(ascending=False)
bar_colors   = [CLASS_COLORS.get(c, DEFAULT_CLASS_COLOR) for c in sev_by_class.index]

axes[1].bar(sev_by_class.index, sev_by_class.values, color=bar_colors, alpha=0.85)
axes[1].set_ylabel("Mean Severity Score", fontsize=10)
axes[1].set_title("Mean Severity by Dominant Class", fontsize=11, fontweight="bold")
axes[1].set_xticklabels(sev_by_class.index, rotation=20, ha="right", fontsize=9)
axes[1].grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
fig3b_path = os.path.join(OUTPUT_DIR, "figure3b_class_distribution.png")
fig.savefig(fig3b_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figure 3b saved → {fig3b_path}")

# ------------------------------------------------------------
# TABLE CSV for paper
# ------------------------------------------------------------

table_df = df[[
    "road_name", "track_length_km", "total_segments",
    "schedule", "monitor", "reinspection_flags",
    "mean_severity", "mean_priority_score", "dominant_class"
]].copy()
table_df.columns = [
    "Road", "Length (km)", "Segments",
    "Schedule", "Monitor", "Reinsp. Flags",
    "Mean Severity", "Mean Priority", "Dom. Class"
]
table_csv_path = os.path.join(OUTPUT_DIR, "table2_damage_survey.csv")
table_df.to_csv(table_csv_path, index=False)
print(f"Table 2 CSV saved → {table_csv_path}")

print("\nDamage survey analysis complete.")