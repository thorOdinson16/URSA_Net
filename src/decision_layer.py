import os
import numpy as np
import pandas as pd

from video_config import cfg

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SEGMENT_CSV  = cfg.out("segment_severity.csv")
OUTPUT_CSV   = cfg.out("decision_output.csv")
SUMMARY_TXT  = cfg.out("decision_summary.txt")

W_SEVERITY      = 0.70
W_UNCERTAINTY   = 0.15
W_REINSPECTION  = 0.15

PRIORITY_URGENT   = 0.60
PRIORITY_SCHEDULE = 0.30


# ------------------------------------------------------------
# PRIORITY SCORE
# ------------------------------------------------------------

def compute_priority(severity: float, uncertainty: float, reinspection: int) -> float:
    score = (W_SEVERITY     * severity     +
             W_UNCERTAINTY  * uncertainty  +
             W_REINSPECTION * reinspection)
    return round(float(np.clip(score, 0.0, 1.0)), 6)


def maintenance_action(priority: float) -> str:
    if priority >= PRIORITY_URGENT:
        return "urgent"
    if priority >= PRIORITY_SCHEDULE:
        return "schedule"
    return "monitor"


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print(f"URSA-Net - Decision Layer  ({cfg.video})")
    print("=" * 55)

    if not os.path.exists(SEGMENT_CSV):
        raise FileNotFoundError(f"Missing: {SEGMENT_CSV}")

    df = pd.read_csv(SEGMENT_CSV)
    print(f"\nSegments loaded: {len(df)}")

    df["priority_score"] = df.apply(
        lambda r: compute_priority(r["weighted_severity"],
                                   r["weighted_uncertainty"],
                                   r["reinspection_flag"]),
        axis=1
    )

    df["maintenance_action"] = df["priority_score"].apply(maintenance_action)
    df["priority_rank"]      = df["priority_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("priority_rank").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")

    urgent   = df[df["maintenance_action"] == "urgent"]
    schedule = df[df["maintenance_action"] == "schedule"]
    monitor  = df[df["maintenance_action"] == "monitor"]

    lines = []
    lines.append("=" * 55)
    lines.append(f"URSA-Net - DECISION SUMMARY  ({cfg.video})")
    lines.append("=" * 55)
    lines.append(f"  Total segments assessed : {len(df)}")
    lines.append(f"  Urgent maintenance      : {len(urgent)}")
    lines.append(f"  Scheduled maintenance   : {len(schedule)}")
    lines.append(f"  Monitor only            : {len(monitor)}")
    lines.append(f"  Reinspection flags      : {df['reinspection_flag'].sum()}")
    lines.append("")
    lines.append(f"  Mean priority score     : {df['priority_score'].mean():.4f}")
    lines.append(f"  Max priority score      : {df['priority_score'].max():.4f}")
    lines.append("")
    lines.append("-" * 55)
    lines.append("TOP 10 PRIORITY SEGMENTS")
    lines.append("-" * 55)

    top10 = df.head(10)[["priority_rank", "segment_id", "first_frame",
                          "weighted_severity", "priority_score",
                          "maintenance_action", "dominant_class"]]

    for _, row in top10.iterrows():
        lines.append(
            f"  Rank {int(row['priority_rank']):>3} | "
            f"Seg {int(row['segment_id']):>4} | "
            f"Severity {row['weighted_severity']:.4f} | "
            f"Priority {row['priority_score']:.4f} | "
            f"{row['maintenance_action']:<8} | "
            f"{row['dominant_class']}"
        )

    lines.append("")
    lines.append("-" * 55)
    lines.append("URGENT SEGMENTS")
    lines.append("-" * 55)

    if len(urgent) == 0:
        lines.append("  None")
    else:
        for _, row in urgent.iterrows():
            lines.append(
                f"  Seg {int(row['segment_id']):>4} | "
                f"Frames {row['first_frame']} -> {row['last_frame']} | "
                f"Priority {row['priority_score']:.4f} | "
                f"{row['dominant_class']}"
            )

    lines.append("")
    lines.append("-" * 55)
    lines.append("REINSPECTION REQUIRED")
    lines.append("-" * 55)

    reinspect = df[df["reinspection_flag"] == 1]
    if len(reinspect) == 0:
        lines.append("  None")
    else:
        for _, row in reinspect.iterrows():
            lines.append(
                f"  Seg {int(row['segment_id']):>4} | "
                f"Severity {row['weighted_severity']:.4f} | "
                f"Uncertainty {row['weighted_uncertainty']:.4f} | "
                f"{row['dominant_class']}"
            )

    lines.append("")
    lines.append("=" * 55)
    lines.append("Decision layer complete.")
    lines.append("=" * 55)

    report = "\n".join(lines)
    print("\n" + report)

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSummary saved -> {SUMMARY_TXT}")