import os
import pandas as pd

OUTPUTS_DIR = "outputs"
rows = []

for video in sorted(os.listdir(OUTPUTS_DIR)):
    out = os.path.join(OUTPUTS_DIR, video)
    if not os.path.isdir(out):
        continue

    row = {"video": video}

    # Reliability
    rel_path = os.path.join(out, "frame_reliability.csv")
    if os.path.exists(rel_path):
        rel = pd.read_csv(rel_path)
        row["total_frames"] = len(rel)
        row["mean_reliability"] = round(rel["reliability"].mean(), 4)
        row["retention_pct"] = round((rel["reliability"] >= 0.45).sum() / len(rel) * 100, 2)

    # Uncertainty
    unc_path = os.path.join(out, "uncertainty_predictions.csv")
    if os.path.exists(unc_path):
        unc = pd.read_csv(unc_path)
        row["mean_uncertainty"] = round(unc["uncertainty"].mean(), 6)
        row["total_detections"] = len(unc)

    # Severity
    sev_path = os.path.join(out, "severity_scores.csv")
    if os.path.exists(sev_path):
        sev = pd.read_csv(sev_path)
        row["mean_severity"] = round(sev["severity_score"].mean(), 6)
        row["high_risk_frames"] = int((sev["severity_score"] > 0.5).sum())

    # Decision
    dec_path = os.path.join(out, "decision_output.csv")
    if os.path.exists(dec_path):
        dec = pd.read_csv(dec_path)
        row["total_segments"] = len(dec)
        row["urgent"] = int((dec["maintenance_action"] == "urgent").sum())
        row["schedule"] = int((dec["maintenance_action"] == "schedule").sum())
        row["monitor"] = int((dec["maintenance_action"] == "monitor").sum())
        row["mean_priority_score"] = round(dec["priority_score"].mean(), 6)
        row["reinspection_flags"] = int(dec["reinspection_flag"].sum())

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("per_video_metrics.csv", index=False)
print(f"Saved {len(df)} videos → per_video_metrics.csv")
print(df.to_string())