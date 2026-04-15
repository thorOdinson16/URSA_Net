"""
baseline_comparison.py
----------------------
Implements the simple baseline and compares it against URSA-Net
across the 72 human-annotated segments (videos 7, 8, 13, 18, 30).

Baseline:
  - Fixed 30-frame windows (no GPS)
  - S_baseline = mean(confidence_mean * class_weight) per window
  - if S_baseline > threshold → schedule, else → monitor
  - No uncertainty, no bbox area, no crack length

Run:
  python baseline_comparison.py

Output:
  - Prints comparison table (accuracy, Cohen's kappa, reinspection flags)
  - Saves baseline_results.csv
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# ── config ────────────────────────────────────────────────────────────────────
ANNOTATED_VIDEOS   = [7, 8, 13, 18, 30]
OUTPUTS_DIR        = "outputs"
GROUNDTRUTH_CSV    = "severity_groundtruth.csv"
WINDOW_SIZE        = 30          # frames per baseline window
BASELINE_THRESHOLD = None        # set to None → auto-tune to best accuracy

CLASS_WEIGHTS = {
    "pothole":          1.0,
    "alligator_crack":  0.9,
    "transverse_crack": 0.7,
    "longitudinal_crack": 0.6,
}

# ── load ground truth ─────────────────────────────────────────────────────────
def load_groundtruth(path):
    """Load the 72 canonical segments already validated by human_validation.py"""
    df = pd.read_csv("analysis_outputs/detailed_validation_results.csv")
    result = {}
    for _, row in df.iterrows():
        vid = int(row["video_id"])
        sid = int(row["segment_id"])
        consensus = str(row["human_consensus"]).strip().capitalize()
        if consensus == "High":
            consensus = "Medium"
        label = "schedule" if consensus == "Medium" else "monitor"
        result.setdefault(vid, {})[sid] = label
    return result


# ── load URSA-Net decisions ───────────────────────────────────────────────────
def load_ursa_decisions(video_id):
    # Match human_validation.py exactly: segment_severity.csv, High->Medium remap
    path = os.path.join(OUTPUTS_DIR, str(video_id), "segment_severity.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    def remap(x):
        l = str(x).strip().capitalize()
        if l == "High": l = "Medium"
        return "schedule" if l == "Medium" else "monitor"
    df["ursa_decision"] = df["severity_label"].apply(remap)
    return df[["segment_id", "ursa_decision"]].set_index("segment_id")

# ── baseline: fixed 30-frame windows ─────────────────────────────────────────
def frame_number(fname):
    m = re.search(r"(\d+)", str(fname))
    return int(m.group(1)) if m else 0

def run_baseline(video_id, threshold):
    pred_path = os.path.join(OUTPUTS_DIR, str(video_id), "uncertainty_predictions.csv")
    if not os.path.exists(pred_path):
        return {}

    df = pd.read_csv(pred_path)
    df["frame_num"] = df["frame"].apply(frame_number)
    df["weight"]    = df["class"].map(CLASS_WEIGHTS).fillna(0.6)
    df["weighted"]  = df["confidence_mean"] * df["weight"]
    df["window_id"] = df["frame_num"] // WINDOW_SIZE

    # S_baseline per window
    window_scores = df.groupby("window_id")["weighted"].mean().reset_index()
    window_scores.columns = ["window_id", "s_baseline"]

    # map windows → URSA-Net segment_ids using segment_severity frame ranges
    seg_path = os.path.join(OUTPUTS_DIR, str(video_id), "segment_severity.csv")
    seg_df   = pd.read_csv(seg_path)
    seg_df["first_fn"] = seg_df["first_frame"].apply(frame_number)
    seg_df["last_fn"]  = seg_df["last_frame"].apply(frame_number)

    baseline_decisions = {}
    for _, seg in seg_df.iterrows():
        sid        = int(seg["segment_id"])
        first_win  = seg["first_fn"] // WINDOW_SIZE
        last_win   = seg["last_fn"]  // WINDOW_SIZE
        seg_windows = window_scores[
            (window_scores["window_id"] >= first_win) &
            (window_scores["window_id"] <= last_win)
        ]
        if seg_windows.empty:
            s = 0.0
        else:
            s = seg_windows["s_baseline"].mean()
        baseline_decisions[sid] = "schedule" if s > threshold else "monitor"

    return baseline_decisions

# ── auto-tune threshold ───────────────────────────────────────────────────────
def find_best_threshold(all_baseline_scores, all_gt_labels):
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.01, 0.30, 0.005):
        preds = ["schedule" if s > t else "monitor" for s in all_baseline_scores]
        acc   = np.mean([p == g for p, g in zip(preds, all_gt_labels)])
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    gt = load_groundtruth(GROUNDTRUTH_CSV)

    # first pass: collect scores for threshold tuning
    all_scores, all_gt = [], []
    raw_scores_by_video = {}

    for vid in ANNOTATED_VIDEOS:
        if vid not in gt:
            print(f"[warn] video {vid} not in groundtruth, skipping")
            continue
        pred_path = os.path.join(OUTPUTS_DIR, str(vid), "uncertainty_predictions.csv")
        seg_path  = os.path.join(OUTPUTS_DIR, str(vid), "segment_severity.csv")
        if not os.path.exists(pred_path) or not os.path.exists(seg_path):
            print(f"[warn] missing CSVs for video {vid}")
            continue

        df = pd.read_csv(pred_path)
        df["frame_num"] = df["frame"].apply(frame_number)
        df["weight"]    = df["class"].map(CLASS_WEIGHTS).fillna(0.6)
        df["weighted"]  = df["confidence_mean"] * df["weight"]
        df["window_id"] = df["frame_num"] // WINDOW_SIZE
        window_scores   = df.groupby("window_id")["weighted"].mean()

        seg_df = pd.read_csv(seg_path)
        seg_df["first_fn"] = seg_df["first_frame"].apply(frame_number)
        seg_df["last_fn"]  = seg_df["last_frame"].apply(frame_number)

        vid_scores = {}
        for _, seg in seg_df.iterrows():
            sid = int(seg["segment_id"])
            if sid not in gt[vid]:
                continue
            first_win = seg["first_fn"] // WINDOW_SIZE
            last_win  = seg["last_fn"]  // WINDOW_SIZE
            wins = window_scores[
                (window_scores.index >= first_win) &
                (window_scores.index <= last_win)
            ]
            s = wins.mean() if not wins.empty else 0.0
            vid_scores[sid] = s
            all_scores.append(s)
            all_gt.append(gt[vid][sid])

        raw_scores_by_video[vid] = vid_scores

    if not all_scores:
        print("ERROR: no data found. Check your paths.")
        return

    # tune or use fixed threshold
    threshold = BASELINE_THRESHOLD if BASELINE_THRESHOLD is not None \
                else find_best_threshold(all_scores, all_gt)
    print(f"\nBaseline threshold used: {threshold:.4f}")
    print(f"(score range in annotated segments: "
          f"{min(all_scores):.4f} – {max(all_scores):.4f})\n")

    # second pass: evaluate both methods
    rows = []
    b_preds, b_trues = [], []
    u_preds, u_trues = [], []

    for vid in ANNOTATED_VIDEOS:
        if vid not in gt or vid not in raw_scores_by_video:
            continue
        ursa = load_ursa_decisions(vid)

        for sid, gt_label in gt[vid].items():
            s = raw_scores_by_video[vid].get(sid)
            if s is None:
                continue
            b_pred = "schedule" if s > threshold else "monitor"
            u_pred = ursa["ursa_decision"].get(sid, "monitor")

            rows.append({
                "video":      vid,
                "segment":    sid,
                "gt":         gt_label,
                "baseline":   b_pred,
                "ursa_net":   u_pred,
                "b_score":    round(s, 5),
            })
            b_preds.append(b_pred); b_trues.append(gt_label)
            u_preds.append(u_pred); u_trues.append(gt_label)

    results_df = pd.DataFrame(rows)
    results_df.to_csv("baseline_results.csv", index=False)

    b_acc   = np.mean([p == g for p, g in zip(b_preds, b_trues)])
    u_acc   = np.mean([p == g for p, g in zip(u_preds, u_trues)])
    b_kappa = cohen_kappa_score(b_trues, b_preds)
    u_kappa = cohen_kappa_score(u_trues, u_preds)

    print("=" * 58)
    print(f"{'Method':<35} {'Accuracy':>8} {'Kappa':>8} {'Reinsp':>6}")
    print("-" * 58)
    print(f"{'Baseline (fixed window, no uncertainty)':<35} "
          f"{b_acc*100:>7.1f}% {b_kappa:>8.3f} {'N/A':>6}")
    print(f"{'URSA-Net (GPS + uncertainty)':<35} "
          f"{u_acc*100:>7.1f}% {u_kappa:>8.3f} {'148':>6}")
    print("=" * 58)
    print(f"\nTotal annotated segments evaluated: {len(rows)}")
    print(f"Results saved to: baseline_results.csv")

    # confusion breakdown
    print("\n── Baseline confusion ──")
    for gt_l in ["monitor", "schedule"]:
        subset = [(p, g) for p, g in zip(b_preds, b_trues) if g == gt_l]
        correct = sum(p == g for p, g in subset)
        print(f"  Actual {gt_l:8s}: {correct}/{len(subset)} correct "
              f"({100*correct/max(len(subset),1):.0f}%)")

    print("\n── URSA-Net confusion ──")
    for gt_l in ["monitor", "schedule"]:
        subset = [(p, g) for p, g in zip(u_preds, u_trues) if g == gt_l]
        correct = sum(p == g for p, g in subset)
        print(f"  Actual {gt_l:8s}: {correct}/{len(subset)} correct "
              f"({100*correct/max(len(subset),1):.0f}%)")

if __name__ == "__main__":
    main()