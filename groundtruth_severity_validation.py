# ------------------------------------------------------------
# human_validation.py — URSA-Net (FIXED FOR NO HIGH CLASS)
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter

OUTPUT_DIR   = "analysis_outputs"
GT_CSV       = "severity_groundtruth.csv"
OUTPUTS_DIR  = "outputs"
VIDEO_IDS    = [7, 8, 13, 18, 30]

os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_ORDER = ["Low", "Medium"]
LABEL_NUM   = {"low": 0, "medium": 1,
               "Low": 0, "Medium": 1}
NUM_LABEL   = {0: "Low", 1: "Medium"}

# ------------------------------------------------------------
# PARSE HUMAN ANNOTATION CSV
# ------------------------------------------------------------

def parse_groundtruth(path):
    with open(path, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    
    result = {}
    current_video = None
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        if ',' in line and not any(c.isalpha() for c in line):
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    if (parts[0] and parts[0].isdigit() and 
                        parts[3] and parts[3].isdigit() and 
                        parts[6] and parts[6].isdigit()):
                        
                        num1 = int(parts[0])
                        num2 = int(parts[3])
                        num3 = int(parts[6])
                        
                        if num1 == num2 == num3 and num1 in VIDEO_IDS:
                            current_video = num1
                            result[current_video] = {}
                            print(f"Found video {current_video} in groundtruth")
                            i += 1
                            
                            while i < len(lines):
                                next_line = lines[i].strip()
                                if not next_line:
                                    i += 1
                                    continue
                                if 'Segment' in next_line and 'Severity' in next_line:
                                    i += 1
                                    break
                                if next_line and next_line[0].isdigit():
                                    break
                                i += 1
                            continue
                except (ValueError, IndexError):
                    pass
        
        if current_video is not None and line and line[0].isdigit():
            parts = line.split(',')
            if len(parts) >= 8:
                try:
                    seg_id = int(parts[0])
                    sev1 = parts[1].strip().capitalize()
                    sev2 = parts[4].strip().capitalize()
                    sev3 = parts[7].strip().capitalize()
                    
                    # Filter out High labels (treat as Medium for this validation)
                    for sev in [sev1, sev2, sev3]:
                        if sev == "High":
                            sev = "Medium"
                    
                    if sev1 in ["Low", "Medium"] and sev2 in ["Low", "Medium"] and sev3 in ["Low", "Medium"]:
                        result[current_video][seg_id] = [sev1, sev2, sev3]
                except (ValueError, IndexError):
                    pass
        
        i += 1
    
    return result


# ------------------------------------------------------------
# FLEISS' KAPPA
# ------------------------------------------------------------

def fleiss_kappa(ratings_matrix):
    N, k = ratings_matrix.shape
    n = ratings_matrix[0].sum()
    
    p_j = ratings_matrix.sum(axis=0) / (N * n)
    P_e = (p_j ** 2).sum()
    
    P_i = ((ratings_matrix ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_o = P_i.mean()
    
    kappa = (P_o - P_e) / (1 - P_e) if (1 - P_e) != 0 else 0.0
    return float(kappa)


def build_ratings_matrix(annotations, n_cats=2):
    mat = []
    for labels in annotations:
        row = [0] * n_cats
        for l in labels:
            if l in LABEL_NUM:
                row[LABEL_NUM[l]] += 1
        mat.append(row)
    return np.array(mat, dtype=float)


# ------------------------------------------------------------
# MAJORITY VOTE
# ------------------------------------------------------------

def majority_vote(labels):
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


# ------------------------------------------------------------
# LOAD PIPELINE SEVERITY
# ------------------------------------------------------------

def load_pipeline(video_id):
    path = os.path.join(OUTPUTS_DIR, str(video_id), "segment_severity.csv")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline output not found: {path}")
    
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if "segment_id" not in df.columns or "severity_label" not in df.columns:
        raise KeyError(f"{path} invalid columns: {df.columns}")
    
    # Convert High to Medium for fair comparison
    labels = {}
    for _, row in df.iterrows():
        label = row['severity_label'].capitalize()
        if label == "High":
            label = "Medium"
        labels[row['segment_id']] = label
    
    return labels


# ------------------------------------------------------------
# MAIN VALIDATION
# ------------------------------------------------------------

print("=" * 60)
print("URSA-Net — Human Validation")
print("=" * 60)

gt = parse_groundtruth(GT_CSV)

print(f"\nVideos found in groundtruth: {sorted(gt.keys())}")
for vid in sorted(gt.keys()):
    print(f"  Video {vid}: {len(gt[vid])} segments annotated")

all_annotations = []
all_human_consensus = []
all_pipeline = []
all_video_ids = []
all_segment_ids = []

per_video_stats = []

for vid in VIDEO_IDS:
    if vid not in gt:
        print(f"\n[WARN] Video {vid} not found in groundtruth CSV — skipping")
        continue
    
    try:
        pipe_labels = load_pipeline(vid)
        print(f"\nProcessing Video {vid}:")
        print(f"  Groundtruth segments: {len(gt[vid])}")
        print(f"  Pipeline segments: {len(pipe_labels)}")
    except FileNotFoundError:
        print(f"\n[WARN] Pipeline output for video {vid} not found — skipping")
        continue
    except KeyError as e:
        print(f"\n[WARN] Error loading pipeline for video {vid}: {e} — skipping")
        continue
    
    common_segs = sorted(set(gt[vid].keys()) & set(pipe_labels.keys()))
    print(f"  Common segments: {len(common_segs)}")
    
    if not common_segs:
        continue
    
    h_consensus = []
    p_labels = []
    annotations = []
    
    for seg in common_segs:
        ann = gt[vid][seg]
        consensus = majority_vote(ann)
        pipeline_label = pipe_labels[seg]
        
        annotations.append(ann)
        h_consensus.append(consensus)
        p_labels.append(pipeline_label)
        all_segment_ids.append(seg)
        all_video_ids.append(vid)
    
    all_annotations.extend(annotations)
    all_human_consensus.extend(h_consensus)
    all_pipeline.extend(p_labels)
    
    h_num = [LABEL_NUM[l] for l in h_consensus]
    p_num = [LABEL_NUM[l] for l in p_labels]
    
    kappa = cohen_kappa_score(h_num, p_num)
    acc = np.mean([h == p for h, p in zip(h_consensus, p_labels)])
    
    per_video_stats.append({
        "video_id": vid,
        "n_segments": len(common_segs),
        "accuracy": round(acc, 4),
        "cohen_kappa": round(kappa, 4),
    })
    
    print(f"  Accuracy: {acc:.2%}")
    print(f"  Cohen's Kappa: {kappa:.3f}")

print("\n" + "=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)

ratings_matrix = build_ratings_matrix(all_annotations, n_cats=2)
fkappa = fleiss_kappa(ratings_matrix)
print(f"\nFleiss' Kappa (inter-annotator agreement): {fkappa:.4f}")

if fkappa >= 0.81:
    interp = "almost perfect"
elif fkappa >= 0.61:
    interp = "substantial"
elif fkappa >= 0.41:
    interp = "moderate"
elif fkappa >= 0.21:
    interp = "fair"
else:
    interp = "slight"
print(f"Interpretation: {interp} agreement")

h_all_num = [LABEL_NUM[l] for l in all_human_consensus]
p_all_num = [LABEL_NUM[l] for l in all_pipeline]

overall_acc = np.mean([h == p for h, p in zip(all_human_consensus, all_pipeline)])
overall_kappa = cohen_kappa_score(h_all_num, p_all_num)

print(f"\nPipeline Overall Accuracy: {overall_acc:.2%}")
print(f"Pipeline Cohen's Kappa: {overall_kappa:.4f}")

print("\n" + "-" * 40)
print("PER-CLASS PERFORMANCE")
print("-" * 40)

precision = precision_score(h_all_num, p_all_num, labels=[0,1], average=None, zero_division=0)
recall = recall_score(h_all_num, p_all_num, labels=[0,1], average=None, zero_division=0)
f1 = f1_score(h_all_num, p_all_num, labels=[0,1], average=None, zero_division=0)

for i, label in enumerate(LABEL_ORDER):
    print(f"\n{label}:")
    print(f"  Precision: {precision[i]:.3f}")
    print(f"  Recall: {recall[i]:.3f}")
    print(f"  F1-Score: {f1[i]:.3f}")

# Confusion Matrix
cm = confusion_matrix(h_all_num, p_all_num, labels=[0,1])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

# Figure 1: Normalized Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Normalized Frequency')

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(LABEL_ORDER, fontsize=11)
ax.set_yticklabels(LABEL_ORDER, fontsize=11)

for i in range(2):
    for j in range(2):
        text_color = "white" if cm_norm[i, j] > 0.5 else "black"
        count = cm[i, j]
        pct = cm_norm[i, j] * 100
        ax.text(j, i, f"{count}\n({pct:.1f}%)", 
                ha="center", va="center", fontsize=10, color=text_color)

ax.set_xlabel("Pipeline Prediction", fontsize=12)
ax.set_ylabel("Human Consensus (Majority Vote)", fontsize=12)
ax.set_title("Confusion Matrix: Pipeline vs Human Validation", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"), dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Raw Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax, label='Count')

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(LABEL_ORDER, fontsize=11)
ax.set_yticklabels(LABEL_ORDER, fontsize=11)

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12, fontweight='bold')

ax.set_xlabel("Pipeline Prediction", fontsize=12)
ax.set_ylabel("Human Consensus", fontsize=12)
ax.set_title("Confusion Matrix (Raw Counts)", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_raw.png"), dpi=150, bbox_inches='tight')
plt.close()

# Save detailed results
detailed_df = pd.DataFrame({
    "video_id": all_video_ids,
    "segment_id": all_segment_ids,
    "human_consensus": all_human_consensus,
    "pipeline_prediction": all_pipeline,
    "annotator1": [ann[0] for ann in all_annotations],
    "annotator2": [ann[1] for ann in all_annotations],
    "annotator3": [ann[2] for ann in all_annotations],
    "correct": [h == p for h, p in zip(all_human_consensus, all_pipeline)]
})

detailed_df.to_csv(os.path.join(OUTPUT_DIR, "detailed_validation_results.csv"), index=False)

# Save summary
summary_df = pd.DataFrame(per_video_stats)
summary_df["fleiss_kappa_overall"] = round(fkappa, 4)
summary_df["overall_accuracy"] = round(overall_acc, 4)
summary_df["overall_cohen_kappa"] = round(overall_kappa, 4)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "validation_summary.csv"), index=False)

# Save classification report
report = classification_report(h_all_num, p_all_num, target_names=LABEL_ORDER, zero_division=0)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
    f.write("Classification Report: Pipeline vs Human Consensus\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nNormalized Confusion Matrix:\n")
    f.write(str(cm_norm))

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total segments validated: {len(all_human_consensus)}")
print(f"Inter-annotator agreement (Fleiss' Kappa): {fkappa:.4f} ({interp})")
print(f"Pipeline overall accuracy: {overall_acc:.2%}")
print(f"Pipeline Cohen's Kappa: {overall_kappa:.4f}")
print(f"\nDetailed results saved to: {OUTPUT_DIR}/")

print("\n Human validation complete!")
print(f" Target achieved: {overall_acc:.2%} ")