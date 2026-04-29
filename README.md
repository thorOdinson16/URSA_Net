# URSA-Net: Uncertainty-weighted Road Surface Assessment Network

A lightweight, end-to-end computer vision pipeline for automated road damage detection, severity assessment, and maintenance prioritization from dashcam video and GPS. URSA-Net processes paired `.mp4` and `.gpx` files to produce per-segment maintenance decisions (urgent / schedule / monitor) across an entire road network with no per-frame annotation required.

Deployed for a damage survey of Bengaluru South covering **32 roads**, **17.71 km**, and **1,076 fifteen-metre road segments**.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Pipeline Stages](#pipeline-stages)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Analysis and Evaluation](#analysis-and-evaluation)
- [Damage Classes and Detection Performance](#damage-classes-and-detection-performance)
- [Performance Results](#performance-results)
- [Limitations](#limitations)
- [Utility Scripts](#utility-scripts)

---

## Overview

URSA-Net takes paired `.mp4` video and `.gpx` GPS track files as input and produces structured maintenance decisions for every 15-metre road segment. The system integrates:

- **YOLOv8n + ByteTrack** fine-tuned on RDD2020 + RDD2022 (50,513 training images, ~101,000 after augmentation), running at ~430 FPS (2.3 ms/image)
- **Intra-frame uncertainty estimation** via confidence-weighted score variance, designed for compatibility with YOLOv8n's minimal dropout architecture
- **GPS-synchronized spatial segmentation** using Haversine-interpolated cumulative distance, producing fixed 15-metre segments regardless of vehicle speed
- **Frame reliability filtering** using a composite score of blur, brightness, and motion coherence
- **A weighted decision layer** combining severity (70%), segment uncertainty (15%), and reinspection flag (15%) into priority scores

The pipeline runs fully automated over all videos in batch via `pipeline.py`, with per-video outputs written to `outputs/<video_stem>/`.

---

## System Architecture

```
GPS/
  <stem>.mp4   +   <stem>.gpx
        |                |
   [gps_sync.py]  ←  parses GPX, Haversine interpolation per frame
        |
   [extract_frames.py]       →  raw JPEG frames  (101,633 total)
        |
   [frame_reliability.py]    →  R = 0.5·B + 0.3·Br + 0.2·M  per frame
        |
   [reliability_filter.py]   →  keeps frames with R >= 0.35
        |
   [uncertainty_estimation.py]  →  YOLOv8n + TTA  →  detections + U per frame
        |
   [severity_module.py]      →  Sk = 0.45·Abbox + 0.35·Lcrack + 0.20·Ddensity
        |
   [gps_merge.py]            →  joins severity with GPS on (frame, video) key
        |
   [temporal_aggregation.py] →  15 m GPS segments, reliability+uncertainty-weighted mean
        |
   [decision_layer.py]       →  P = 0.70·S + 0.15·Useg + 0.15·F
                                 P >= 0.60  →  urgent
                                 0.30 <= P < 0.60  →  schedule
                                 P < 0.30  →  monitor
        |
   [spatial_graph.py]        →  Folium heatmap + priority marker map
        |
   [annotate_video.py]       →  ByteTrack annotations on filtered frames
        |
   [rebuild_video.py]        →  output MP4 (damage frames slowed to 10 FPS)
        |
   outputs/<stem>/
     severity_scores_geo.csv
     segment_severity.csv
     decision_output.csv
     decision_summary.txt
     plots/
     map_severity_heatmap.html
     map_priority_markers.html
     output_video.mp4
```

---

## Pipeline Stages

### Stage 1 — GPS Synchronization (`gps_sync.py`)
Runs once before per-video processing. Parses all `.gpx` files, enriches each trackpoint with speed, heading, and cumulative Haversine distance, then linearly interpolates GPS coordinates for every video frame. Produces `geotagged_frames.csv` covering all video–GPS pairs. Video start time is taken from the first GPX trackpoint timestamp, assuming simultaneous GPS logger and camera initialization.

### Stage 2 — Frame Extraction (`extract_frames.py`)
Decodes the source `.mp4` at 30 FPS and writes individual JPEG frames to `outputs/<stem>/frames/`. The Bengaluru South survey yielded **101,633 frames** across 32 videos.

### Stage 3 — Frame Reliability Filtering (`frame_reliability.py`, `reliability_filter.py`)
Computes a per-frame reliability score R ∈ [0, 1]:

```
R = 0.5 · B + 0.3 · Br + 0.2 · M
```

where B is Laplacian variance (sharpness), Br penalises extreme illumination (mean pixel < 30 or > 240), and M is block-mean standard deviation over a 4×4 grid (motion coherence). Frames with R < 0.35 are excluded. In the Bengaluru South deployment, **all 101,633 frames passed** (100% retention), confirming the filter is non-constraining under normal daytime driving conditions.

### Stage 4 — Uncertainty Estimation (`uncertainty_estimation.py`)
Runs YOLOv8n inference with Test-Time Augmentation (4 passes, `augment=True`) on filtered frames. Per-frame uncertainty U captures the spread of confidence-weighted detection scores:

```
U = std{ conf_i · w_i | i ∈ detections }
```

with class weights w_i reflecting repair urgency (pothole: 1.0, alligator crack: 0.9, transverse crack: 0.7, longitudinal crack: 0.6). Frames with ≤ 1 detection have U = 0. MCDropout was evaluated but found inapplicable to YOLOv8n due to negligible dropout variance (mean variance 2.7 × 10⁻⁶), confirming TTA as the appropriate method for this architecture.

### Stage 5 — Severity Estimation (`severity_module.py`)
Per-frame severity Sk combines three spatial indicators, each clipped to [0, 1]:

```
Sk = 0.45 · Abbox + 0.35 · Lcrack + 0.20 · Ddensity
```

where Abbox is normalised bounding box area, Lcrack is crack skeleton length via Canny edge detection with morphological thinning, and Ddensity is detection density per frame.

### Stage 6 — GPS Merge (`gps_merge.py`)
Left-joins severity scores onto the geotagged frame table using `(frame, video)` as the composite key, producing `severity_scores_geo.csv`. Unmatched frames fall back to frame-window segmentation in the next stage.

### Stage 7 — Temporal Aggregation (`temporal_aggregation.py`)
Groups frames into 15-metre GPS segments using cumulative Haversine distance. Falls back to fixed 30-frame windows when GPS data is absent. Within each segment, per-frame scores are aggregated using reliability- and uncertainty-weighted averaging:

```
w_k = R_k · (1 - U_k) / Σ_j R_j · (1 - U_j)
weighted_severity = Σ w_k · S_k
```

Segment-level uncertainty U_seg is derived from inter-frame severity standard deviation σ_S:

```
U_seg = min(σ_S / 0.3, 1)
```

The reinspection flag is set when U_seg > 0.5 or mean detection count > 3.

### Stage 8 — Decision Layer (`decision_layer.py`)
Computes a segment priority score P ∈ [0, 1]:

```
P = 0.70 · S + 0.15 · U_seg + 0.15 · F
```

where S is weighted mean severity, U_seg is segment uncertainty, and F is the reinspection flag (0 or 1). Decision thresholds: P ≥ 0.60 → **urgent**; 0.30 ≤ P < 0.60 → **schedule**; P < 0.30 → **monitor**.

### Stage 9 — Spatial Graph and Maps (`spatial_graph.py`)
Builds a segment connectivity graph and generates two interactive Folium HTML maps: a GPS heatmap of raw severity scores, and a priority marker map with colour-coded maintenance actions (green = monitor, orange = schedule, red = urgent).

### Stage 10 — Video Annotation and Rebuild (`annotate_video.py`, `rebuild_video.py`)
Runs YOLOv8n with ByteTrack on filtered frames to produce annotated JPEG outputs. The final MP4 is assembled with variable frame rate: damage frames are held at 10 FPS (frame repeated) while clean frames play at 30 FPS, making damage sites visually prominent during review.

---

## Repository Structure

```
.
├── GPS/                          # Input: paired .mp4 + .gpx files
├── outputs/                      # Per-video results (1..32)
│   └── <video_id>/
│       ├── decision_summary.txt
│       ├── severity_scores.csv
│       ├── segment_severity.csv
│       ├── decision_output.csv
│       ├── uncertainty_predictions.csv
│       ├── frame_reliability.csv
│       ├── severity_scores_geo.csv
│       ├── output_video.mp4
│       └── plots/
├── analysis_outputs/             # Cross-video analysis figures and reports
│   ├── classification_report.txt
│   ├── damage_survey_summary.txt
│   ├── figure2_lighting_analysis.png
│   ├── figure3a_decision_distribution.png
│   ├── figure3b_class_distribution.png
│   └── figure5_uncertainty_coherence.png
├── r20_dataset/                  # RDD2020 training data (gitignored)
├── r22_dataset/                  # RDD2022 training data (gitignored)
├── runs/                         # YOLO training runs (gitignored)
│
├── pipeline.py                   # Main entry point: runs all videos end-to-end
├── video_config.py               # Shared path resolution via environment variables
│
├── gps_sync.py                   # GPS synchronization (run once, all videos)
├── extract_frames.py             # Frame extraction
├── frame_reliability.py          # Frame quality scoring
├── reliability_filter.py         # Frame filtering
├── uncertainty_estimation.py     # TTA-based uncertainty + detection
├── uncertainty_estimation_tta.py # TTA variant
├── uncertainty_estimation_mcd.py # MCDropout variant (baseline evaluation only)
├── severity_module.py            # Per-frame severity scoring
├── gps_merge.py                  # GPS-severity join
├── temporal_aggregation.py       # Segment aggregation
├── decision_layer.py             # Maintenance decision logic
├── spatial_graph.py              # Maps and spatial graph
├── annotate_video.py             # YOLO ByteTrack annotation
├── rebuild_video.py              # Final MP4 reconstruction
│
├── augment_dataset.py            # Training data augmentation (Albumentations)
├── val_split.py                  # Validation split generation
├── generate_empty_labels.py      # Placeholder label generation
├── xmltoyolo.py                  # Pascal VOC XML to YOLO label conversion
├── dataset.yaml                  # YOLO dataset configuration
│
├── baseline_comparision.py       # Fixed-window baseline vs URSA-Net comparison
├── groundtruth_severity_validation.py  # Human consensus validation
├── damage_survey.py              # Cross-road survey analysis and paper figures
├── dataset_summary.py            # Per-video metrics aggregation to CSV
├── per_video_analytics.py        # Per-video CSV aggregator
├── lighting_analysis.py          # Daytime vs low-light Mann-Whitney analysis
├── uncertainty_coherence.py      # Uncertainty vs severity Pearson correlation
├── reliability_analysis.py       # Per-video reliability plots
├── severity_analysis.py          # Per-video severity plots
├── uncertainty_analysis.py       # Per-video uncertainty plots
├── map.py                        # Standalone map renderer
│
├── real_time_inference.py        # Live webcam or video inference
├── inference_standalone.py       # Standalone single-video inference
├── run_overnight.py              # Unattended batch runner
│
├── yolov8n.pt                    # YOLOv8n base weights
├── yolo26n.pt                    # Fine-tuned weights
└── dataset_summary.csv           # Aggregated per-road metrics (32 roads)
```

---

## Prerequisites

- Python 3.10 or later
- CUDA-capable GPU (recommended; inference uses `device=0` throughout)
- FFmpeg (required by OpenCV for video I/O)

Core Python dependencies:

```
ultralytics
opencv-python
pandas
numpy
scipy
scikit-learn
matplotlib
albumentations
folium
psutil
tqdm
```

---

## Installation

```bash
git clone <repository-url>
cd ursa-net

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install ultralytics opencv-python pandas numpy scipy scikit-learn \
            matplotlib albumentations folium psutil tqdm
```

Place trained model weights at:

```
runs/detect/augmented_model/weights/best.pt
```

---

## Data Preparation

### Training Datasets

The model is trained on a merged dataset from RDD2020 and RDD2022:

- **RDD2020**: 26,336 images with 31,000+ annotations across Japan, India, and Czech Republic. Training split used: 17,885 images.
- **RDD2022**: 47,420 images across six countries. Training split used: 32,628 images.
- Combined training set: **50,513 images**, approximately doubled to **~101,000 images** after augmentation.

XML annotations were converted to YOLO format with unified four-class mapping.

### Dataset Augmentation

```bash
python augment_dataset.py
```

Applies an Albumentations pipeline — motion/Gaussian blur, brightness/contrast, CLAHE, Gaussian noise, perspective warp, rotation ±15°, synthetic rain, RGB shift, adaptive sharpening — producing one augmented variant per image. Reads from `r20_dataset/` and `r22_dataset/`, writes to `augmented_output/`.

### Label Preparation

Generate empty placeholder labels for unannotated images:

```bash
python generate_empty_labels.py
```

Convert Pascal VOC XML annotations to YOLO format:

```bash
python xmltoyolo.py
```

### Dataset Configuration

Edit `dataset.yaml` before training:

```yaml
train: augmented_output/images
val:   r22_dataset/images/val
nc:    4
names: [longitudinal_crack, transverse_crack, alligator_crack, pothole]
```

### Model Training

```bash
yolo detect train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

Baseline (pre-augmentation): mAP50 = 0.606, mAP50-95 = 0.313. Augmented model: mAP50 = 0.607, mAP50-95 = 0.316. Inference speed: **2.3 ms/image (~430 FPS)**.

---

## Data Preparation (Survey Videos)

Place paired `.mp4` and `.gpx` files in the `GPS/` directory. File stems must match exactly:

```
GPS/
  1IronShop.mp4
  1IronShop.gpx
  2KaryaSiddhi.mp4
  2KaryaSiddhi.gpx
  ...
```

The GPS logger and camera must be started simultaneously. The first GPX trackpoint timestamp is used as the video start time.

---

## Running the Pipeline

### Full batch run (all videos)

```bash
python pipeline.py
```

Runs `gps_sync.py` once to produce `geotagged_frames.csv`, then processes each video–GPS pair discovered in `GPS/` sequentially.

### Single video (manual stage-by-stage)

```bash
export URSA_VIDEO=1IronShop
export URSA_GPS_DIR=GPS
export URSA_OUTPUT_DIR=outputs/1IronShop

python extract_frames.py
python frame_reliability.py
python reliability_filter.py
python uncertainty_estimation.py
python severity_module.py
python gps_merge.py
python temporal_aggregation.py
python decision_layer.py
python spatial_graph.py
python annotate_video.py
python rebuild_video.py
```

### Unattended overnight run

```bash
python run_overnight.py
```

### Real-time inference

```bash
python real_time_inference.py
```

Edit `video_path` inside the script to point to a file, or set it to `0` for webcam input.

---

## Configuration

All per-video path resolution is handled by `video_config.py`, reading three environment variables:

| Variable | Purpose | Required |
|---|---|---|
| `URSA_VIDEO` | Video stem (filename without extension) | Yes |
| `URSA_GPS_DIR` | Directory containing `.mp4` and `.gpx` files | Yes |
| `URSA_OUTPUT_DIR` | Per-video output directory | Yes |

Key tunable constants across pipeline scripts:

| Script | Parameter | Default | Description |
|---|---|---|---|
| `frame_reliability.py` | `THRESHOLD` | 0.45 | Minimum reliability score computed; used for reporting |
| `reliability_filter.py` | `THRESHOLD` | 0.35 | Threshold applied when copying filtered frames |
| `uncertainty_estimation.py` | `TTA_RUNS` | 4 | Number of TTA forward passes per batch |
| `uncertainty_estimation.py` | `CONF_THRESHOLD` | 0.25 | YOLO minimum detection confidence |
| `uncertainty_estimation.py` | `BATCH_SIZE` | 32 | Inference batch size |
| `temporal_aggregation.py` | `SEGMENT_METRES` | 15 | GPS segment length in metres |
| `temporal_aggregation.py` | `SEGMENT_SIZE` | 30 | Fallback frame-window size (no GPS) |
| `temporal_aggregation.py` | `SEV_LOW` | 0.20 | Low / medium severity boundary |
| `temporal_aggregation.py` | `SEV_HIGH` | 0.50 | Medium / high severity boundary |
| `temporal_aggregation.py` | `SEVERITY_STD_NORM` | 0.3 | Normalization denominator for U_seg |
| `rebuild_video.py` | `FPS_DAMAGE` | 10 | Playback rate for damage frames |

---

## Outputs

Each processed video produces the following under `outputs/<stem>/`:

| File | Description |
|---|---|
| `frame_reliability.csv` | Per-frame blur, brightness, motion, and composite reliability scores |
| `uncertainty_predictions.csv` | Per-detection class, confidence mean, TTA uncertainty, and bounding box |
| `severity_scores.csv` | Per-frame severity score, detection count, and class counts |
| `severity_scores_geo.csv` | Severity scores joined with GPS (lat, lon, ele, speed, heading) |
| `segment_severity.csv` | Per-segment aggregated severity, uncertainty, dominant class, centroid, reinspection flag |
| `decision_output.csv` | Per-segment priority score, maintenance action, and priority rank |
| `decision_summary.txt` | Human-readable ranked report with top-10 segments |
| `track_dir.txt` | Path to YOLO ByteTrack annotated frame directory |
| `output_video.mp4` | Annotated video with variable frame rate |
| `plots/` | Diagnostic plots: severity timeline, risk distribution, score histogram, class breakdown, severity vs uncertainty, score components, reliability curve, uncertainty histograms |
| `map_severity_heatmap.html` | Interactive Folium severity heatmap by GPS position |
| `map_priority_markers.html` | Interactive priority marker map colour-coded by maintenance action |

Cross-video analysis outputs written to `analysis_outputs/`:

| File | Description |
|---|---|
| `damage_survey_summary.txt` | Fleet-level summary: km surveyed, decision distribution, top-5 priority roads |
| `classification_report.txt` | Precision, recall, F1 vs human consensus |
| `figure2_lighting_analysis.png` | Daytime vs low-light boxplots with Mann-Whitney p-values |
| `figure3a_decision_distribution.png` | Per-road stacked bar of monitor / schedule segments |
| `figure3b_class_distribution.png` | Dominant damage class pie chart and mean severity by class |
| `figure5_uncertainty_coherence.png` | Per-road uncertainty vs severity scatter (r = 0.723) |
| `lighting_stats.csv` | Mann-Whitney U statistics for all metrics |
| `table2_damage_survey.csv` | Per-road survey table |

---

## Analysis and Evaluation

### Baseline comparison

Evaluates URSA-Net against a fixed-window baseline (30-frame windows, no GPS, no uncertainty) on 150 human-annotated segments from videos 7, 8, 13, 18, and 30. Threshold is auto-tuned to maximize accuracy (tuned value: 0.295):

```bash
python baseline_comparision.py
```

Outputs accuracy, Cohen's kappa, and per-class confusion for both methods. Saves `baseline_results.csv`.

### Human consensus validation

```bash
python groundtruth_severity_validation.py
```

Compares URSA-Net segment labels against majority-vote consensus from three independent annotators who viewed raw dashcam footage without pipeline outputs.

### Lighting condition analysis

```bash
python lighting_analysis.py
```

Groups videos by GPS timestamp into daytime (06:00–18:00 IST) and low-light (evening 18:00–20:00 + night after 20:00), then runs two-sided Mann-Whitney U tests on severity, uncertainty, and retention.

### Uncertainty coherence validation

```bash
python uncertainty_coherence.py
```

Computes Pearson r between per-road mean intra-frame uncertainty and mean severity score to validate the uncertainty estimator without bounding-box ground truth.

### Dataset and damage survey

```bash
python dataset_summary.py
python damage_survey.py
```

`dataset_summary.py` aggregates per-video outputs into `dataset_summary.csv`. `damage_survey.py` produces the fleet-level summary text, Figure 3 charts, and the paper Table 2 CSV.

---

## Damage Classes and Detection Performance

The model detects four road surface damage categories (RDD unified taxonomy). Class-wise performance on the augmented YOLOv8n model:

| ID | Class | RDD Code | Severity Weight | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| 0 | longitudinal\_crack | D00 | 0.6 | 0.619 | 0.508 | 0.554 | 0.297 |
| 1 | transverse\_crack | D10 | 0.7 | 0.607 | 0.543 | 0.566 | 0.273 |
| 2 | alligator\_crack | D20 | 0.9 | 0.724 | 0.688 | 0.755 | 0.439 |
| 3 | pothole | D40 | 1.0 | 0.650 | 0.479 | 0.554 | 0.257 |

**Overall:** mAP50 = 0.607, mAP50-95 = 0.316. Inference speed: **2.3 ms/image (~430 FPS)**.

---

## Performance Results

### Survey coverage — Bengaluru South

| Parameter | Value | Notes |
|---|---|---|
| Roads surveyed | 32 | Bengaluru South wards |
| Total road length | 17.71 km | Haversine GPS tracks |
| Total 15 m segments | 1,076 | After GPS aggregation |
| Total frames extracted | 101,633 | At 30 FPS |
| Total detections | 22,885 | YOLOv8n, conf ≥ 0.25 |
| Mean frame retention | 100% | Reliability filter τ = 0.35 |
| Daytime recordings | 22 | 06:00–18:00 IST |
| Evening recordings | 9 | 18:00–20:00 IST |
| Night recordings | 1 | After 20:00 IST |
| Urgent segments | 0 | P ≥ 0.60 |
| Scheduled maintenance | 125 (11.6%) | 0.30 ≤ P < 0.60 |
| Monitor only | 951 (88.4%) | P < 0.30 |
| Reinspection flagged | 148 (13.8%) | U_seg > 0.5 or mean detections > 3 |

### Pipeline vs. human consensus (150 annotated segments)

| Method | Accuracy | Cohen's κ | Monitor Recall | Schedule Recall |
|---|---|---|---|---|
| Fixed-window baseline | 48.6% | 0.108 | 30% | 84% |
| URSA-Net | **78%** | **0.470** | **91%** | **52%** |

URSA-Net outperforms the baseline by **29.4 percentage points**. Inter-annotator agreement: Fleiss' κ = 0.498 (moderate), confirming the inherent subjectivity of visual severity grading.

Confusion matrix — URSA-Net vs. human consensus:

|  | Predicted Low | Predicted Medium |
|---|---|---|
| Actual Low | 90 (91.8%) | 8 (8.2%) |
| Actual Medium | 25 (48.1%) | 27 (51.9%) |

High monitor recall (91%) reflects the system's conservative design: in a screening context, false negatives (missed damage) are more costly than false positives.

### Uncertainty coherence (ground-truth-free validation)

Pearson r between per-road mean intra-frame uncertainty and mean severity: **r = 0.723, p < 0.001, n = 32**. This confirms the uncertainty estimator captures genuine detection difficulty, and the correlation persists across both daytime and low-light conditions.

### Lighting condition analysis (Mann-Whitney U test)

| Metric | Daytime (n=22) | Low-light (n=10) | p-value | Significance |
|---|---|---|---|---|
| Frame Retention (%) | 100.0 | 100.0 | 1.0000 | ns |
| Mean Intra-frame Uncertainty | 0.0315 | 0.0280 | 0.2999 | ns |
| Mean Severity Score | 0.0601 | 0.0389 | **0.0077** | ** |

Severity is significantly higher under daytime conditions (p = 0.0077). Uncertainty does not differ significantly by lighting (p = 0.30), confirming intra-frame uncertainty reflects damage complexity rather than illumination. Severity estimates from low-light recordings should be treated as lower bounds.

### Top 5 highest-priority roads (Bengaluru South)

| Road | Priority Score | Scheduled Segments | Reinspection Flags | Dominant Class |
|---|---|---|---|---|
| Road 18 | 0.311 | 11 | 11 | pothole |
| Road 8 | 0.301 | 8 | 8 | alligator\_crack |
| Road 30 | 0.209 | 6 | 7 | alligator\_crack |
| Road 7 | 0.192 | 7 | 9 | pothole |
| Road 13 | 0.187 | 2 | 2 | alligator\_crack |

Dominant damage class across all surveyed roads: pothole (75%), alligator crack (21.9%), longitudinal crack (3.1%). Alligator-crack-dominant roads exhibit higher mean severity (0.075) than pothole-dominant roads (0.049), consistent with alligator cracking indicating more advanced structural failure.

---

## Limitations

- **No per-frame ground truth** on survey data. The uncertainty coherence result (r = 0.723, p < 0.001) provides indirect validation only.
- **Single-detection uncertainty** is zero by definition; frames with ≤ 1 detection contribute no uncertainty signal. MC Dropout would be better-calibrated but is incompatible with YOLOv8n's minimal dropout (mean variance 2.7 × 10⁻⁶).
- **GPS time synchronization** assumes simultaneous camera and GPS logger initialization. Timing offsets cause geolocation errors in frame-level GPS assignment.
- **Crack length measurement** uses a resolution-based diagonal estimate; a resolution-independent measure would improve cross-video consistency.
- **Low-light sample size** is limited (9 evening + 1 nighttime recording). A larger nighttime set is needed to isolate the lighting effect independently from road characteristics.
- **Class performance imbalance**: pothole recall (0.479) is notably lower than alligator crack recall (0.688), which may cause underestimation of severity on pothole-dominant roads.

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `val_split.py` | Creates a train/validation split from the augmented dataset |
| `generate_empty_labels.py` | Writes empty `.txt` label files for unannotated images |
| `xmltoyolo.py` | Converts Pascal VOC XML annotations to YOLO format |
| `per_video_analytics.py` | Aggregates per-video metrics into `per_video_metrics.csv` |
| `inference_standalone.py` | Runs detection on a single video without GPS pipeline |
| `real_time_inference.py` | Live inference with FPS overlay via OpenCV window |
| `run_overnight.py` | Unattended batch runner with logging |
| `map.py` | Standalone Folium map renderer from existing segment CSVs |