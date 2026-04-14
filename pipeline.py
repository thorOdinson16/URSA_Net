# --------------------------------------------
# PIPELINE CONTROL SCRIPT — MULTI-VIDEO
# Option B: per-video output folder
#   outputs/1IronShop/
#   outputs/2KaryaSiddhi/
#   ...
# --------------------------------------------

import os
import subprocess
import sys
import time

# --------------------------------------------
# CONFIG
# --------------------------------------------

GPS_DIR     = "GPS"       # contains <stem>.mp4 + <stem>.gpx pairs
OUTPUTS_DIR = "outputs"   # per-video output root


# --------------------------------------------
# DISCOVER VIDEOS
# --------------------------------------------

def discover_videos(gps_dir: str) -> list[str]:
    """Return sorted list of video stems that have both .mp4 and .gpx in GPS_DIR."""
    if not os.path.isdir(gps_dir):
        raise FileNotFoundError(f"GPS directory not found: '{gps_dir}'")

    stems = []
    for fname in os.listdir(gps_dir):
        if fname.endswith(".mp4"):
            stem = fname[:-4]
            if os.path.exists(os.path.join(gps_dir, stem + ".gpx")):
                stems.append(stem)
            else:
                print(f"[warn] No .gpx found for '{stem}' — skipping")

    return sorted(stems)


# --------------------------------------------
# STEP RUNNER
# --------------------------------------------

def run_step(script_name: str, step_name: str, env: dict):
    """Run a pipeline script, streaming its output live. Exits on failure."""

    print("\n===================================")
    print(f"STARTING: {step_name}")
    print("===================================\n")

    step_start = time.time()

    merged_env = {**os.environ, **env}

    process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"\nERROR in {script_name} (exit {process.returncode}). Pipeline stopped.")
        sys.exit(1)

    elapsed = time.time() - step_start
    print(f"\nCompleted: {step_name}  ({elapsed:.1f}s)\n")


# --------------------------------------------
# PER-VIDEO PIPELINE
# --------------------------------------------

def run_video(stem: str):
    out_dir = os.path.join(OUTPUTS_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)

    # Environment variables every script reads to know:
    #   - which video to process
    #   - where to write its outputs
    env = {
        "URSA_VIDEO":      stem,
        "URSA_GPS_DIR":    GPS_DIR,
        "URSA_OUTPUT_DIR": out_dir,
    }

    print("\n" + "=" * 55)
    print(f"PROCESSING VIDEO: {stem}")
    print(f"Output dir:       {out_dir}")
    print("=" * 55)

    video_start = time.time()

    # -- Core pipeline steps (order matters) --
    run_step("extract_frames.py",         "Frame Extraction",                   env)
    run_step("frame_reliability.py",      "Frame Reliability Calculation",      env)
    run_step("reliability_filter.py",     "Reliability Filtering",              env)
    run_step("uncertainty_estimation.py", "Uncertainty Estimation + Detection", env)
    run_step("severity_module.py",        "Severity Estimation",                env)
    run_step("gps_merge.py",              "GPS Merge",                          env)
    run_step("temporal_aggregation.py",   "Temporal Aggregation",               env)
    run_step("decision_layer.py",         "Decision Layer",                     env)
    run_step("spatial_graph.py",          "Spatial Graph",                      env)
    run_step("annotate_video.py",         "Annotating Video with Detections",   env)
    run_step("rebuild_video.py",          "Rebuilding Video",                   env)

    # -- Diagnostics / plots (non-critical) --
    run_step("reliability_analysis.py",   "Reliability Analysis",               env)
    run_step("severity_analysis.py",      "Severity Visualization",             env)
    run_step("uncertainty_analysis.py",   "Uncertainty Analysis",               env)

    elapsed = time.time() - video_start
    print(f"\n>>> Video '{stem}' complete in {elapsed:.1f}s")
    print(f">>> Outputs in: {out_dir}\n")


# --------------------------------------------
# MAIN
# --------------------------------------------

def main():
    # gps_sync.py is run once up-front for all videos together,
    # producing a single geotagged_frames.csv that covers all 32.
    run_step("gps_sync.py", "GPS Sync (all videos)", env={})

    stems = discover_videos(GPS_DIR)

    if not stems:
        print(f"No video+GPX pairs found in '{GPS_DIR}'. Nothing to do.")
        sys.exit(1)

    print(f"\nFound {len(stems)} videos: {stems}\n")

    pipeline_start = time.time()

    for i, stem in enumerate(stems, 1):
        print(f"\n{'='*55}")
        print(f"VIDEO {i}/{len(stems)}: {stem}")
        print(f"{'='*55}")
        run_video(stem)

    total = time.time() - pipeline_start
    print("\n===================================")
    print("ALL VIDEOS COMPLETE")
    print("===================================")
    print(f"Total runtime: {total:.1f}s  ({total/60:.1f} min)")
    print(f"Outputs root:  {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()