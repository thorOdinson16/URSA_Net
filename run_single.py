"""
run_single.py  —  Run the URSA-Net pipeline on a single video.

Usage:
    python run_single.py <stem>
    python run_single.py myroad          # expects GPS/myroad.mp4 + GPS/myroad.gpx

Optional overrides (env vars):
    URSA_GPS_DIR     default: GPS
    URSA_OUTPUTS_DIR default: outputs
"""

import os
import sys
import subprocess
import time


# ── Config ────────────────────────────────────────────────────────────────────

GPS_DIR     = os.environ.get("URSA_GPS_DIR",     "GPS")
OUTPUTS_DIR = os.environ.get("URSA_OUTPUTS_DIR", "outputs")


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_step(script_name: str, step_name: str, env: dict):
    """Run one pipeline script, stream its output, exit on failure."""
    print(f"\n{'='*50}")
    print(f"STARTING: {step_name}")
    print(f"{'='*50}\n")

    t0 = time.time()
    process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, **env},
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()

    if process.returncode != 0:
        print(f"\nERROR in {script_name} (exit {process.returncode}). Stopping.")
        sys.exit(1)

    print(f"\nDone: {step_name}  ({time.time() - t0:.1f}s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    stem = sys.argv[1].removesuffix(".mp4")  # accept stem or full filename

    mp4 = os.path.join(GPS_DIR, stem + ".mp4")
    gpx = os.path.join(GPS_DIR, stem + ".gpx")

    if not os.path.exists(mp4):
        print(f"ERROR: video not found: {mp4}")
        sys.exit(1)
    if not os.path.exists(gpx):
        print(f"ERROR: GPX not found: {gpx}")
        sys.exit(1)

    out_dir = os.path.join(OUTPUTS_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)

    env = {
        "URSA_VIDEO":      stem,
        "URSA_GPS_DIR":    GPS_DIR,
        "URSA_OUTPUT_DIR": out_dir,
    }

    print(f"\nProcessing : {stem}")
    print(f"Video      : {mp4}")
    print(f"GPX        : {gpx}")
    print(f"Output dir : {out_dir}\n")

    t_total = time.time()

    # gps_sync.py normally runs once across all videos; here we scope it to
    # just this stem by passing URSA_VIDEO so it only syncs the one file.
    run_step("gps_sync.py",              "GPS Sync",                           env)

    # Core pipeline
    run_step("extract_frames.py",        "Frame Extraction",                   env)
    run_step("frame_reliability.py",     "Frame Reliability Calculation",      env)
    run_step("reliability_filter.py",    "Reliability Filtering",              env)
    run_step("uncertainty_estimation.py","Uncertainty Estimation + Detection", env)
    run_step("severity_module.py",       "Severity Estimation",                env)
    run_step("gps_merge.py",             "GPS Merge",                          env)
    run_step("temporal_aggregation.py",  "Temporal Aggregation",               env)
    run_step("decision_layer.py",        "Decision Layer",                     env)
    run_step("spatial_graph.py",         "Spatial Graph",                      env)
    run_step("annotate_video.py",        "Annotate Video",                     env)
    run_step("rebuild_video.py",         "Rebuild Video",                      env)

    # Diagnostics / plots
    run_step("reliability_analysis.py",  "Reliability Analysis",               env)
    run_step("severity_analysis.py",     "Severity Visualization",             env)
    run_step("uncertainty_analysis.py",  "Uncertainty Analysis",               env)

    print(f"\n{'='*50}")
    print(f"DONE — {stem}  ({time.time() - t_total:.1f}s)")
    print(f"Outputs: {out_dir}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()