# ------------------------------------------------------------
# video_config.py — shared path resolver for URSA-Net scripts
#
# Every per-video script imports this module and calls:
#
#     from video_config import cfg
#
# Then uses cfg.out("somefile.csv") to get the correct
# per-video output path, and cfg.video / cfg.gps_dir /
# cfg.out_dir for other references.
#
# When run standalone (for testing a single video) you can
# set the env vars manually:
#
#     URSA_VIDEO=1IronShop URSA_OUTPUT_DIR=outputs/1IronShop \
#         python uncertainty_estimation.py
# ------------------------------------------------------------

import os


class _Config:
    def __init__(self):
        self.video   = os.environ.get("URSA_VIDEO", "")
        self.gps_dir = os.environ.get("URSA_GPS_DIR", "GPS")
        self.out_dir = os.environ.get("URSA_OUTPUT_DIR", ".")

        if not self.video:
            raise EnvironmentError(
                "URSA_VIDEO environment variable is not set.\n"
                "Run via pipeline.py, or set it manually:\n"
                "  URSA_VIDEO=1IronShop URSA_OUTPUT_DIR=outputs/1IronShop python <script>.py"
            )

        os.makedirs(self.out_dir, exist_ok=True)

    def out(self, filename: str) -> str:
        """Return full path for an output file in this video's output directory."""
        return os.path.join(self.out_dir, filename)

    def plots_dir(self) -> str:
        """Return (and create) the per-video plots subdirectory."""
        d = os.path.join(self.out_dir, "plots")
        os.makedirs(d, exist_ok=True)
        return d

    def frames_dir(self) -> str:
        """Temporary frames directory (inside out_dir, cleared between videos)."""
        return os.path.join(self.out_dir, "frames")

    def filtered_frames_dir(self) -> str:
        """Temporary filtered_frames directory."""
        return os.path.join(self.out_dir, "filtered_frames")

    def __repr__(self):
        return (f"<Config video={self.video!r} "
                f"gps_dir={self.gps_dir!r} "
                f"out_dir={self.out_dir!r}>")


cfg = _Config()