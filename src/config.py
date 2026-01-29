import math
import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class Args:
    image: str
    cache_dir: str
    force_refresh: bool
    start_time: str
    end_time: str
    target_time: str
    time_unit: str
    modes: int
    eps: float
    num_peaks: int
    power_cum: float
    ignore_dc_hz: float
    refine_peaks: bool
    include_trend: bool
    ridge: float
    freq_weight: float
    huber_iters: int
    huber_delta: float
    min_obs: int
    n_jobs: int
    show_progress: bool
    progress_every: int
    output_path: str


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input multi-band time-series GeoTIFF", type=str,
                    default="drive/MyDrive/test_cube_0.1_degree/cube_test_B2_0.1.tif")
    ap.add_argument("-c", "--cache-dir", type=str, default="./cache", help="directory for cached npz cubes")
    ap.add_argument("--force-refresh", action="store_true", help="ignore cached npz and rebuild from the tif")

    # Time settings
    ap.add_argument("--start-time", type=str, default="2015-01-01T00:00:00",
                    help="start time in ISO format")
    ap.add_argument("--end-time", type=str, default="2024-01-01T00:00:00",
                    help="end time in ISO format")
    ap.add_argument("--target-time", type=str, default=None,
                    help="target reconstruction time in ISO format")
    ap.add_argument("--time-unit", type=str, default="seconds", choices=("seconds", "days"),
                    help="units used for timestamps")

    # Frequency/fitting settings
    ap.add_argument("--modes", type=int, default=4096)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--num-peaks", type=int, default=8)
    ap.add_argument("--power-cum", type=float, default=0.7)
    ap.add_argument("--ignore-dc-hz", type=float, default=1e-6)
    ap.add_argument("--refine-peaks", action="store_false", help="disable parabolic peak refinement")
    ap.add_argument("--include-trend", action="store_false", help="disable linear trend in fit")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--freq-weight", type=float, default=2.0)
    ap.add_argument("--huber-iters", type=int, default=3)
    ap.add_argument("--huber-delta", type=float, default=1.5)
    ap.add_argument("--min-obs", type=int, default=12, help="minimum valid observations per pixel")
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="number of parallel workers; 0=auto, 1=serial")
    ap.add_argument("--show-progress", action="store_true",
                    help="show progress bar/ETA if available")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="serial mode: print progress every N rows when tqdm is unavailable")

    # Output settings
    ap.add_argument("--output-path", type=str, default="./recon.tif",
                    help="output GeoTIFF path for reconstructed image")

    return ap


def build_args() -> Args:
    ap = build_arg_parser()
    # FIX 2: Removed [] to allow command line arguments
    args = ap.parse_args()

    if args.target_time is None:
        args.target_time = args.start_time
    return Args(**vars(args))


def build_args_from_dict(overrides: Optional[dict] = None) -> Args:
    """Build Args using default parser values and override with a dict.

    This is useful for notebook usage without command-line parsing.
    """
    ap = build_arg_parser()
    defaults = ap.parse_args([])
    payload = vars(defaults)
    if overrides:
        payload.update(overrides)
    if payload.get("target_time") is None:
        payload["target_time"] = payload.get("start_time")
    return Args(**payload)
