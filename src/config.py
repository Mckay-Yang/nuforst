import math
import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class Args:
    image: str = ""
    cache_dir: str = "./cache"
    force_refresh: bool = False
    start_time: str = "2015-01-01T00:00:00"
    end_time: str = "2024-01-01T00:00:00"
    target_time: Optional[str] = None
    time_unit: str = "seconds"
    modes: int = 4096
    eps: float = 1e-12
    num_peaks: int = 8
    power_cum: float = 0.7
    ignore_dc_hz: float = 1e-6
    refine_peaks: bool = True
    include_trend: bool = True
    ridge: float = 1e-2
    freq_weight: float = 2.0
    huber_iters: int = 3
    huber_delta: float = 1.5
    min_obs: int = 12
    n_jobs: int = -1
    show_progress: bool = True
    progress_every: int = 50
    output_path: str = "./recon.tif"


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


def build_args(overrides: Optional[dict] = None) -> Args:
    """解析命令行参数或从字典构建参数对象"""
    if overrides is not None:
        # 从默认值开始，应用 overrides
        args_obj = Args()
        for k, v in overrides.items():
            if hasattr(args_obj, k):
                setattr(args_obj, k, v)
        if args_obj.target_time is None:
            args_obj.target_time = args_obj.start_time
        return args_obj

    # 命令行模式
    ap = build_arg_parser()
    parsed = ap.parse_args()
    if parsed.target_time is None:
        parsed.target_time = parsed.start_time
    return Args(**vars(parsed))
