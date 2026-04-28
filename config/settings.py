import math
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Union, List

# Define path to config.yaml (assumed to be in the same directory)
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_yaml_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

@dataclass
class Args:
    # Defaults here serve as code-level fallbacks if YAML is missing/incomplete
    image: Union[Path, List[Path], List[str], str] = field(default_factory=Path)
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/local"))
    force_refresh: bool = False
    start_time: str = "2015-01-01T00:00:00"
    end_time: str = "2024-01-01T00:00:00"
    target_time: Optional[str] = None
    time_unit: str = "seconds"
    modes: int = 4096
    eps: float = 1e-12
    num_peaks: int = 10
    power_cum: float = 0.7
    ignore_dc_hz: float = 1e-10
    frequency_selection: str = "hybrid"
    preferred_periods_days: str = "365.25,182.625,91.3125,30.4375"
    preferred_top_k: int = 4
    spectral_top_k: int = 4
    spectral_merge_tol: float = 0.15
    refine_peaks: bool = True
    include_trend: bool = True
    ridge: float = 0.005
    freq_weight: float = 2.0
    huber_iters: int = 3
    huber_delta: float = 0.05
    min_obs: int = 12
    n_jobs: int = -1
    show_progress: bool = True
    progress_every: int = 50
    output_path: Path = field(default_factory=Path)

def build_arg_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Helper to get default
    def d(key, fallback):
        return defaults.get(key, fallback)

    ap.add_argument("-i", "--image", help="path to input multi-band time-series GeoTIFF", type=Path,
                    default=Path(d("image", "")) if d("image", "") else Path())
    ap.add_argument("-c", "--cache-dir", type=Path, default=Path(d("cache_dir", "data/cache/local")),
                    help="directory for cached npz cubes")
    ap.add_argument("--force-refresh", action="store_true", default=d("force_refresh", False),
                    help="ignore cached npz and rebuild from the tif")

    # Time settings
    ap.add_argument("--start-time", type=str, default=d("start_time", "2015-01-01T00:00:00"),
                    help="start time in ISO format")
    ap.add_argument("--end-time", type=str, default=d("end_time", "2024-01-01T00:00:00"),
                    help="end time in ISO format")
    ap.add_argument("--target-time", type=str, default=d("target_time", None),
                    help="target reconstruction time in ISO format")
    ap.add_argument("--time-unit", type=str, default=d("time_unit", "seconds"), choices=("seconds", "days"),
                    help="units used for timestamps")

    # Frequency/fitting settings
    ap.add_argument("--modes", type=int, default=d("modes", 4096))
    ap.add_argument("--eps", type=float, default=d("eps", 1e-12))
    ap.add_argument("--num-peaks", type=int, default=d("num_peaks", 10))
    ap.add_argument("--power-cum", type=float, default=d("power_cum", 0.7))
    ap.add_argument("--ignore-dc-hz", type=float, default=d("ignore_dc_hz", 1e-10))
    ap.add_argument("--frequency-selection", type=str, default=d("frequency_selection", "hybrid"),
                    choices=("spectral", "preferred", "hybrid", "shared_spectral"))
    ap.add_argument("--preferred-periods-days", type=str, default=d("preferred_periods_days", "365.25,182.625,91.3125,30.4375"),
                    help="comma-separated preferred periods in days, e.g. annual/semiannual/seasonal/monthly")
    ap.add_argument("--preferred-top-k", type=int, default=d("preferred_top_k", 4))
    ap.add_argument("--spectral-top-k", type=int, default=d("spectral_top_k", 4))
    ap.add_argument("--spectral-merge-tol", type=float, default=d("spectral_merge_tol", 0.15),
                    help="relative tolerance for merging spectral peaks with preferred frequencies")

    # Boolean flags handling
    # If default is True, we want a flag to disable it (store_false)
    # If default is False, we want a flag to enable it (store_true)

    if d("refine_peaks", True):
        ap.add_argument("--no-refine-peaks", dest="refine_peaks", action="store_false",
                        help="disable parabolic peak refinement")
    else:
        ap.add_argument("--refine-peaks", action="store_true", default=False,
                        help="enable parabolic peak refinement")

    if d("include_trend", True):
        ap.add_argument("--no-include-trend", dest="include_trend", action="store_false",
                        help="disable linear trend in fit")
    else:
        ap.add_argument("--include-trend", action="store_true", default=False,
                        help="enable linear trend in fit")

    ap.add_argument("--ridge", type=float, default=d("ridge", 0.005))
    ap.add_argument("--freq-weight", type=float, default=d("freq_weight", 2.0))
    ap.add_argument("--huber-iters", type=int, default=d("huber_iters", 3))
    ap.add_argument("--huber-delta", type=float, default=d("huber_delta", 0.05))
    ap.add_argument("--min-obs", type=int, default=d("min_obs", 12), help="minimum valid observations per pixel")
    ap.add_argument("--n-jobs", type=int, default=d("n_jobs", -1),
                    help="number of parallel workers; 0=auto, 1=serial")

    if d("show_progress", True):
        ap.add_argument("--no-progress", dest="show_progress", action="store_false",
                        help="disable progress bar")
    else:
        ap.add_argument("--show-progress", action="store_true", default=False,
                        help="show progress bar")

    ap.add_argument("--progress-every", type=int, default=d("progress_every", 50),
                    help="serial mode: print progress every N rows when tqdm is unavailable")

    # Output settings
    ap.add_argument("--output-path", type=Path, default=Path(d("output_path", "./recon.tif")),
                    help="output GeoTIFF path for reconstructed image")

    # Set defaults explicitly to ensure they are propagated
    ap.set_defaults(**defaults)

    return ap


def build_args(overrides: Optional[dict] = None) -> Args:
    """Parse args from YAML + CLI or overrides."""
    yaml_config = load_yaml_config()

    # If overrides provided (Python API usage)
    if overrides is not None:
        # Merge: YAML -> Overrides
        merged = yaml_config.copy()
        merged.update(overrides)

        args_obj = Args()
        for k, v in merged.items():
            if hasattr(args_obj, k):
                if k in ("cache_dir", "output_path") and v is not None:
                    setattr(args_obj, k, Path(v))
                elif k == "image" and v is not None:
                    if isinstance(v, list):
                        setattr(args_obj, k, v)
                    else:
                        setattr(args_obj, k, Path(v))
                else:
                    setattr(args_obj, k, v)

        if args_obj.target_time is None:
            args_obj.target_time = args_obj.start_time
        return args_obj

    # CLI usage
    # YAML defaults are passed to parser
    ap = build_arg_parser(yaml_config)
    parsed = ap.parse_args()

    if parsed.target_time is None:
        parsed.target_time = parsed.start_time

    return Args(**vars(parsed))
