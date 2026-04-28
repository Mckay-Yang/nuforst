import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

CONFIG_DIR = Path(__file__).parent


def load_json_config(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f) or {}
    return {}


@dataclass
class NufrostArgs:
    image: object = None
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/local"))
    force_refresh: bool = False
    n_jobs: int = -1
    time_unit: str = "seconds"
    start_time: str = "2015-01-01T00:00:00"
    end_time: str = "2024-01-01T00:00:00"
    target_time: Optional[str] = None
    output_path: Path = field(default_factory=Path)
    show_progress: bool = True
    progress_every: int = 50
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


@dataclass
class HantsArgs:
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/local"))
    force_refresh: bool = False
    n_jobs: int = -1
    time_unit: str = "seconds"
    output_path: Path = field(default_factory=Path)
    nof: int = 3
    sf: str = "low"
    fet: float = 0.05
    dod: int = 5
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    period: float = 365.25


@dataclass
class Zhu2015Args:
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/local"))
    force_refresh: bool = False
    n_jobs: int = -1
    time_unit: str = "seconds"
    output_path: Path = field(default_factory=Path)
    lasso_alpha: float = 0.001


METHOD_ARGS = {
    "nufrost": NufrostArgs,
    "hants": HantsArgs,
    "zhu2015": Zhu2015Args,
}

Args = NufrostArgs


def build_args(method: str = "nufrost", overrides: Optional[dict] = None):
    shared = load_json_config("config")
    method_config = load_json_config(method)
    merged = {**shared, **method_config}
    if overrides:
        merged.update(overrides)

    cls = METHOD_ARGS.get(method)
    if cls is None:
        raise ValueError(f"Unknown method: {method}")

    args_obj = cls()
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
    if hasattr(args_obj, "target_time") and args_obj.target_time is None:
        args_obj.target_time = args_obj.start_time
    return args_obj
