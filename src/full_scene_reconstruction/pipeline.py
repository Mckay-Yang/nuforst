from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed, cpu_count
from rasterio import Affine
from tqdm import tqdm

from config import build_args

from ..data_loader import (
    RSCube,
    _cache_subdir,
    _is_stale,
    _resolve_cache_root,
    _write_stacked_vrt,
    find_image_chunks,
)
from ..hants import hants_pixel
from ..logger import log as _log
from ..nufrost import nufrost_core, timestamps_to_seconds
from ..zhu2015 import fit_predict_pixel


SENTINEL_SOURCE = "sentinel-2"
HLS_SOURCE = "hls"
SUPPORTED_SOURCES = {SENTINEL_SOURCE, HLS_SOURCE}
DEFAULT_OUTPUT_ROOT = Path("data/output")
DEFAULT_CACHE_ROOT = Path("data/cache/local")
DEFAULT_LATE_FRACTION = 0.25
DEFAULT_MIN_VALID_RATIO = 0.9


def _resolve_data_dir(source_name: str, data_root: Path) -> Path:
    if source_name not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source: {source_name}")
    preferred = data_root / source_name
    if preferred.exists():
        return preferred

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidate = parent / "data" / source_name
        if candidate.exists():
            return candidate
    return preferred


def _location_token(lon: float, lat: float) -> str:
    return f"lon{lon:.4f}_lat{lat:.4f}"


def _location_output_token(lon: float, lat: float) -> str:
    return f"lon{lon:.6f}_lat{lat:.6f}"


def _sentinel_band_sort_key(name: str) -> Tuple[int, str]:
    match = re.fullmatch(r"B(\d+)([A-Z]?)", name)
    if not match:
        return (10_000, name)
    return (int(match.group(1)), match.group(2))


def _build_multi_file_vrt(
    stack_paths: Sequence[Path],
    *,
    cache_dir: Path | str,
    band_name: str,
    lon: float,
    lat: float,
) -> List[Path]:
    cache_root = _resolve_cache_root(cache_dir)
    vrt_dir = _cache_subdir(cache_root, "vrts")
    vrt_path = vrt_dir / f"sentinel_{band_name}_{_location_output_token(lon, lat)}.vrt"
    source_files = list(stack_paths)
    if _is_stale(vrt_path, source_files):
        _write_stacked_vrt(vrt_path, [source_files])
    return [vrt_path]


def discover_available_locations(data_dir: Path, source_name: str) -> List[Tuple[float, float]]:
    if source_name == SENTINEL_SOURCE:
        pattern = re.compile(r"COPERNICUS_S2_HARMONIZED_[A-Z0-9]+_lon(?P<lon>-?\d+\.\d+)_lat(?P<lat>-?\d+\.\d+)")
    elif source_name == HLS_SOURCE:
        pattern = re.compile(r"NASA_HLS_v\d+_[A-Z0-9]+_lon(?P<lon>-?\d+\.\d+)_lat(?P<lat>-?\d+\.\d+)")
    else:
        raise ValueError(f"Unsupported source: {source_name}")

    locations = {
        (float(match.group("lon")), float(match.group("lat")))
        for path in data_dir.glob("*.tif")
        for match in [pattern.search(path.name)]
        if match is not None
    }
    return sorted(locations)


def discover_location_band_stacks(
    data_dir: Path,
    source_name: str,
    lon: float,
    lat: float,
    cache_dir: Path | str = DEFAULT_CACHE_ROOT,
) -> Dict[str, List[Path]]:
    location_token = _location_token(lon, lat)
    stacks: Dict[str, List[Path]] = {}

    if source_name == SENTINEL_SOURCE:
        pattern = re.compile(r"COPERNICUS_S2_HARMONIZED_(?P<band>B\d+[A-Z]?)_lon")
        for path in sorted(data_dir.glob(f"*{location_token}*.tif")):
            match = pattern.search(path.name)
            if match:
                stacks.setdefault(match.group("band"), []).append(path)
        for band_name, paths in list(stacks.items()):
            if len(paths) > 1:
                stacks[band_name] = _build_multi_file_vrt(paths, cache_dir=cache_dir, band_name=band_name, lon=lon, lat=lat)
        return dict(sorted(stacks.items(), key=lambda item: _sentinel_band_sort_key(item[0])))

    if source_name == HLS_SOURCE:
        pattern = re.compile(r"NASA_HLS_v\d+_(?P<band>[A-Z0-9]+)_lon")
        bands = []
        for path in sorted(data_dir.glob(f"*{location_token}*.tif")):
            match = pattern.search(path.name)
            if match:
                bands.append(match.group("band"))

        for band in sorted(set(bands)):
            chunks = find_image_chunks(str(data_dir), lon=lon, lat=lat, band=band, cache_dir=cache_dir)
            if chunks:
                stacks[band] = [Path(chunk) for chunk in chunks]
        return dict(sorted(stacks.items()))

    raise ValueError(f"Unsupported source: {source_name}")


def intersect_band_timestamps(band_to_timestamps: Mapping[str, Sequence[str]]) -> List[str]:
    shared: Optional[set[str]] = None
    for timestamps in band_to_timestamps.values():
        current = set(str(ts) for ts in timestamps)
        shared = current if shared is None else shared & current
    return sorted(shared or [])


_VALID_RATIO_SUBSAMPLE_STEP = 8


def _batch_score_candidates(
    band_to_stack_paths: Mapping[str, Sequence[Path]],
    band_to_timestamps: Mapping[str, Sequence[str]],
    candidates: Sequence[str],
    cache_dir: Path | str = DEFAULT_CACHE_ROOT,
) -> Dict[str, Dict[str, float]]:
    _log("_batch_score_candidates", f"Scoring {len(candidates)} candidates across {len(band_to_stack_paths)} bands")
    scores: Dict[str, Dict[str, float]] = {band: {} for band in band_to_stack_paths}
    for band, stack_paths in band_to_stack_paths.items():
        resolved = [str(p) for p in stack_paths]
        loader = RSCube(resolved, cache_dir=cache_dir)
        data = loader.load()
        cube = np.asarray(data["cube"], dtype=np.float32)
        timestamps = [str(ts) for ts in data["timestamps"]]
        index_map = {ts: idx for idx, ts in enumerate(timestamps)}
        step = _VALID_RATIO_SUBSAMPLE_STEP
        sampled = cube[:, ::step, ::step]
        for candidate in candidates:
            idx = index_map.get(str(candidate))
            if idx is None:
                continue
            layer = sampled[idx]
            finite = np.isfinite(layer)
            total = finite.size
            if total == 0:
                scores[band][candidate] = 0.0
                continue
            if np.any(~finite):
                scores[band][candidate] = float(np.mean(finite))
            else:
                scores[band][candidate] = 1.0
    return scores


def select_shared_target_timestamp(
    candidates: Sequence[str],
    completeness_by_band: Mapping[str, Mapping[str, float]],
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    late_fraction: float = DEFAULT_LATE_FRACTION,
) -> str:
    if not candidates:
        raise ValueError("No shared timestamps available.")

    ordered = sorted(str(candidate) for candidate in candidates)
    tail_len = max(1, int(np.ceil(len(ordered) * late_fraction)))
    preferred = ordered[-tail_len:]
    fallback = ordered[:-tail_len]

    def _pick(pool: Sequence[str]) -> Optional[str]:
        for candidate in reversed(pool):
            if all(completeness_by_band[band].get(candidate, 0.0) >= min_valid_ratio for band in completeness_by_band):
                return candidate
        return None

    chosen = _pick(preferred)
    if chosen is not None:
        return chosen

    chosen = _pick(fallback)
    if chosen is not None:
        return chosen

    raise ValueError("No shared timestamp passed the completeness threshold.")


def choose_shared_target_timestamp(
    band_to_stack_paths: Mapping[str, Sequence[Path]],
    band_to_timestamps: Mapping[str, Sequence[str]],
    cache_dir: Path | str = DEFAULT_CACHE_ROOT,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    late_fraction: float = DEFAULT_LATE_FRACTION,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    shared = intersect_band_timestamps(band_to_timestamps)
    if not shared:
        raise ValueError("No shared timestamps exist across selected bands.")

    completeness = _batch_score_candidates(band_to_stack_paths, band_to_timestamps, shared, cache_dir=cache_dir)
    chosen = select_shared_target_timestamp(shared, completeness, min_valid_ratio=min_valid_ratio, late_fraction=late_fraction)
    return chosen, completeness


def build_output_path(output_root: Path, method_name: str, source_file: Path, target_time: str, *, source_name: str = "", lon: float = 0.0, lat: float = 0.0) -> Path:
    safe_time = target_time.replace(":", "-")
    if source_name and lon and lat:
        return output_root / f"{source_name}_recon" / f"{lon:.4f}_{lat:.4f}" / f"[{method_name}]_{source_name}_{_location_output_token(lon, lat)}_{safe_time}.tif"
    return output_root / method_name / f"[{method_name}]_{source_file.stem}_{safe_time}.tif"


def build_ground_truth_output_path(
    output_root: Path,
    source_name: str,
    lon: float,
    lat: float,
    target_time: str,
) -> Path:
    safe_time = target_time.replace(":", "-")
    return output_root / f"{source_name}_recon" / f"{lon:.4f}_{lat:.4f}" / f"[ground_truth]_{source_name}_{_location_output_token(lon, lat)}_{safe_time}.tif"


def build_scene_stack_output_path(
    output_root: Path,
    method_name: str,
    source_name: str,
    lon: float,
    lat: float,
    target_time: str,
    suffix: str,
) -> Path:
    safe_time = target_time.replace(":", "-")
    return output_root / f"{source_name}_recon" / f"{lon:.4f}_{lat:.4f}" / f"[{method_name}]_{source_name}_{_location_output_token(lon, lat)}_{safe_time}_{suffix}.tif"


def collapse_duplicate_timestamps(cube: np.ndarray, timestamps: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    timestamp_array = np.asarray([str(ts) for ts in timestamps], dtype="U32")
    unique_timestamps: List[str] = []
    merged_slices: List[np.ndarray] = []

    for timestamp in timestamp_array:
        if timestamp in unique_timestamps:
            continue
        match_indices = np.flatnonzero(timestamp_array == timestamp)
        if len(match_indices) == 1:
            merged = cube[int(match_indices[0])].astype(np.float32, copy=False)
        else:
            merged = np.nanmean(cube[match_indices], axis=0, dtype=np.float32)
        unique_timestamps.append(str(timestamp))
        merged_slices.append(np.asarray(merged, dtype=np.float32))

    merged_cube = np.stack(merged_slices, axis=0).astype(np.float32)
    return merged_cube, np.asarray(unique_timestamps, dtype="U32")


def make_masked_time_series(cube: np.ndarray, timestamps: Sequence[str], target_time: str) -> Tuple[np.ndarray, np.ndarray, int]:
    timestamp_array = np.asarray([str(ts) for ts in timestamps], dtype="U32")
    match_indices = np.flatnonzero(timestamp_array == target_time)
    if len(match_indices) != 1:
        raise ValueError(f"Expected exactly one matching timestamp for {target_time}, found {len(match_indices)}")
    target_idx = int(match_indices[0])
    masked_cube = np.delete(cube, target_idx, axis=0)
    masked_timestamps = np.delete(timestamp_array, target_idx)
    return masked_cube, masked_timestamps, target_idx


def extract_prediction_2d(method_name: str, prediction: np.ndarray) -> np.ndarray:
    if method_name == "zhu2015":
        if prediction.ndim != 3 or prediction.shape[0] < 1:
            raise ValueError("Zhu2015 prediction must have shape (2, H, W) or compatible.")
        return np.asarray(prediction[0], dtype=np.float32)
    return np.asarray(prediction, dtype=np.float32)


def write_run_summary(output_root: Path, payload: Dict[str, Any]) -> Path:
    summary_dir = output_root / "run_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    safe_time = str(payload["target_time"]).replace(":", "-")
    lon_token = f"{float(payload['lon']):.6f}"
    lat_token = f"{float(payload['lat']):.6f}"
    source_token = str(payload["source"]).replace("/", "-")
    summary_path = summary_dir / f"reconstruction_summary_{source_token}_lon{lon_token}_lat{lat_token}_{safe_time}.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def _summary_path_for_location(output_root: Path, source_name: str, lon: float, lat: float) -> Path:
    safe_source = source_name.replace("/", "-")
    lon_token = f"{lon:.6f}"
    lat_token = f"{lat:.6f}"
    return output_root / "run_summaries" / f"reconstruction_summary_{safe_source}_lon{lon_token}_lat{lat_token}.json"


def _summary_matches_run(
    candidate: Path,
    *,
    methods: Sequence[str],
    window_size: Optional[int],
    source_files: Mapping[str, Sequence[str]],
    min_valid_ratio: float,
    late_fraction: float,
) -> bool:
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return False

    candidate_methods = tuple(str(method) for method in payload.get("methods", ("nufrost", "hants", "zhu2015")))
    candidate_window_size = payload.get("window_size")
    candidate_source_files = payload.get("source_files", {})
    candidate_min_valid_ratio = float(payload.get("min_valid_ratio", DEFAULT_MIN_VALID_RATIO))
    candidate_late_fraction = float(payload.get("late_fraction", DEFAULT_LATE_FRACTION))
    return (
        candidate_methods == tuple(methods)
        and candidate_window_size == window_size
        and candidate_source_files == {band: list(paths) for band, paths in source_files.items()}
        and candidate_min_valid_ratio == float(min_valid_ratio)
        and candidate_late_fraction == float(late_fraction)
    )


def _find_existing_summary(
    output_root: Path,
    source_name: str,
    lon: float,
    lat: float,
    *,
    methods: Sequence[str],
    window_size: Optional[int],
    source_files: Mapping[str, Sequence[str]],
    min_valid_ratio: float,
    late_fraction: float,
) -> Optional[Path]:
    pattern = _summary_path_for_location(output_root, source_name, lon, lat)
    pattern_parent = pattern.parent
    if not pattern_parent.exists():
        return None
    stem_prefix = pattern.stem
    for candidate in sorted(pattern_parent.glob(f"{stem_prefix}_*.json")):
        if _summary_matches_run(
            candidate,
            methods=methods,
            window_size=window_size,
            source_files=source_files,
            min_valid_ratio=min_valid_ratio,
            late_fraction=late_fraction,
        ):
            return candidate
    return None


def _parse_target_datetime(target_time: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(target_time, utc=True)
    except Exception:
        return pd.to_datetime(target_time)


def _resolve_n_jobs(n_jobs: int, height: int) -> int:
    if n_jobs <= 0:
        n_jobs = max(1, int(cpu_count()))
    return max(1, min(n_jobs, height))


def _run_parallel_rows(height: int, worker, n_jobs: int, desc: str):
    resolved_jobs = _resolve_n_jobs(n_jobs, height)
    if resolved_jobs == 1:
        return [worker(row_idx) for row_idx in tqdm(range(height), total=height, desc=desc)]
    results_gen = Parallel(n_jobs=resolved_jobs, prefer="processes", return_as="generator")(
        delayed(worker)(row_idx) for row_idx in range(height)
    )
    return list(tqdm(results_gen, total=height, desc=desc))


def _crop_loaded_cube(data: Mapping[str, object], window_size: Optional[int]) -> Dict[str, object]:
    if window_size is None:
        return dict(data)
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    cropped = dict(data)
    cube = data.get("cube")
    if cube is None:
        raise ValueError("Loaded cube data must include 'cube'")
    cropped["cube"] = cube[:, :window_size, :window_size]
    return cropped


def reconstruct_hants_from_cube(
    cube: np.ndarray,
    timestamps: Sequence[str],
    target_time: str,
    *,
    nof: int = 3,
    sf: str = "low",
    fet: float = 0.05,
    dod: int = 5,
    n_jobs: int = -1,
) -> np.ndarray:
    target_dt = _parse_target_datetime(target_time)
    timestamps_sec = timestamps_to_seconds(np.asarray(timestamps, dtype="U32"))
    t0_sec = float(np.min(timestamps_sec))
    t_days = (timestamps_sec - t0_sec) / 86400.0
    target_t_day = (target_dt.timestamp() - t0_sec) / 86400.0

    _, height, width = cube.shape
    out = np.full((height, width), np.nan, dtype=np.float32)

    def _process_row(row_idx: int):
        row = np.full(width, np.nan, dtype=np.float32)
        for col_idx in range(width):
            row[col_idx] = hants_pixel(t_days, cube[:, row_idx, col_idx], target_t_day, nof=nof, sf=sf, fet=fet, dod=dod)
        return row_idx, row

    for row_idx, row in _run_parallel_rows(height, _process_row, n_jobs=n_jobs, desc="HANTS Rows"):
        out[row_idx, :] = row
    return out


def reconstruct_zhu2015_from_cube(
    cube: np.ndarray,
    timestamps: Sequence[str],
    target_time: str,
    *,
    lasso_alpha: float = 0.001,
    n_jobs: int = -1,
) -> np.ndarray:
    target_dt = _parse_target_datetime(target_time)
    timestamps_sec = timestamps_to_seconds(np.asarray(timestamps, dtype="U32"))
    t0_sec = float(np.min(timestamps_sec))
    t_days = (timestamps_sec - t0_sec) / 86400.0
    target_t_day = (target_dt.timestamp() - t0_sec) / 86400.0

    _, height, width = cube.shape
    out = np.full((2, height, width), np.nan, dtype=np.float32)

    def _process_row(row_idx: int):
        row = np.full((2, width), np.nan, dtype=np.float32)
        for col_idx in range(width):
            pred, qa = fit_predict_pixel(t_days, cube[:, row_idx, col_idx], target_t_day, lasso_alpha=lasso_alpha)
            row[0, col_idx] = pred
            row[1, col_idx] = qa
        return row_idx, row

    for row_idx, row in _run_parallel_rows(height, _process_row, n_jobs=n_jobs, desc="Zhu2015 Rows"):
        out[:, row_idx, :] = row
    return out


def _write_prediction(output_path: Path, array: np.ndarray, meta: Mapping[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        height, width = array.shape
        count = 1
    else:
        count, height, width = array.shape

    transform = None
    if meta.get("transform") is not None:
        transform = Affine(*meta["transform"])

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=meta.get("crs_wkt"),
        transform=transform,
    ) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)


def write_band_stack(
    output_path: Path,
    arrays_by_band: Mapping[str, np.ndarray],
    ordered_bands: Sequence[str],
    meta: Mapping[str, object],
) -> None:
    stack = np.stack([np.asarray(arrays_by_band[band], dtype=np.float32) for band in ordered_bands], axis=0)
    _write_prediction(output_path, stack, meta)
    with rasterio.open(output_path, "r+") as dst:
        for idx, band_name in enumerate(ordered_bands, start=1):
            dst.set_band_description(idx, str(band_name))


def validate_band_metadata_consistency(
    ordered_bands: Sequence[str],
    band_meta: Mapping[str, Mapping[str, object]],
) -> None:
    if not ordered_bands:
        return
    reference_band = ordered_bands[0]
    reference_transform = tuple(band_meta[reference_band].get("transform") or ())
    reference_crs = band_meta[reference_band].get("crs_wkt")
    reference_shape = band_meta[reference_band]["cube"].shape[1:]

    for band_name in ordered_bands[1:]:
        current_transform = tuple(band_meta[band_name].get("transform") or ())
        current_crs = band_meta[band_name].get("crs_wkt")
        current_shape = band_meta[band_name]["cube"].shape[1:]
        if current_transform != reference_transform or current_crs != reference_crs or current_shape != reference_shape:
            raise ValueError(
                f"Band metadata mismatch between {reference_band} and {band_name}: "
                f"transform/crs/shape must match before writing merged stacks."
            )


def reconstruct_full_scene_for_location(
    source_name: str,
    lon: float,
    lat: float,
    *,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    data_root: Path | str = Path("data"),
    cache_dir: Path | str = DEFAULT_CACHE_ROOT,
    methods: Sequence[str] = ("nufrost", "hants", "zhu2015"),
    n_jobs: int = -1,
    force_refresh: bool = False,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    late_fraction: float = DEFAULT_LATE_FRACTION,
    window_size: Optional[int] = None,
) -> Dict[str, Any]:
    _log("reconstruct_full_scene_for_location", f"Start source={source_name} lon={lon:.4f} lat={lat:.4f} methods={list(methods)} window_size={window_size}")
    output_root = Path(output_root)
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)

    data_dir = _resolve_data_dir(source_name, data_root)
    band_stacks = discover_location_band_stacks(data_dir, source_name=source_name, lon=lon, lat=lat, cache_dir=cache_dir)
    if not band_stacks:
        raise FileNotFoundError(f"No stacks found for {source_name} lon={lon:.4f} lat={lat:.4f}")

    source_map = {band: [str(path) for path in stack_paths] for band, stack_paths in band_stacks.items()}
    existing_summary = _find_existing_summary(
        output_root,
        source_name,
        lon,
        lat,
        methods=methods,
        window_size=window_size,
        source_files=source_map,
        min_valid_ratio=min_valid_ratio,
        late_fraction=late_fraction,
    )
    if existing_summary is not None:
        _log("reconstruct_full_scene_for_location", f"Skipping (existing summary): {existing_summary}")
        return {"skipped": True, "summary_path": str(existing_summary)}

    band_to_timestamps: Dict[str, List[str]] = {}
    band_to_data: Dict[str, Dict[str, object]] = {}
    _log("reconstruct_full_scene_for_location", f"Loading {len(band_stacks)} bands from cache")
    for band_name, stack_paths in band_stacks.items():
        loader = RSCube([str(p) for p in stack_paths], cache_dir=cache_dir, force_refresh=force_refresh)
        data = loader.load()
        band_to_timestamps[band_name] = [str(ts) for ts in data["timestamps"]]
        band_to_data[band_name] = data

    target_time, completeness = choose_shared_target_timestamp(
        band_stacks,
        band_to_timestamps,
        cache_dir=cache_dir,
        min_valid_ratio=min_valid_ratio,
        late_fraction=late_fraction,
    )
    _log("reconstruct_full_scene_for_location", f"Selected target_time={target_time}")

    output_map: Dict[str, Dict[str, str]] = {method: {} for method in methods}
    merged_prediction_map: Dict[str, str] = {}
    timing_seconds: Dict[str, Dict[str, float]] = {method: {} for method in methods}
    mask_indices: Dict[str, int] = {}
    counts_before: Dict[str, int] = {}
    counts_after: Dict[str, int] = {}
    prediction_arrays: Dict[str, Dict[str, np.ndarray]] = {method: {} for method in methods}
    ground_truth_arrays: Dict[str, np.ndarray] = {}
    band_meta: Dict[str, Mapping[str, object]] = {}

    for band_name, stack_paths in band_stacks.items():
        _log("reconstruct_full_scene_for_location", f"Band {band_name}: cube shape={band_to_data[band_name]['cube'].shape}, timestamps={counts_before.get(band_name, '?')}")
        data = _crop_loaded_cube(band_to_data[band_name], window_size)
        cube = np.ma.filled(data["cube"], np.nan).astype(np.float32)
        timestamps = [str(ts) for ts in data["timestamps"]]
        cube, deduped_timestamps = collapse_duplicate_timestamps(cube, timestamps)
        timestamps = deduped_timestamps.tolist()
        masked_cube, masked_timestamps, target_idx = make_masked_time_series(cube, timestamps, target_time)
        held_out_truth = cube[target_idx].astype(np.float32)
        ground_truth_arrays[band_name] = held_out_truth
        mask_indices[band_name] = target_idx
        counts_before[band_name] = int(len(timestamps))
        counts_after[band_name] = int(len(masked_timestamps))
        band_meta[band_name] = data

        build_nufrost_args = {
            "cache_dir": cache_dir,
            "n_jobs": n_jobs,
            "force_refresh": force_refresh,
            "target_time": target_time,
        }
        for method_name in methods:
            _log("reconstruct_full_scene_for_location", f"Band {band_name}: running {method_name}")
            output_path = build_output_path(output_root=output_root, method_name=method_name, source_file=stack_paths[0], target_time=target_time, source_name=source_name, lon=lon, lat=lat)
            t0 = time.perf_counter()
            if method_name == "nufrost":
                args = build_args(dict(build_nufrost_args))
                prediction = nufrost_core(masked_cube, masked_timestamps, target_time, args=args)
            elif method_name == "hants":
                prediction = reconstruct_hants_from_cube(masked_cube, masked_timestamps, target_time, n_jobs=n_jobs)
            elif method_name == "zhu2015":
                prediction = reconstruct_zhu2015_from_cube(masked_cube, masked_timestamps, target_time, n_jobs=n_jobs)
            else:
                raise ValueError(f"Unsupported method: {method_name}")
            elapsed = time.perf_counter() - t0
            timing_seconds[method_name][band_name] = elapsed
            _log("reconstruct_full_scene_for_location", f"Band {band_name} / {method_name}: {elapsed:.1f}s")

            prediction_2d = extract_prediction_2d(method_name, prediction)
            _write_prediction(output_path, prediction_2d, data)
            output_map[method_name][band_name] = str(output_path)
            prediction_arrays[method_name][band_name] = prediction_2d

    ordered_bands = list(band_stacks.keys())
    validate_band_metadata_consistency(ordered_bands, band_meta)
    first_meta = band_meta[ordered_bands[0]]

    gt_path = build_ground_truth_output_path(
        output_root=output_root,
        source_name=source_name,
        lon=lon,
        lat=lat,
        target_time=target_time,
    )
    write_band_stack(gt_path, ground_truth_arrays, ordered_bands, first_meta)

    for method_name in methods:
        merged_prediction_path = build_scene_stack_output_path(
            output_root=output_root,
            method_name=method_name,
            source_name=source_name,
            lon=lon,
            lat=lat,
            target_time=target_time,
            suffix="prediction",
        )
        write_band_stack(merged_prediction_path, prediction_arrays[method_name], ordered_bands, first_meta)
        merged_prediction_map[method_name] = str(merged_prediction_path)

    for method_name in methods:
        for band_name, per_band_path in output_map[method_name].items():
            Path(per_band_path).unlink(missing_ok=True)

    payload: Dict[str, Any] = {
        "source": source_name,
        "lon": lon,
        "lat": lat,
        "target_time": target_time,
        "methods": list(methods),
        "bands": list(band_stacks.keys()),
        "source_files": source_map,
        "mask_indices": mask_indices,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "completeness": completeness,
        "min_valid_ratio": min_valid_ratio,
        "late_fraction": late_fraction,
        "merged_prediction_outputs": merged_prediction_map,
        "ground_truth_output": str(gt_path),
        "timing_seconds": timing_seconds,
        "window_size": window_size,
    }
    summary_path = write_run_summary(output_root, payload)
    payload["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log("reconstruct_full_scene_for_location", f"Complete source={source_name} lon={lon:.4f} lat={lat:.4f} target={target_time}")
    return payload


def reconstruct_full_scene_for_all_locations(
    source_name: str,
    *,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    data_root: Path | str = Path("data"),
    cache_dir: Path | str = DEFAULT_CACHE_ROOT,
    methods: Sequence[str] = ("nufrost", "hants", "zhu2015"),
    n_jobs: int = -1,
    force_refresh: bool = False,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    late_fraction: float = DEFAULT_LATE_FRACTION,
) -> List[Dict[str, Any]]:
    data_root = Path(data_root)
    data_dir = _resolve_data_dir(source_name, data_root)
    locations = discover_available_locations(data_dir, source_name=source_name)
    if not locations:
        raise FileNotFoundError(f"No locations found for {source_name} in {data_dir}")

    _log("reconstruct_full_scene_for_all_locations", f"Batch start source={source_name} locations={len(locations)}")
    results: List[Dict[str, Any]] = []
    for idx, (lon, lat) in enumerate(locations):
        _log("reconstruct_full_scene_for_all_locations", f"Location {idx+1}/{len(locations)}: lon={lon:.4f} lat={lat:.4f}")
        results.append(
            reconstruct_full_scene_for_location(
                source_name=source_name,
                lon=lon,
                lat=lat,
                output_root=output_root,
                data_root=data_root,
                cache_dir=cache_dir,
                methods=methods,
                n_jobs=n_jobs,
                force_refresh=force_refresh,
                min_valid_ratio=min_valid_ratio,
                late_fraction=late_fraction,
            )
        )
    return results
