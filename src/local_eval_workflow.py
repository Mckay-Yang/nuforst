from __future__ import annotations

import glob
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import src.data_loader
import src.evaluation
from config import build_args


DEFAULT_SPARSE_POINT_LEVELS = [1000, 5000, 10000, 20000]
DEFAULT_GAP_INDEX_TARGETS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.46, 0.52, 0.60, 0.70, 0.80]
DEFAULT_REPEATABILITY_SEEDS = [11, 23, 37, 53, 71]
DEFAULT_ABLATION_VARIANTS = [
    {"name": "Full NUFROST", "overrides": {}},
    {"name": "w/o preferred frequencies", "overrides": {"frequency_selection": "spectral"}},
    {"name": "w/o parabolic refinement", "overrides": {"refine_peaks": False}},
    {"name": "w/o Huber robust fitting", "overrides": {"huber_iters": 0}},
    {"name": "w/o frequency-weighted ridge", "overrides": {"freq_weight": 0.0}},
    {"name": "w/o linear trend", "overrides": {"include_trend": False}},
]


@dataclass
class LocalEvalConfig:
    source_name: str
    project_dir: Path
    output_dir: Path
    cache_dir: Path
    max_images: Optional[int] = None
    n_jobs: int = -1
    base_seed: int = 42
    sparse_point_levels: List[int] = field(default_factory=lambda: list(DEFAULT_SPARSE_POINT_LEVELS))
    gap_index_targets: List[float] = field(default_factory=lambda: list(DEFAULT_GAP_INDEX_TARGETS))
    max_gap_samples: Optional[int] = None
    ablation_gap_index: float = 0.30
    gap_max_missing_ratio: float = 0.08
    gap_max_native_gap_days: int = 60
    repeatability_seeds: List[int] = field(default_factory=lambda: list(DEFAULT_REPEATABILITY_SEEDS))
    repeatability_image_limit: int = 5
    repeatability_random_points: int = 10000
    repeatability_gap_index: float = 0.30
    repeatability_gap_samples: int = 500
    ablation_variants: List[dict] = field(default_factory=lambda: list(DEFAULT_ABLATION_VARIANTS))
    run_ablation: bool = True
    run_sparse: bool = True
    run_gap: bool = True
    run_repeatability: bool = True

    @property
    def image_dir(self) -> Path:
        return self.project_dir / f"data/{self.source_name}"

    @property
    def output_paths(self) -> Dict[str, Path]:
        return {
            "ablation": self.output_dir / f"{self.source_name}_ablation_results.csv",
            "sparse": self.output_dir / f"{self.source_name}_sparse_sweep_results.csv",
            "gap": self.output_dir / f"{self.source_name}_gap_sweep_results.csv",
            "repeatability": self.output_dir / f"{self.source_name}_repeatability_results.csv",
        }

    @property
    def max_random_points(self) -> int:
        return max(self.sparse_point_levels)

    @property
    def ablation_random_points(self) -> int:
        return self.max_random_points


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return base_seed + (zlib.adler32(payload) % 1_000_000)


def loc_id_from_paths(image_paths: Sequence[str]) -> str:
    stem = Path(image_paths[0]).stem
    match = re.search(r"([A-Z0-9]+_lon[0-9.]+_lat[0-9.]+)", stem)
    return match.group(1) if match else stem


def parse_loc_id(loc_id: str) -> Dict[str, object]:
    match = re.fullmatch(r"([A-Z0-9]+)_lon([0-9.]+)_lat([0-9.]+)", loc_id)
    if not match:
        return {"Band": None, "Lon": None, "Lat": None}
    return {"Band": match.group(1), "Lon": float(match.group(2)), "Lat": float(match.group(3))}


def append_rows(csv_path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def log_step(message: str) -> None:
    from .logger import log as _log
    _log("log_step", message)


def load_done_keys(csv_path: Path, key_columns: Sequence[str]) -> set[tuple]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Could not read {csv_path.name}: {exc}")
        return set()
    if any(col not in df.columns for col in key_columns):
        return set()
    return set(tuple(row[col] for col in key_columns) for _, row in df.iterrows())


def make_batch_checkpoint_writer(
    csv_path: Path,
    *,
    loc_id: str,
    scenario: str,
    variant: str,
    gap_length: Optional[int] = None,
    gap_index_target: Optional[float] = None,
    loc_meta: Optional[Mapping[str, object]] = None,
):
    checkpoint_path = csv_path.parent / f"{csv_path.stem}_checkpoint.csv"

    def on_batch_done(batch_results, batch_start: int, batch_end: int) -> None:
        n_valid = sum(1 for result in batch_results if result is not None)
        if n_valid == 0:
            return
        print(f"  [checkpoint] {scenario}/{variant} pixels {batch_start}-{batch_end}: {n_valid}/{len(batch_results)} valid", flush=True)
        row = {
            "Image": loc_id,
            "Scenario": scenario,
            "Variant": variant,
            "BatchStart": batch_start,
            "BatchEnd": batch_end,
            "ValidPixels": n_valid,
            "Timestamp": pd.Timestamp.now().isoformat(),
        }
        if gap_length is not None:
            row["GapLength"] = gap_length
        if gap_index_target is not None:
            row["GapIndexTarget"] = gap_index_target
        if loc_meta:
            row.update(loc_meta)
        append_rows(checkpoint_path, pd.DataFrame([row]))

    return on_batch_done, checkpoint_path


def discover_image_paths_list(config: LocalEvalConfig) -> List[List[str]]:
    files = glob.glob((config.image_dir / "*.tif").as_posix())
    loc_ids = set()
    for file_path in files:
        name = Path(file_path).name
        match = re.search(r"_([A-Z0-9]+)_lon([0-9.]+)_lat([0-9.]+).*?(?:_part\d+)?(?:-\d{10}-\d{10})?\.tif$", name)
        if match:
            loc_ids.add((match.group(1), float(match.group(2)), float(match.group(3))))

    image_paths_list: List[List[str]] = []
    for band, lon, lat in sorted(loc_ids):
        chunks = src.data_loader.find_image_chunks(
            config.image_dir.as_posix(),
            lon,
            lat,
            band,
            cache_dir=config.cache_dir.as_posix(),
        )
        if chunks:
            image_paths_list.append(chunks)

    if config.max_images is not None:
        image_paths_list = image_paths_list[: config.max_images]
    return image_paths_list


def get_pixel_stats(source, t_days: np.ndarray, loc_id: str, min_obs: int, pixel_stats_cache: Dict[str, Dict[str, np.ndarray]]):
    if loc_id not in pixel_stats_cache:
        log_step(f"Scanning pixel stats for {loc_id} (window-based, no per-pixel I/O)")
        pixel_stats_cache[loc_id] = src.evaluation.scan_pixel_stats_from_source(source, t_days)
        stats = pixel_stats_cache[loc_id]
        vc = stats["valid_counts"]
        n_valid = int(np.sum(vc >= max(min_obs + 1, 3)))
        n_gap = int(np.sum(vc >= max(min_obs + 3, 15)))
        log_step(f"Pixel stats done: {n_valid} pixels with >={max(min_obs + 1, 3)} obs, {n_gap} with >={max(min_obs + 3, 15)} obs")
    return pixel_stats_cache[loc_id]


def select_gap_pixels(
    source,
    t_days: np.ndarray,
    *,
    min_obs: int,
    num_samples: Optional[int],
    seed: int,
    max_missing_ratio: float,
    max_native_gap_days: int,
    loc_id: str,
    pixel_stats_cache: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, int, int]:
    stats = get_pixel_stats(source, t_days, loc_id, min_obs, pixel_stats_cache)
    candidates = src.evaluation.scan_gap_candidates_from_source(
        source,
        t_days,
        min_obs,
        max_candidates=50000,
        seed=seed,
        precomputed_stats=stats,
    )
    filtered = [(r, c) for r, c, missing_ratio, native_gap_days in candidates if missing_ratio <= max_missing_ratio and native_gap_days <= max_native_gap_days]
    if not filtered:
        return np.empty((0, 2), dtype=int), len(candidates), 0
    filtered_arr = np.array(filtered, dtype=int)
    if num_samples is not None and len(filtered_arr) > num_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(filtered_arr), num_samples, replace=False)
        filtered_arr = filtered_arr[idx]
    return filtered_arr, len(candidates), len(filtered)


def gap_days_from_index_targets(t_days: np.ndarray, targets: Sequence[float]) -> Tuple[List[Tuple[int, float]], float]:
    total_span_days = float(np.nanmax(t_days) - np.nanmin(t_days))
    if (not np.isfinite(total_span_days)) or total_span_days <= 0:
        return [], total_span_days
    gap_specs = []
    for idx_val in targets:
        if idx_val <= 0:
            continue
        approx_days = int(round(total_span_days * np.sqrt(idx_val)))
        approx_days = max(1, min(approx_days, int(total_span_days * 0.95)))
        gap_specs.append((approx_days, float(idx_val)))
    dedup = {}
    for gap_days, idx_val in gap_specs:
        dedup[gap_days] = idx_val
    return sorted(dedup.items()), total_span_days


def gap_days_for_index(t_days: np.ndarray, index_value: float) -> Tuple[int, float]:
    gap_specs, total_span_days = gap_days_from_index_targets(t_days, [index_value])
    if not gap_specs:
        return 1, total_span_days
    return gap_specs[0][0], total_span_days


def build_eval_args(image_paths: Sequence[str], cache_dir: Path, n_jobs: int, overrides: Optional[Mapping[str, object]] = None):
    args = build_args("nufrost", dict(overrides or {}))
    args.image = list(image_paths)
    args.cache_dir = cache_dir.as_posix()
    args.force_refresh = False
    args.n_jobs = n_jobs
    return args


def _add_loc_meta(df: pd.DataFrame, loc_id: str, loc_meta: Mapping[str, object]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Image"] = loc_id
    for key, value in loc_meta.items():
        df[key] = value
    return df


def run_local_evals_workflow(
    *,
    source_name: str,
    project_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    max_images: Optional[int] = None,
    n_jobs: int = -1,
    run_ablation: bool = True,
    run_sparse: bool = True,
    run_gap: bool = True,
    run_repeatability: bool = True,
) -> Dict[str, object]:
    config = LocalEvalConfig(
        source_name=source_name,
        project_dir=project_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        max_images=max_images,
        n_jobs=n_jobs,
        run_ablation=run_ablation,
        run_sparse=run_sparse,
        run_gap=run_gap,
        run_repeatability=run_repeatability,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths_list = discover_image_paths_list(config)
    log_step(f"Found {len(image_paths_list)} distinct spatial/band chunks to evaluate.")
    if not image_paths_list:
        return {"status": "no_inputs", "source_name": source_name, "image_count": 0}

    output_paths = config.output_paths
    ablation_done = load_done_keys(output_paths["ablation"], ["Image", "Scenario", "Variant"])
    sparse_done = load_done_keys(output_paths["sparse"], ["Image", "NumPoints"])
    gap_done = load_done_keys(output_paths["gap"], ["Image", "GapLength"])
    repeat_done = load_done_keys(output_paths["repeatability"], ["Image", "Scenario", "RepeatSeed"])
    repeatability_targets = {loc_id_from_paths(paths) for paths in image_paths_list[: config.repeatability_image_limit]}
    pixel_stats_cache: Dict[str, Dict[str, np.ndarray]] = {}
    summary = {"ablation": 0, "sparse": 0, "gap": 0, "repeatability": 0}

    for image_index, image_paths in enumerate(image_paths_list, start=1):
        loc_id = loc_id_from_paths(image_paths)
        loc_meta = parse_loc_id(loc_id)
        chunk_start = time.time()
        log_step(f"=== [{image_index}/{len(image_paths_list)}] {loc_id} ===")

        base_args = build_eval_args(image_paths, config.cache_dir, config.n_jobs)
        log_step(f"Opening streaming source for {loc_id}")
        with src.evaluation.open_evaluation_source(image_paths, base_args) as prepared:
            source = prepared["source"]
            t_sec = prepared["t_sec"]
            t_days = prepared["t_days"]
            log_step(f"Source ready: shape=({prepared['meta']['count']}, {prepared['meta']['height']}, {prepared['meta']['width']})")
            log_step(f"Sampling random-point pool (max={config.max_random_points})")
            stats = get_pixel_stats(source, t_days, loc_id, base_args.min_obs, pixel_stats_cache)
            random_points_full = src.evaluation.sample_random_points_from_source(
                source,
                t_days,
                base_args.min_obs,
                config.max_random_points,
                seed=stable_seed(config.base_seed, loc_id, "random_pool"),
                precomputed_stats=stats,
            )
            log_step(f"Random-point pool ready: {len(random_points_full)} samples")

            log_step("Selecting gap candidates from relatively complete pixels only")
            gap_pixels_full, gap_candidate_total, gap_candidate_filtered = select_gap_pixels(
                source,
                t_days,
                min_obs=base_args.min_obs,
                num_samples=config.max_gap_samples,
                seed=stable_seed(config.base_seed, loc_id, "gap_pool"),
                max_missing_ratio=config.gap_max_missing_ratio,
                max_native_gap_days=config.gap_max_native_gap_days,
                loc_id=loc_id,
                pixel_stats_cache=pixel_stats_cache,
            )
            log_step(
                f"Gap candidates: total={gap_candidate_total}, filtered={gap_candidate_filtered}, sampled={len(gap_pixels_full)} "
                f"(missing_ratio<={config.gap_max_missing_ratio:.2f}, native_gap<={config.gap_max_native_gap_days}d)"
            )

            gap_specs_for_chunk, total_span_days = gap_days_from_index_targets(t_days, config.gap_index_targets)
            ablation_gap_days, _ = gap_days_for_index(t_days, config.ablation_gap_index)
            repeatability_gap_days, _ = gap_days_for_index(t_days, config.repeatability_gap_index)
            log_step(f"Gap span planning: total_span={total_span_days:.1f}d, targets={len(config.gap_index_targets)}, derived_specs={gap_specs_for_chunk}")
            log_step(
                f"Representative gap settings: ablation={ablation_gap_days}d (I~{config.ablation_gap_index:.2f}), "
                f"repeatability={repeatability_gap_days}d (I~{config.repeatability_gap_index:.2f})"
            )

            if len(random_points_full) == 0 or len(gap_pixels_full) == 0:
                log_step("Skipping chunk because no valid evaluation samples were found.")
                continue

            if config.run_ablation:
                summary["ablation"] += _run_ablation_stage(source, t_sec, t_days, image_paths, config, output_paths, base_args, loc_id, loc_meta, random_points_full, gap_pixels_full, ablation_done, ablation_gap_days)
            if config.run_sparse:
                summary["sparse"] += _run_sparse_stage(source, t_sec, t_days, config, output_paths, base_args, loc_id, loc_meta, random_points_full, sparse_done)
            if config.run_gap:
                summary["gap"] += _run_gap_stage(source, t_sec, t_days, config, output_paths, base_args, loc_id, loc_meta, gap_pixels_full, gap_done, gap_specs_for_chunk)
            if config.run_repeatability and loc_id in repeatability_targets:
                summary["repeatability"] += _run_repeatability_stage(source, t_sec, t_days, config, output_paths, base_args, loc_id, loc_meta, repeat_done, repeatability_gap_days, pixel_stats_cache)

        log_step(f"Chunk complete: {loc_id} in {time.time() - chunk_start:.1f}s")

    return {
        "status": "ok",
        "source_name": source_name,
        "image_count": len(image_paths_list),
        **summary,
        "output_paths": {key: str(path) for key, path in output_paths.items()},
    }


def _run_ablation_stage(source, t_sec, t_days, image_paths: Sequence[str], config: LocalEvalConfig, output_paths: Mapping[str, Path], base_args, loc_id: str, loc_meta: Mapping[str, object], random_points_full: np.ndarray, gap_pixels_full: np.ndarray, ablation_done: set[tuple], ablation_gap_days: int) -> int:
    rows_written = 0
    for variant in config.ablation_variants:
        variant_name = variant["name"]
        variant_args = build_eval_args(image_paths, config.cache_dir, config.n_jobs, variant["overrides"])
        random_key = (loc_id, "random", variant_name)
        if random_key not in ablation_done:
            stage_start = time.time()
            log_step(f"Ablation random start: {variant_name} ({config.ablation_random_points} points)")
            df_random = src.evaluation.evaluate_algorithms_from_source(source, t_sec, t_days, variant_args, sampled_points=random_points_full[: config.ablation_random_points], n_jobs=config.n_jobs)
            df_random = df_random[df_random["Algorithm"] == "NuFrost"].copy()
            df_random["Scenario"] = "random"
            df_random["Variant"] = variant_name
            append_rows(output_paths["ablation"], _add_loc_meta(df_random, loc_id, loc_meta))
            rows_written += len(df_random)
            log_step(f"Ablation random done: {variant_name} in {time.time() - stage_start:.1f}s")
            ablation_done.add(random_key)

        gap_key = (loc_id, "gap", variant_name)
        if gap_key not in ablation_done:
            stage_start = time.time()
            log_step(f"Ablation gap start: {variant_name} ({ablation_gap_days} days, {len(gap_pixels_full)} pixels, I~{config.ablation_gap_index:.2f})")
            on_batch, ckpt_path = make_batch_checkpoint_writer(output_paths["ablation"], loc_id=loc_id, scenario="gap", variant=variant_name, gap_length=ablation_gap_days, gap_index_target=config.ablation_gap_index, loc_meta=loc_meta)
            df_gap = src.evaluation.evaluate_timeseries_from_source(source, t_sec, t_days, variant_args, simulate_gap_days=ablation_gap_days, sampled_pixels=gap_pixels_full, n_jobs=config.n_jobs, on_batch_done=on_batch)
            df_gap = df_gap[df_gap["Algorithm"] == "NuFrost"].copy()
            df_gap["Scenario"] = "gap"
            df_gap["Variant"] = variant_name
            df_gap["GapLength"] = ablation_gap_days
            df_gap["GapIndexTarget"] = config.ablation_gap_index
            append_rows(output_paths["ablation"], _add_loc_meta(df_gap, loc_id, loc_meta))
            rows_written += len(df_gap)
            if ckpt_path.exists():
                ckpt_path.unlink()
            log_step(f"Ablation gap done: {variant_name} in {time.time() - stage_start:.1f}s")
            ablation_done.add(gap_key)

    baseline_random_key = (loc_id, "random", "Baselines")
    if baseline_random_key not in ablation_done:
        stage_start = time.time()
        log_step(f"Ablation random baseline start ({config.ablation_random_points} points)")
        df_baseline_random = src.evaluation.evaluate_algorithms_from_source(source, t_sec, t_days, base_args, sampled_points=random_points_full[: config.ablation_random_points], n_jobs=config.n_jobs)
        df_baseline_random = df_baseline_random[df_baseline_random["Algorithm"].isin(["Zhu2015", "HANTS"])].copy()
        df_baseline_random["Scenario"] = "random"
        df_baseline_random["Variant"] = df_baseline_random["Algorithm"]
        append_rows(output_paths["ablation"], _add_loc_meta(df_baseline_random, loc_id, loc_meta))
        rows_written += len(df_baseline_random)
        log_step(f"Ablation random baseline done in {time.time() - stage_start:.1f}s")
        ablation_done.add(baseline_random_key)

    baseline_gap_key = (loc_id, "gap", "Baselines")
    if baseline_gap_key not in ablation_done:
        stage_start = time.time()
        log_step(f"Ablation gap baseline start ({ablation_gap_days} days, {len(gap_pixels_full)} pixels, I~{config.ablation_gap_index:.2f})")
        on_batch, ckpt_path = make_batch_checkpoint_writer(output_paths["ablation"], loc_id=loc_id, scenario="gap", variant="Baselines", gap_length=ablation_gap_days, gap_index_target=config.ablation_gap_index, loc_meta=loc_meta)
        df_baseline_gap = src.evaluation.evaluate_timeseries_from_source(source, t_sec, t_days, base_args, simulate_gap_days=ablation_gap_days, sampled_pixels=gap_pixels_full, n_jobs=config.n_jobs, on_batch_done=on_batch)
        df_baseline_gap = df_baseline_gap[df_baseline_gap["Algorithm"].isin(["Zhu2015", "HANTS"])].copy()
        df_baseline_gap["Scenario"] = "gap"
        df_baseline_gap["Variant"] = df_baseline_gap["Algorithm"]
        df_baseline_gap["GapLength"] = ablation_gap_days
        df_baseline_gap["GapIndexTarget"] = config.ablation_gap_index
        append_rows(output_paths["ablation"], _add_loc_meta(df_baseline_gap, loc_id, loc_meta))
        rows_written += len(df_baseline_gap)
        if ckpt_path.exists():
            ckpt_path.unlink()
        log_step(f"Ablation gap baseline done in {time.time() - stage_start:.1f}s")
        ablation_done.add(baseline_gap_key)
    return rows_written


def _run_sparse_stage(source, t_sec, t_days, config: LocalEvalConfig, output_paths: Mapping[str, Path], base_args, loc_id: str, loc_meta: Mapping[str, object], random_points_full: np.ndarray, sparse_done: set[tuple]) -> int:
    rows_written = 0
    for num_points in config.sparse_point_levels:
        sparse_key = (loc_id, num_points)
        if sparse_key in sparse_done:
            continue
        stage_start = time.time()
        log_step(f"Sparse sweep start: {num_points} points")
        df_sparse = src.evaluation.evaluate_algorithms_from_source(source, t_sec, t_days, base_args, sampled_points=random_points_full[:num_points], n_jobs=config.n_jobs)
        df_sparse["NumPoints"] = num_points
        append_rows(output_paths["sparse"], _add_loc_meta(df_sparse, loc_id, loc_meta))
        rows_written += len(df_sparse)
        log_step(f"Sparse sweep done: {num_points} points in {time.time() - stage_start:.1f}s")
        sparse_done.add(sparse_key)
    return rows_written


def _run_gap_stage(source, t_sec, t_days, config: LocalEvalConfig, output_paths: Mapping[str, Path], base_args, loc_id: str, loc_meta: Mapping[str, object], gap_pixels_full: np.ndarray, gap_done: set[tuple], gap_specs_for_chunk: Sequence[Tuple[int, float]]) -> int:
    rows_written = 0
    for gap_days, gap_index_target in gap_specs_for_chunk:
        gap_key = (loc_id, gap_days)
        if gap_key in gap_done:
            continue
        stage_start = time.time()
        log_step(f"Gap sweep start: {gap_days} days on {len(gap_pixels_full)} pixels (target I~{gap_index_target:.2f})")
        on_batch, ckpt_path = make_batch_checkpoint_writer(output_paths["gap"], loc_id=loc_id, scenario="gap", variant=f"GapSweep_{gap_days}d", gap_length=gap_days, gap_index_target=gap_index_target, loc_meta=loc_meta)
        df_gap_sweep = src.evaluation.evaluate_timeseries_from_source(source, t_sec, t_days, base_args, simulate_gap_days=gap_days, sampled_pixels=gap_pixels_full, n_jobs=config.n_jobs, on_batch_done=on_batch)
        df_gap_sweep["GapLength"] = gap_days
        df_gap_sweep["GapIndexTarget"] = gap_index_target
        append_rows(output_paths["gap"], _add_loc_meta(df_gap_sweep, loc_id, loc_meta))
        rows_written += len(df_gap_sweep)
        if ckpt_path.exists():
            ckpt_path.unlink()
        log_step(f"Gap sweep done: {gap_days} days in {time.time() - stage_start:.1f}s")
        gap_done.add(gap_key)
    return rows_written


def _run_repeatability_stage(source, t_sec, t_days, config: LocalEvalConfig, output_paths: Mapping[str, Path], base_args, loc_id: str, loc_meta: Mapping[str, object], repeat_done: set[tuple], repeatability_gap_days: int, pixel_stats_cache: Dict[str, Dict[str, np.ndarray]]) -> int:
    rows_written = 0
    for repeat_seed in config.repeatability_seeds:
        repeat_random_key = (loc_id, "random", repeat_seed)
        if repeat_random_key not in repeat_done:
            stage_start = time.time()
            log_step(f"Repeatability random start: seed={repeat_seed}")
            sampled_points = src.evaluation.sample_random_points_from_source(source, t_days, base_args.min_obs, config.repeatability_random_points, seed=stable_seed(config.base_seed, loc_id, "repeat_random", repeat_seed))
            df_repeat_random = src.evaluation.evaluate_algorithms_from_source(source, t_sec, t_days, base_args, sampled_points=sampled_points, n_jobs=config.n_jobs)
            df_repeat_random["Scenario"] = "random"
            df_repeat_random["RepeatSeed"] = repeat_seed
            append_rows(output_paths["repeatability"], _add_loc_meta(df_repeat_random, loc_id, loc_meta))
            rows_written += len(df_repeat_random)
            log_step(f"Repeatability random done: seed={repeat_seed} in {time.time() - stage_start:.1f}s")
            repeat_done.add(repeat_random_key)

        repeat_gap_key = (loc_id, "gap", repeat_seed)
        if repeat_gap_key not in repeat_done:
            stage_start = time.time()
            log_step(f"Repeatability gap start: seed={repeat_seed}")
            sampled_pixels, _, filtered_count = select_gap_pixels(
                source,
                t_days,
                min_obs=base_args.min_obs,
                num_samples=config.repeatability_gap_samples,
                seed=stable_seed(config.base_seed, loc_id, "repeat_gap", repeat_seed),
                max_missing_ratio=config.gap_max_missing_ratio,
                max_native_gap_days=config.gap_max_native_gap_days,
                loc_id=loc_id,
                pixel_stats_cache=pixel_stats_cache,
            )
            log_step(f"Repeatability gap sampled {len(sampled_pixels)} pixels from {filtered_count} filtered candidates")
            on_batch, ckpt_path = make_batch_checkpoint_writer(output_paths["repeatability"], loc_id=loc_id, scenario="gap", variant=f"Repeat_seed{repeat_seed}", gap_length=repeatability_gap_days, gap_index_target=config.repeatability_gap_index, loc_meta=loc_meta)
            df_repeat_gap = src.evaluation.evaluate_timeseries_from_source(source, t_sec, t_days, base_args, simulate_gap_days=repeatability_gap_days, sampled_pixels=sampled_pixels, n_jobs=config.n_jobs, on_batch_done=on_batch)
            df_repeat_gap["Scenario"] = "gap"
            df_repeat_gap["RepeatSeed"] = repeat_seed
            df_repeat_gap["GapLength"] = repeatability_gap_days
            df_repeat_gap["GapIndexTarget"] = config.repeatability_gap_index
            append_rows(output_paths["repeatability"], _add_loc_meta(df_repeat_gap, loc_id, loc_meta))
            rows_written += len(df_repeat_gap)
            if ckpt_path.exists():
                ckpt_path.unlink()
            log_step(f"Repeatability gap done: seed={repeat_seed} in {time.time() - stage_start:.1f}s")
            repeat_done.add(repeat_gap_key)
    return rows_written
