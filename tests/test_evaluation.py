import numpy as np

from config import build_args
from src.evaluation import (
    evaluate_algorithms_from_source,
    evaluate_algorithms_on_cube,
    evaluate_timeseries_from_source,
    evaluate_timeseries_on_cube,
    load_evaluation_cube,
    open_evaluation_source,
    sample_gap_pixels_from_source,
    sample_gap_pixels,
    sample_random_points_from_source,
    sample_random_points,
)


def test_load_evaluation_cube_and_sampling_helpers(single_tile_path: str, cache_dir) -> None:
    args = build_args({"cache_dir": cache_dir, "force_refresh": False})
    prepared = load_evaluation_cube([single_tile_path], args)

    assert prepared["cube"].ndim == 3
    assert prepared["cube"].shape[0] == len(prepared["timestamps"])

    points = sample_random_points(prepared["cube"], prepared["t_days"], min_obs=6, num_points=20, seed=123)
    pixels = sample_gap_pixels(prepared["cube"], prepared["t_days"], min_obs=6, num_samples=10, seed=123)

    assert points.ndim == 2 and points.shape[1] == 3
    assert pixels.ndim == 2 and pixels.shape[1] == 2


def test_load_evaluation_cube_uses_split_cache_layout(fixture_input_dir, tmp_path) -> None:
    from src.data_loader import find_image_chunks

    cache_root = tmp_path / "cache" / "local"
    chunks = find_image_chunks(fixture_input_dir.as_posix(), lon=100.112, lat=25.654, band="B2", cache_dir=cache_root)
    args = build_args({"cache_dir": cache_root, "force_refresh": True})

    prepared = load_evaluation_cube(chunks, args)

    assert prepared["cube"].ndim == 3
    assert (cache_root / "vrts").exists()
    assert (cache_root / "npz").exists()


def test_evaluate_algorithms_on_cube_smoke(single_tile_path: str, cache_dir) -> None:
    args = build_args({"cache_dir": cache_dir, "force_refresh": False, "min_obs": 6, "n_jobs": 1})
    prepared = load_evaluation_cube([single_tile_path], args)
    sampled_points = sample_random_points(prepared["cube"], prepared["t_days"], min_obs=args.min_obs, num_points=10, seed=123)

    df = evaluate_algorithms_on_cube(
        prepared["cube"],
        prepared["t_sec"],
        prepared["t_days"],
        args,
        sampled_points=sampled_points,
        n_jobs=1,
    )

    assert set(df["Algorithm"]) == {"NuFrost", "Zhu2015", "HANTS"}
    assert np.isfinite(df["RMSE"]).all()


def test_evaluate_timeseries_on_cube_smoke(single_tile_path: str, cache_dir) -> None:
    args = build_args({"cache_dir": cache_dir, "force_refresh": False, "min_obs": 6, "n_jobs": 1})
    prepared = load_evaluation_cube([single_tile_path], args)
    sampled_pixels = sample_gap_pixels(prepared["cube"], prepared["t_days"], min_obs=args.min_obs, num_samples=5, seed=123)

    df = evaluate_timeseries_on_cube(
        prepared["cube"],
        prepared["t_sec"],
        prepared["t_days"],
        args,
        simulate_gap_days=30,
        sampled_pixels=sampled_pixels,
        n_jobs=1,
    )

    assert set(df["Algorithm"]) == {"NuFrost", "Zhu2015", "HANTS"}
    assert np.isfinite(df["MAE"]).all()


def test_streaming_evaluation_path_avoids_npz(single_tile_path: str, cache_dir) -> None:
    args = build_args({"cache_dir": cache_dir, "force_refresh": False, "min_obs": 6, "n_jobs": 1})

    with open_evaluation_source([single_tile_path], args) as prepared:
        sampled_points = sample_random_points_from_source(prepared["source"], prepared["t_days"], min_obs=args.min_obs, num_points=10, seed=123)
        sampled_pixels = sample_gap_pixels_from_source(prepared["source"], prepared["t_days"], min_obs=args.min_obs, num_samples=5, seed=123)

        df_random = evaluate_algorithms_from_source(
            prepared["source"],
            prepared["t_sec"],
            prepared["t_days"],
            args,
            sampled_points=sampled_points,
            n_jobs=1,
        )
        df_gap = evaluate_timeseries_from_source(
            prepared["source"],
            prepared["t_sec"],
            prepared["t_days"],
            args,
            simulate_gap_days=30,
            sampled_pixels=sampled_pixels,
            n_jobs=1,
        )

    assert set(df_random["Algorithm"]) == {"NuFrost", "Zhu2015", "HANTS"}
    assert set(df_gap["Algorithm"]) == {"NuFrost", "Zhu2015", "HANTS"}
    assert not any((cache_dir / "npz").glob("*.npz"))


def test_evaluate_algorithms_from_source_parallel(single_tile_path: str, cache_dir) -> None:
    args = build_args({"cache_dir": cache_dir, "force_refresh": False, "min_obs": 6, "n_jobs": 2})
    with open_evaluation_source([single_tile_path], args) as prepared:
        sampled_points = sample_random_points_from_source(
            prepared["source"], prepared["t_days"], min_obs=args.min_obs, num_points=20, seed=123
        )
        df = evaluate_algorithms_from_source(
            prepared["source"],
            prepared["t_sec"],
            prepared["t_days"],
            args,
            sampled_points=sampled_points,
            n_jobs=2,
        )
    assert set(df["Algorithm"]) == {"NuFrost", "Zhu2015", "HANTS"}
    assert np.isfinite(df["RMSE"]).all()


def test_evaluate_timeseries_from_source_parallel(single_tile_path: str, cache_dir) -> None:
    args = build_args({"cache_dir": cache_dir, "force_refresh": False, "min_obs": 6, "n_jobs": 2})
    with open_evaluation_source([single_tile_path], args) as prepared:
        sampled_pixels = sample_gap_pixels_from_source(
            prepared["source"], prepared["t_days"], min_obs=args.min_obs, num_samples=10, seed=123
        )
        df = evaluate_timeseries_from_source(
            prepared["source"],
            prepared["t_sec"],
            prepared["t_days"],
            args,
            simulate_gap_days=30,
            sampled_pixels=sampled_pixels,
            n_jobs=2,
        )
    assert set(df["Algorithm"]) == {"NuFrost", "Zhu2015", "HANTS"}
    assert np.isfinite(df["MAE"]).all()
