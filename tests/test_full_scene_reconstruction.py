import importlib.util
from pathlib import Path

import numpy as np
import rasterio

from src.full_scene_reconstruction import (
    build_ground_truth_output_path,
    build_output_path,
    build_scene_stack_output_path,
    collapse_duplicate_timestamps,
    discover_available_locations,
    discover_location_band_stacks,
    extract_prediction_2d,
    intersect_band_timestamps,
    make_masked_time_series,
    reconstruct_full_scene_for_all_locations,
    select_shared_target_timestamp,
    write_band_stack,
    write_run_summary,
    validate_band_metadata_consistency,
)


def test_intersect_band_timestamps_returns_shared_sorted_values() -> None:
    band_to_timestamps = {
        "B2": ["2024-01-01T00:00:00", "2024-02-01T00:00:00", "2024-03-01T00:00:00"],
        "B3": ["2024-02-01T00:00:00", "2024-03-01T00:00:00", "2024-04-01T00:00:00"],
        "B4": ["2024-01-15T00:00:00", "2024-02-01T00:00:00", "2024-03-01T00:00:00"],
    }

    shared = intersect_band_timestamps(band_to_timestamps)

    assert shared == ["2024-02-01T00:00:00", "2024-03-01T00:00:00"]


def test_select_shared_target_timestamp_prefers_latest_complete_candidate() -> None:
    candidates = [
        "2024-02-01T00:00:00",
        "2024-03-01T00:00:00",
        "2024-04-01T00:00:00",
        "2024-05-01T00:00:00",
    ]
    completeness = {
        "B2": {
            "2024-02-01T00:00:00": 0.95,
            "2024-03-01T00:00:00": 0.92,
            "2024-04-01T00:00:00": 0.40,
            "2024-05-01T00:00:00": 0.91,
        },
        "B3": {
            "2024-02-01T00:00:00": 0.95,
            "2024-03-01T00:00:00": 0.93,
            "2024-04-01T00:00:00": 0.93,
            "2024-05-01T00:00:00": 0.15,
        },
    }

    chosen = select_shared_target_timestamp(
        candidates,
        completeness,
        min_valid_ratio=0.90,
        late_fraction=0.50,
    )

    assert chosen == "2024-03-01T00:00:00"


def test_discover_location_band_stacks_for_sentinel2_groups_by_band(tmp_path: Path) -> None:
    data_dir = tmp_path / "sentinel-2"
    data_dir.mkdir()
    (data_dir / "COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733.tif").touch()
    (data_dir / "COPERNICUS_S2_HARMONIZED_B3_lon94.2605_lat29.7733.tif").touch()
    (data_dir / "COPERNICUS_S2_HARMONIZED_B4_lon94.2605_lat29.7733.tif").touch()

    stacks = discover_location_band_stacks(data_dir, source_name="sentinel-2", lon=94.2605, lat=29.7733)

    assert sorted(stacks) == ["B2", "B3", "B4"]
    assert stacks["B2"][0].name == "COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733.tif"


def test_discover_location_band_stacks_for_sentinel2_uses_vrt_for_multi_tile_band(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "sentinel-2"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    cache_dir.mkdir()
    (data_dir / "COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733-0000000000-0000000000.tif").touch()
    (data_dir / "COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733-0000000000-0000001024.tif").touch()

    def fake_build_multi_file_vrt(stack_paths, *, cache_dir, band_name, lon, lat):
        return [tmp_path / f"sentinel_{band_name}_lon{lon:.4f}_lat{lat:.4f}.vrt"]

    monkeypatch.setattr("src.full_scene_reconstruction.pipeline._build_multi_file_vrt", fake_build_multi_file_vrt)

    stacks = discover_location_band_stacks(data_dir, source_name="sentinel-2", lon=94.2605, lat=29.7733, cache_dir=cache_dir)

    assert len(stacks["B2"]) == 1
    assert stacks["B2"][0].suffix == ".vrt"


def test_discover_location_band_stacks_for_hls_groups_by_band(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "hls"
    data_dir.mkdir()
    (data_dir / "NASA_HLS_v002_BLUE_lon94.2605_lat29.7733_part1-0000000000-0000000000.tif").touch()
    (data_dir / "NASA_HLS_v002_BLUE_lon94.2605_lat29.7733_part1-0000000000-0000000512.tif").touch()
    (data_dir / "NASA_HLS_v002_RED_lon94.2605_lat29.7733_part1-0000000000-0000000000.tif").touch()

    def fake_find_image_chunks(data_dir_str: str, lon: float, lat: float, band: str, cache_dir=None):
        return [f"{data_dir_str}/{band}_lon{lon:.4f}_lat{lat:.4f}.vrt"]

    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.find_image_chunks", fake_find_image_chunks)

    stacks = discover_location_band_stacks(data_dir, source_name="hls", lon=94.2605, lat=29.7733)

    assert sorted(stacks) == ["BLUE", "RED"]
    assert stacks["BLUE"][0].name == "BLUE_lon94.2605_lat29.7733.vrt"


def test_discover_available_locations_for_sentinel2_returns_sorted_unique_pairs(tmp_path: Path) -> None:
    data_dir = tmp_path / "sentinel-2"
    data_dir.mkdir()
    (data_dir / "COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733.tif").touch()
    (data_dir / "COPERNICUS_S2_HARMONIZED_B3_lon94.2605_lat29.7733.tif").touch()
    (data_dir / "COPERNICUS_S2_HARMONIZED_B2_lon91.2734_lat29.7904.tif").touch()

    locations = discover_available_locations(data_dir, source_name="sentinel-2")

    assert locations == [(91.2734, 29.7904), (94.2605, 29.7733)]


def test_discover_available_locations_for_hls_returns_sorted_unique_pairs(tmp_path: Path) -> None:
    data_dir = tmp_path / "hls"
    data_dir.mkdir()
    (data_dir / "NASA_HLS_v002_BLUE_lon94.2605_lat29.7733_part1-0000000000-0000000000.tif").touch()
    (data_dir / "NASA_HLS_v002_RED_lon94.2605_lat29.7733_part1-0000000000-0000000000.tif").touch()
    (data_dir / "NASA_HLS_v002_BLUE_lon91.2734_lat29.7904_part1-0000000000-0000000000.tif").touch()

    locations = discover_available_locations(data_dir, source_name="hls")

    assert locations == [(91.2734, 29.7904), (94.2605, 29.7733)]


def test_build_output_path_includes_timestamp_and_method(tmp_path: Path) -> None:
    output_path = build_output_path(
        output_root=tmp_path,
        method_name="nufrost",
        source_file=Path("COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733.tif"),
        target_time="2026-01-27T04:19:39",
        source_name="sentinel-2",
        lon=94.2605,
        lat=29.7733,
    )

    assert output_path.parent == tmp_path / "sentinel-2_recon" / "94.2605_29.7733"
    assert output_path.name == "[nufrost]_sentinel-2_lon94.260500_lat29.773300_2026-01-27T04-19-39.tif"


def test_build_ground_truth_output_path(tmp_path: Path) -> None:
    gt_path = build_ground_truth_output_path(
        output_root=tmp_path,
        source_name="sentinel-2",
        lon=94.2605,
        lat=29.7733,
        target_time="2026-01-27T04:19:39",
    )
    assert gt_path.parent == tmp_path / "sentinel-2_recon" / "94.2605_29.7733"
    assert gt_path.name == "[ground_truth]_sentinel-2_lon94.260500_lat29.773300_2026-01-27T04-19-39.tif"


def test_extract_prediction_2d_handles_nufrost_and_zhu2015() -> None:
    nufrost_pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    zhu_pred = np.array(
        [[[5.0, 6.0], [7.0, 8.0]], [[99.0, 99.0], [99.0, 99.0]]],
        dtype=np.float32,
    )

    assert np.array_equal(extract_prediction_2d("nufrost", nufrost_pred), nufrost_pred)
    assert np.array_equal(extract_prediction_2d("zhu2015", zhu_pred), zhu_pred[0])


def test_write_band_stack_writes_ordered_multiband_geotiff(tmp_path: Path) -> None:
    stack_path = build_scene_stack_output_path(
        output_root=tmp_path,
        method_name="nufrost",
        source_name="sentinel-2",
        lon=94.2605,
        lat=29.7733,
        target_time="2026-01-27T04:19:39",
        suffix="prediction",
    )
    meta = {
        "transform": (1.0, 0.0, 10.0, 0.0, -1.0, 20.0, 0.0, 0.0, 1.0),
        "crs_wkt": None,
    }
    arrays = {
        "B4": np.full((2, 2), 4.0, dtype=np.float32),
        "B2": np.full((2, 2), 2.0, dtype=np.float32),
        "B3": np.full((2, 2), 3.0, dtype=np.float32),
    }

    write_band_stack(stack_path, arrays, ["B2", "B3", "B4"], meta)

    with rasterio.open(stack_path) as ds:
        assert ds.count == 3
        assert ds.descriptions == ("B2", "B3", "B4")
        assert ds.read(1)[0, 0] == 2.0
        assert ds.read(2)[0, 0] == 3.0
        assert ds.read(3)[0, 0] == 4.0


def test_build_scene_stack_output_path_distinguishes_qa_stack_name(tmp_path: Path) -> None:
    output_path = build_scene_stack_output_path(
        output_root=tmp_path,
        method_name="nufrost",
        source_name="sentinel-2",
        lon=94.2605,
        lat=29.7733,
        target_time="2026-01-27T04:19:39",
        suffix="QA_stack",
    )

    assert output_path.parent == tmp_path / "sentinel-2_recon" / "94.2605_29.7733"
    assert output_path.name == "[nufrost]_sentinel-2_lon94.260500_lat29.773300_2026-01-27T04-19-39_QA_stack.tif"


def test_reconstruct_full_scene_dispatches_methods_with_shared_job_budget(tmp_path: Path, monkeypatch) -> None:
    from src.full_scene_reconstruction import reconstruct_full_scene_for_location

    calls = []

    def fake_discover_location_band_stacks(*args, **kwargs):
        return {"B2": [tmp_path / "B2.vrt"]}

    def fake_choose_shared_target_timestamp(*args, **kwargs):
        return "2024-02-01T00:00:00", {"B2": {"2024-02-01T00:00:00": 1.0}}

    class FakeRSCube:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return {
                "cube": np.ma.array(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ],
                    dtype=np.float32,
                ),
                "timestamps": np.asarray(["2024-01-01T00:00:00", "2024-02-01T00:00:00"], dtype="U32"),
                "transform": (1.0, 0.0, 10.0, 0.0, -1.0, 20.0, 0.0, 0.0, 1.0),
                "crs_wkt": None,
            }

    def fake_nufrost_core(cube, timestamps, target_time, args=None, **kwargs):
        calls.append(("nufrost", cube.shape, target_time))
        return np.full((cube.shape[1], cube.shape[2]), 10.0, dtype=np.float32)

    def fake_reconstruct_hants(cube, timestamps, target_time, **kwargs):
        calls.append(("hants", cube.shape, target_time))
        return np.full((cube.shape[1], cube.shape[2]), 20.0, dtype=np.float32)

    def fake_reconstruct_zhu2015(cube, timestamps, target_time, **kwargs):
        calls.append(("zhu2015", cube.shape, target_time))
        pred = np.full((cube.shape[1], cube.shape[2]), 30.0, dtype=np.float32)
        qa = np.full((cube.shape[1], cube.shape[2]), 0.0, dtype=np.float32)
        return np.stack([pred, qa], axis=0)

    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.discover_location_band_stacks", fake_discover_location_band_stacks)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.choose_shared_target_timestamp", fake_choose_shared_target_timestamp)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.RSCube", FakeRSCube)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.nufrost_core", fake_nufrost_core)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.reconstruct_hants_from_cube", fake_reconstruct_hants)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.reconstruct_zhu2015_from_cube", fake_reconstruct_zhu2015)

    result = reconstruct_full_scene_for_location(
        source_name="sentinel-2",
        lon=94.2605,
        lat=29.7733,
        output_root=tmp_path,
        data_root=tmp_path,
        n_jobs=9,
    )

    method_names = [c[0] for c in calls]
    assert method_names == ["nufrost", "hants", "zhu2015"]
    assert result["ground_truth_output"].endswith("[ground_truth]_sentinel-2_lon94.260500_lat29.773300_2024-02-01T00-00-00.tif")
    assert result["merged_prediction_outputs"]["nufrost"].endswith("[nufrost]_sentinel-2_lon94.260500_lat29.773300_2024-02-01T00-00-00_prediction.tif")


def test_reconstruct_full_scene_passes_limited_budget_into_shared_scheduler(tmp_path: Path, monkeypatch) -> None:
    from src.full_scene_reconstruction import reconstruct_full_scene_for_location

    calls = []

    def fake_discover_location_band_stacks(*args, **kwargs):
        return {"B2": [tmp_path / "B2.vrt"]}

    def fake_choose_shared_target_timestamp(*args, **kwargs):
        return "2024-02-01T00:00:00", {"B2": {"2024-02-01T00:00:00": 1.0}}

    class FakeRSCube:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return {
                "cube": np.ma.array(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ],
                    dtype=np.float32,
                ),
                "timestamps": np.asarray(["2024-01-01T00:00:00", "2024-02-01T00:00:00"], dtype="U32"),
                "transform": (1.0, 0.0, 10.0, 0.0, -1.0, 20.0, 0.0, 0.0, 1.0),
                "crs_wkt": None,
            }

    def fake_nufrost_core(cube, timestamps, target_time, args=None, **kwargs):
        calls.append(("nufrost", kwargs.get("n_jobs", getattr(args, "n_jobs", -1))))
        return np.full((cube.shape[1], cube.shape[2]), 10.0, dtype=np.float32)

    def fake_reconstruct_hants(cube, timestamps, target_time, **kwargs):
        calls.append(("hants", kwargs.get("n_jobs", -1)))
        return np.full((cube.shape[1], cube.shape[2]), 20.0, dtype=np.float32)

    def fake_reconstruct_zhu2015(cube, timestamps, target_time, **kwargs):
        calls.append(("zhu2015", kwargs.get("n_jobs", -1)))
        pred = np.full((cube.shape[1], cube.shape[2]), 30.0, dtype=np.float32)
        qa = np.full((cube.shape[1], cube.shape[2]), 0.0, dtype=np.float32)
        return np.stack([pred, qa], axis=0)

    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.discover_location_band_stacks", fake_discover_location_band_stacks)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.choose_shared_target_timestamp", fake_choose_shared_target_timestamp)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.RSCube", FakeRSCube)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.nufrost_core", fake_nufrost_core)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.reconstruct_hants_from_cube", fake_reconstruct_hants)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.reconstruct_zhu2015_from_cube", fake_reconstruct_zhu2015)

    reconstruct_full_scene_for_location(
        source_name="sentinel-2",
        lon=94.2605,
        lat=29.7733,
        output_root=tmp_path,
        data_root=tmp_path,
        n_jobs=2,
    )

    method_names = [c[0] for c in calls]
    assert method_names == ["nufrost", "hants", "zhu2015"]


def test_reconstruct_full_scene_skips_when_summary_exists(tmp_path: Path, monkeypatch) -> None:
    from src.full_scene_reconstruction import reconstruct_full_scene_for_location

    source_name = "sentinel-2"
    lon = 94.2605
    lat = 29.7733
    safe_source = source_name.replace("/", "-")
    summary_dir = tmp_path / "run_summaries"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / f"reconstruction_summary_{safe_source}_lon{lon:.6f}_lat{lat:.6f}_2026-02-06T04-18-39.json"
    summary_path.write_text(
        '{"source":"sentinel-2","lon":94.2605,"lat":29.7733,"target_time":"2026-02-06T04:18:39","methods":["nufrost","hants","zhu2015"],"window_size":null,"source_files":{"B2":["'
        + str(tmp_path / "B2.vrt")
        + '"]},"min_valid_ratio":0.9,"late_fraction":0.25}',
        encoding="utf-8",
    )

    def fake_discover_location_band_stacks(*args, **kwargs):
        return {"B2": [tmp_path / "B2.vrt"]}

    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.discover_location_band_stacks", fake_discover_location_band_stacks)

    result = reconstruct_full_scene_for_location(
        source_name=source_name,
        lon=lon,
        lat=lat,
        output_root=tmp_path,
        data_root=tmp_path / "does-not-matter",
    )

    assert result["skipped"] is True
    assert result["summary_path"] == str(summary_path)


def test_reconstruct_full_scene_does_not_skip_when_window_size_differs(tmp_path: Path, monkeypatch) -> None:
    from src.full_scene_reconstruction import reconstruct_full_scene_for_location

    summary_dir = tmp_path / "run_summaries"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "reconstruction_summary_sentinel-2_lon94.260500_lat29.773300_2025-01-01T00-00-00.json"
    summary_path.write_text(
        '{"source":"sentinel-2","lon":94.2605,"lat":29.7733,"target_time":"2025-01-01T00:00:00","window_size":8,"methods":["hants"]}',
        encoding="utf-8",
    )

    def fake_discover_location_band_stacks(*args, **kwargs):
        return {"B2": [tmp_path / "B2.vrt"]}

    def fake_choose_shared_target_timestamp(*args, **kwargs):
        return "2025-01-01T00:00:00", {"B2": {"2025-01-01T00:00:00": 1.0}}

    class FakeRSCube:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return {
                "cube": np.ma.array(
                    [
                        np.full((16, 16), 1.0, dtype=np.float32),
                        np.full((16, 16), 2.0, dtype=np.float32),
                    ]
                ),
                "timestamps": np.asarray(["2024-01-01T00:00:00", "2025-01-01T00:00:00"], dtype="U32"),
                "transform": (30.0, 0.0, 10.0, 0.0, -30.0, 20.0, 0.0, 0.0, 1.0),
                "crs_wkt": None,
            }

    def fake_nufrost_core(cube, timestamps, target_time, args=None, **kwargs):
        return np.full((cube.shape[1], cube.shape[2]), 7.0, dtype=np.float32)

    def fake_reconstruct_hants(cube, timestamps, target_time, **kwargs):
        return np.full((cube.shape[1], cube.shape[2]), 7.0, dtype=np.float32)

    def fake_reconstruct_zhu2015(cube, timestamps, target_time, **kwargs):
        pred = np.full((cube.shape[1], cube.shape[2]), 7.0, dtype=np.float32)
        qa = np.full((cube.shape[1], cube.shape[2]), 0.0, dtype=np.float32)
        return np.stack([pred, qa], axis=0)

    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.discover_location_band_stacks", fake_discover_location_band_stacks)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.choose_shared_target_timestamp", fake_choose_shared_target_timestamp)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.RSCube", FakeRSCube)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.nufrost_core", fake_nufrost_core)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.reconstruct_hants_from_cube", fake_reconstruct_hants)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.reconstruct_zhu2015_from_cube", fake_reconstruct_zhu2015)

    result = reconstruct_full_scene_for_location(
        source_name="sentinel-2",
        lon=94.2605,
        lat=29.7733,
        output_root=tmp_path,
        data_root=tmp_path,
        methods=("hants",),
        window_size=4,
        n_jobs=1,
    )

    assert result.get("skipped") is not True
    assert result["window_size"] == 4


def test_reconstruct_full_scene_does_not_skip_when_source_files_differ(tmp_path: Path, monkeypatch) -> None:
    from src.full_scene_reconstruction import reconstruct_full_scene_for_location

    summary_dir = tmp_path / "run_summaries"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "reconstruction_summary_sentinel-2_lon94.260500_lat29.773300_2025-01-01T00-00-00.json"
    summary_path.write_text(
        '{"source":"sentinel-2","lon":94.2605,"lat":29.7733,"target_time":"2025-01-01T00:00:00","window_size":4,"methods":["hants"],"source_files":{"B2":["/old/B2.vrt"]}}',
        encoding="utf-8",
    )

    def fake_discover_location_band_stacks(*args, **kwargs):
        return {"B2": [tmp_path / "B2-new.vrt"]}

    def fake_choose_shared_target_timestamp(*args, **kwargs):
        return "2025-01-01T00:00:00", {"B2": {"2025-01-01T00:00:00": 1.0}}

    class FakeRSCube:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return {
                "cube": np.ma.array(
                    [
                        np.full((8, 8), 1.0, dtype=np.float32),
                        np.full((8, 8), 2.0, dtype=np.float32),
                    ]
                ),
                "timestamps": np.asarray(["2024-01-01T00:00:00", "2025-01-01T00:00:00"], dtype="U32"),
                "transform": (30.0, 0.0, 10.0, 0.0, -30.0, 20.0, 0.0, 0.0, 1.0),
                "crs_wkt": None,
            }

    def fake_nufrost_core(cube, timestamps, target_time, args=None, **kwargs):
        return np.full((cube.shape[1], cube.shape[2]), 5.0, dtype=np.float32)

    def fake_reconstruct_hants(cube, timestamps, target_time, **kwargs):
        return np.full((cube.shape[1], cube.shape[2]), 5.0, dtype=np.float32)

    def fake_reconstruct_zhu2015(cube, timestamps, target_time, **kwargs):
        pred = np.full((cube.shape[1], cube.shape[2]), 5.0, dtype=np.float32)
        qa = np.full((cube.shape[1], cube.shape[2]), 0.0, dtype=np.float32)
        return np.stack([pred, qa], axis=0)

    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.discover_location_band_stacks", fake_discover_location_band_stacks)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.choose_shared_target_timestamp", fake_choose_shared_target_timestamp)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.RSCube", FakeRSCube)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.nufrost_core", fake_nufrost_core)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.reconstruct_hants_from_cube", fake_reconstruct_hants)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.reconstruct_zhu2015_from_cube", fake_reconstruct_zhu2015)

    result = reconstruct_full_scene_for_location(
        source_name="sentinel-2",
        lon=94.2605,
        lat=29.7733,
        output_root=tmp_path,
        data_root=tmp_path,
        methods=("hants",),
        window_size=4,
        n_jobs=1,
    )

    assert result.get("skipped") is not True
    assert result["source_files"]["B2"] == [str(tmp_path / "B2-new.vrt")]


def test_validate_band_metadata_consistency_rejects_mismatched_shapes() -> None:
    band_meta = {
        "B2": {"transform": (1, 0, 0, 0, -1, 0, 0, 0, 1), "crs_wkt": None, "cube": np.zeros((3, 2, 2), dtype=np.float32)},
        "B3": {"transform": (1, 0, 0, 0, -1, 0, 0, 0, 1), "crs_wkt": None, "cube": np.zeros((3, 3, 2), dtype=np.float32)},
    }

    try:
        validate_band_metadata_consistency(["B2", "B3"], band_meta)
    except ValueError as exc:
        assert "Band metadata mismatch" in str(exc)
    else:
        raise AssertionError("Expected metadata mismatch to raise ValueError")


def test_make_masked_time_series_removes_selected_timestamp_once() -> None:
    timestamps = ["2024-01-01T00:00:00", "2024-02-01T00:00:00", "2024-03-01T00:00:00"]
    cube = __import__("numpy").zeros((3, 2, 2), dtype="float32")

    masked_cube, masked_timestamps, target_idx = make_masked_time_series(cube, timestamps, "2024-02-01T00:00:00")

    assert masked_cube.shape == (2, 2, 2)
    assert masked_timestamps.tolist() == ["2024-01-01T00:00:00", "2024-03-01T00:00:00"]
    assert target_idx == 1


def test_collapse_duplicate_timestamps_merges_same_time_slices() -> None:
    cube = __import__("numpy").array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype="float32",
    )
    timestamps = ["2024-01-01T00:00:00", "2024-02-01T00:00:00", "2024-02-01T00:00:00"]

    merged_cube, merged_timestamps = collapse_duplicate_timestamps(cube, timestamps)

    assert merged_cube.shape == (2, 2, 2)
    assert merged_timestamps.tolist() == ["2024-01-01T00:00:00", "2024-02-01T00:00:00"]
    assert float(merged_cube[1, 0, 0]) == 7.0
    assert float(merged_cube[1, 1, 1]) == 10.0


def test_write_run_summary_persists_selected_timestamp(tmp_path: Path) -> None:
    summary_path = write_run_summary(
        output_root=tmp_path,
        payload={
            "source": "sentinel-2",
            "lon": 94.2605,
            "lat": 29.7733,
            "target_time": "2026-01-27T04:19:39",
            "bands": ["B2", "B3", "B4"],
        },
    )

    assert summary_path.exists()
    assert "sentinel-2_lon94.260500_lat29.773300" in summary_path.name
    assert "2026-01-27T04:19:39" in summary_path.read_text()


def test_small_window_full_scene_run_writes_8x8_outputs(tmp_path: Path, monkeypatch) -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_small_window_full_scene.py"
    spec = importlib.util.spec_from_file_location("run_small_window_full_scene", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script module from {script_path}")
    script_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_module)

    timestamps = np.asarray(
        [
            "2024-01-01T00:00:00",
            "2024-02-01T00:00:00",
            "2024-03-01T00:00:00",
            "2024-04-01T00:00:00",
            "2024-05-01T00:00:00",
            "2024-06-01T00:00:00",
            "2024-07-01T00:00:00",
            "2024-08-01T00:00:00",
            "2024-09-01T00:00:00",
            "2024-10-01T00:00:00",
            "2024-11-01T00:00:00",
            "2024-12-01T00:00:00",
            "2025-01-01T00:00:00",
        ],
        dtype="U32",
    )
    cube = np.stack(
        [np.full((16, 16), float(idx + 1), dtype=np.float32) for idx in range(len(timestamps))],
        axis=0,
    )

    def fake_discover_location_band_stacks(*args, **kwargs):
        return {"B2": [tmp_path / "B2.vrt"]}

    def fake_choose_shared_target_timestamp(*args, **kwargs):
        return str(timestamps[-1]), {"B2": {str(timestamps[-1]): 1.0}}

    class FakeRSCube:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return {
                "cube": np.ma.array(cube, dtype=np.float32),
                "timestamps": timestamps,
                "transform": (30.0, 0.0, 10.0, 0.0, -30.0, 20.0, 0.0, 0.0, 1.0),
                "crs_wkt": None,
            }

    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.discover_location_band_stacks", fake_discover_location_band_stacks)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.choose_shared_target_timestamp", fake_choose_shared_target_timestamp)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.RSCube", FakeRSCube)

    exit_code = script_module.main(
        [
            "--source-name",
            "sentinel-2",
            "--lon",
            "94.2605",
            "--lat",
            "29.7733",
            "--output-root",
            str(tmp_path / "output"),
            "--data-root",
            str(tmp_path),
            "--window-size",
            "8",
            "--methods",
            "hants",
            "--n-jobs",
            "1",
        ]
    )

    assert exit_code == 0

    summary_dir = tmp_path / "output" / "run_summaries"
    summary_paths = list(summary_dir.glob("*.json"))
    assert len(summary_paths) == 1

    with rasterio.open(tmp_path / "output" / "sentinel-2_recon" / "94.2605_29.7733" / "[ground_truth]_sentinel-2_lon94.260500_lat29.773300_2025-01-01T00-00-00.tif") as ds:
        assert ds.height == 8
        assert ds.width == 8

    with rasterio.open(tmp_path / "output" / "sentinel-2_recon" / "94.2605_29.7733" / "[hants]_sentinel-2_lon94.260500_lat29.773300_2025-01-01T00-00-00_prediction.tif") as ds:
        assert ds.height == 8
        assert ds.width == 8


def test_reconstruct_hants_from_cube_executes_on_small_cube() -> None:
    from src.full_scene_reconstruction.pipeline import reconstruct_hants_from_cube

    timestamps = np.asarray(
        [
            "2024-01-01T00:00:00",
            "2024-02-01T00:00:00",
            "2024-03-01T00:00:00",
            "2024-04-01T00:00:00",
            "2024-05-01T00:00:00",
            "2024-06-01T00:00:00",
            "2024-07-01T00:00:00",
            "2024-08-01T00:00:00",
            "2024-09-01T00:00:00",
            "2024-10-01T00:00:00",
            "2024-11-01T00:00:00",
            "2024-12-01T00:00:00",
        ],
        dtype="U32",
    )
    cube = np.stack(
        [np.full((2, 2), 0.2 + 0.01 * idx, dtype=np.float32) for idx in range(len(timestamps))],
        axis=0,
    )

    output = reconstruct_hants_from_cube(cube, timestamps, "2024-12-01T00:00:00", n_jobs=1)

    assert output.shape == (2, 2)
    assert np.isfinite(output).all()


def test_reconstruct_full_scene_for_all_locations_dispatches_every_coordinate(tmp_path: Path, monkeypatch) -> None:
    calls = []
    (tmp_path / "sentinel-2").mkdir()

    def fake_discover_available_locations(data_dir, source_name):
        assert data_dir == tmp_path / "sentinel-2"
        assert source_name == "sentinel-2"
        return [(91.2734, 29.7904), (94.2605, 29.7733)]

    def fake_reconstruct_full_scene_for_location(source_name, lon, lat, **kwargs):
        calls.append((source_name, lon, lat, kwargs["n_jobs"], kwargs["methods"]))
        return {"source": source_name, "lon": lon, "lat": lat}

    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.discover_available_locations", fake_discover_available_locations)
    monkeypatch.setattr("src.full_scene_reconstruction.pipeline.reconstruct_full_scene_for_location", fake_reconstruct_full_scene_for_location)

    results = reconstruct_full_scene_for_all_locations(
        source_name="sentinel-2",
        output_root=tmp_path / "output",
        data_root=tmp_path,
        cache_dir=tmp_path / "cache",
        methods=("nufrost", "hants"),
        n_jobs=-1,
    )

    assert calls == [
        ("sentinel-2", 91.2734, 29.7904, -1, ("nufrost", "hants")),
        ("sentinel-2", 94.2605, 29.7733, -1, ("nufrost", "hants")),
    ]
    assert results == [
        {"source": "sentinel-2", "lon": 91.2734, "lat": 29.7904},
        {"source": "sentinel-2", "lon": 94.2605, "lat": 29.7733},
    ]
