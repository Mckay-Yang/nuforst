from pathlib import Path

import numpy as np
import rasterio

from src.full_scene_reconstruction import (
    build_ground_truth_output_path,
    build_output_path,
    build_scene_stack_output_path,
    collapse_duplicate_timestamps,
    discover_location_band_stacks,
    extract_prediction_2d,
    intersect_band_timestamps,
    make_masked_time_series,
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


def test_build_output_path_includes_timestamp_and_method(tmp_path: Path) -> None:
    output_path = build_output_path(
        output_root=tmp_path,
        method_name="nufrost",
        source_file=Path("COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733.tif"),
        target_time="2026-01-27T04:19:39",
    )

    assert output_path.parent == tmp_path / "nufrost"
    assert output_path.name == "[nufrost]_COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733_2026-01-27T04-19-39__nufrost.tif"


def test_build_ground_truth_output_path(tmp_path: Path) -> None:
    gt_path = build_ground_truth_output_path(
        output_root=tmp_path,
        source_name="sentinel-2",
        lon=94.2605,
        lat=29.7733,
        target_time="2026-01-27T04:19:39",
    )
    assert gt_path.parent == tmp_path / "grand_truth"
    assert gt_path.name == "sentinel-2_lon94.260500_lat29.773300_2026-01-27T04-19-39__grand_truth.tif"


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

    assert output_path.name == "[nufrost]_sentinel-2_lon94.260500_lat29.773300_2026-01-27T04-19-39__nufrost_QA_stack.tif"


def test_reconstruct_full_scene_skips_when_summary_exists(tmp_path: Path) -> None:
    from src.full_scene_reconstruction import reconstruct_full_scene_for_location

    source_name = "sentinel-2"
    lon = 94.2605
    lat = 29.7733
    safe_source = source_name.replace("/", "-")
    summary_dir = tmp_path / "run_summaries"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / f"reconstruction_summary_{safe_source}_lon{lon:.6f}_lat{lat:.6f}_2026-02-06T04-18-39.json"
    summary_path.write_text(
        '{"source":"sentinel-2","lon":94.2605,"lat":29.7733,"target_time":"2026-02-06T04:18:39"}',
        encoding="utf-8",
    )

    result = reconstruct_full_scene_for_location(
        source_name=source_name,
        lon=lon,
        lat=lat,
        output_root=tmp_path,
        data_root=tmp_path / "does-not-matter",
    )

    assert result["skipped"] is True
    assert result["summary_path"] == str(summary_path)


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
