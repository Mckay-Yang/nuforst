import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compute_full_scene_diff_metrics.py"
SPEC = importlib.util.spec_from_file_location("compute_full_scene_diff_metrics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compute_full_scene_diff_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute_full_scene_diff_metrics)


def _write_stack(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
        for band_idx in range(1, data.shape[0] + 1):
            dst.set_band_description(band_idx, f"B{band_idx}")


def _make_scene(scene_dir: Path) -> None:
    gt = np.array(
        [
            [[1.0, 2.0], [3.0, np.nan]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        dtype=np.float32,
    )
    nufrost = np.array(
        [
            [[2.0, 4.0], [10001.0, 5.0]],
            [[8.0, 25.0], [np.nan, 35.0]],
        ],
        dtype=np.float32,
    )
    hants = gt + 1.0

    token = "sentinel-2_lon1.000000_lat2.000000_2026-01-01T00-00-00"
    _write_stack(scene_dir / f"[ground_truth]_{token}.tif", gt)
    _write_stack(scene_dir / f"[nufrost]_{token}_prediction.tif", nufrost)
    _write_stack(scene_dir / f"[hants]_{token}_prediction.tif", hants)
    _write_stack(scene_dir / f"[diff_old]_{token}_prediction_minus_ground_truth.tif", hants)


def test_process_scene_writes_diff_rasters_and_markdown_with_finite_only_metrics(tmp_path: Path) -> None:
    scene_dir = tmp_path / "sentinel-2_recon" / "1.0000_2.0000"
    _make_scene(scene_dir)

    result = compute_full_scene_diff_metrics.process_scene_dir(scene_dir)

    assert sorted(result["methods"]) == ["hants", "nufrost"]
    diff_path = scene_dir / "[diff_nufrost]_sentinel-2_lon1.000000_lat2.000000_2026-01-01T00-00-00_prediction_minus_ground_truth.tif"
    report_path = scene_dir / "[diff_metrics]_sentinel-2_lon1.000000_lat2.000000_2026-01-01T00-00-00_metrics.md"
    assert diff_path.exists()
    assert report_path.exists()

    with rasterio.open(diff_path) as src:
        diff = src.read()

    assert diff.shape == (2, 2, 2)
    assert diff[0, 0, 0] == 1.0
    assert diff[0, 1, 0] == 9998.0
    assert diff[1, 0, 0] == -2.0
    assert diff[1, 1, 1] == -5.0
    assert np.isnan(diff[0, 1, 1])

    nufrost = result["methods"]["nufrost"]
    expected_diffs = np.array([1.0, 2.0, 9998.0, -2.0, 5.0, -5.0], dtype=np.float64)
    assert nufrost["valid_pixels"] == 6
    assert nufrost["prediction_out_of_range_pixels"] == 1
    assert nufrost["ground_truth_out_of_range_pixels"] == 0
    assert np.isclose(nufrost["overall_rmse"], np.sqrt(np.mean(expected_diffs ** 2)))
    assert np.isclose(nufrost["overall_mae"], np.mean(np.abs(expected_diffs)))

    report = report_path.read_text(encoding="utf-8")
    assert "finite prediction and finite ground truth only" in report
    assert "Diff definition: `prediction - ground_truth`" in report
    assert "| nufrost |" in report
    assert "| hants |" in report


def test_process_all_scene_dirs_skips_directories_without_ground_truth(tmp_path: Path) -> None:
    root = tmp_path / "sentinel-2_recon"
    scene_a = root / "1.0000_2.0000"
    scene_b = root / "3.0000_4.0000"
    _make_scene(scene_a)
    scene_b.mkdir(parents=True)

    results = compute_full_scene_diff_metrics.process_all_scene_dirs(root)

    assert len(results) == 1
    assert results[0]["scene_dir"] == str(scene_a)
