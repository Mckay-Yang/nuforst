import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import rasterio


def test_full_scene_scripts_show_help_when_run_as_python_scripts() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_paths = [
        repo_root / "scripts" / "run_full_scene_reconstruction.py",
        repo_root / "scripts" / "run_small_window_full_scene.py",
    ]

    for script_path in script_paths:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_full_scene_script_runs_without_window_crop(tmp_path: Path, monkeypatch) -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_full_scene_reconstruction.py"
    spec = importlib.util.spec_from_file_location("run_full_scene_reconstruction", script_path)
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
            "--methods",
            "hants",
            "--n-jobs",
            "1",
        ]
    )

    assert exit_code == 0

    with rasterio.open(tmp_path / "output" / "sentinel-2_recon" / "94.2605_29.7733" / "[ground_truth]_sentinel-2_lon94.260500_lat29.773300_2025-01-01T00-00-00.tif") as ds:
        assert ds.height == 16
        assert ds.width == 16

    with rasterio.open(tmp_path / "output" / "sentinel-2_recon" / "94.2605_29.7733" / "[hants]_sentinel-2_lon94.260500_lat29.773300_2025-01-01T00-00-00_prediction.tif") as ds:
        assert ds.height == 16
        assert ds.width == 16


def test_full_scene_script_runs_all_locations_when_flag_is_set(tmp_path: Path, monkeypatch) -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_full_scene_reconstruction.py"
    spec = importlib.util.spec_from_file_location("run_full_scene_reconstruction", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script module from {script_path}")
    script_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_module)

    calls = []

    def fake_reconstruct_full_scene_for_all_locations(**kwargs):
        calls.append(kwargs)
        return [{"source": kwargs["source_name"], "count": 2}]

    monkeypatch.setattr(script_module, "reconstruct_full_scene_for_all_locations", fake_reconstruct_full_scene_for_all_locations)

    exit_code = script_module.main(
        [
            "--source-name",
            "hls",
            "--all-coordinates",
            "--output-root",
            str(tmp_path / "output"),
            "--data-root",
            str(tmp_path),
            "--n-jobs",
            "-1",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "source_name": "hls",
            "output_root": tmp_path / "output",
            "data_root": tmp_path,
            "cache_dir": Path("data/cache/local"),
            "methods": ("nufrost", "hants", "zhu2015"),
            "n_jobs": -1,
            "force_refresh": False,
            "rerun_methods": (),
        }
    ]
