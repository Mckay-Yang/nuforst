import importlib.util
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


def test_local_evals_script_parses_cli_and_dispatches_workflow(tmp_path: Path, monkeypatch) -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_local_evals.py"
    if not script_path.exists():
        raise AssertionError(f"Expected script to exist at {script_path}")

    spec = importlib.util.spec_from_file_location("run_local_evals", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script module from {script_path}")
    script_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_module)

    calls = []

    def fake_run_local_evals_workflow(**kwargs):
        calls.append(kwargs)
        return {"ablation": 1, "sparse": 2, "gap": 3, "repeatability": 4}

    monkeypatch.setattr(script_module, "run_local_evals_workflow", fake_run_local_evals_workflow)

    exit_code = script_module.main(
        [
            "--source-name",
            "sentinel-2",
            "--output-dir",
            str(tmp_path / "output"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--max-images",
            "3",
            "--n-jobs",
            "2",
            "--run-ablation",
            "--run-gap",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "source_name": "sentinel-2",
            "project_dir": script_module.REPO_ROOT,
            "output_dir": tmp_path / "output",
            "cache_dir": tmp_path / "cache",
            "max_images": 3,
            "n_jobs": 2,
            "run_ablation": True,
            "run_sparse": False,
            "run_gap": True,
            "run_repeatability": False,
        }
    ]


def test_local_eval_workflow_skips_completed_sparse_levels(tmp_path: Path, monkeypatch) -> None:
    from src import local_eval_workflow as workflow

    loc_id = "BLUE_lon91.2734_lat29.7904"
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    output_dir.mkdir()
    cache_dir.mkdir()
    sparse_csv = output_dir / "sentinel-2_sparse_sweep_results.csv"
    pd.DataFrame([{"Image": loc_id, "NumPoints": 1000, "Algorithm": "NuFrost"}]).to_csv(sparse_csv, index=False)

    monkeypatch.setattr(workflow, "discover_image_paths_list", lambda config: [[f"/tmp/{loc_id}.vrt"]])

    @contextmanager
    def fake_open_evaluation_source(image_paths, args):
        yield {
            "source": object(),
            "t_sec": np.array([0.0, 86400.0, 2 * 86400.0]),
            "t_days": np.array([0.0, 1.0, 2.0]),
            "meta": {"count": 3, "height": 2, "width": 2},
        }

    monkeypatch.setattr(workflow.src.evaluation, "open_evaluation_source", fake_open_evaluation_source)
    monkeypatch.setattr(workflow.src.evaluation, "scan_pixel_stats_from_source", lambda source, t_days: {"valid_counts": np.full((2, 2), 20, dtype=np.int32)})
    monkeypatch.setattr(workflow.src.evaluation, "sample_random_points_from_source", lambda source, t_days, min_obs, num_points, seed, precomputed_stats=None: np.zeros((num_points, 3), dtype=int))
    monkeypatch.setattr(workflow, "select_gap_pixels", lambda *args, **kwargs: (np.ones((10, 2), dtype=int), 10, 10))
    monkeypatch.setattr(workflow, "gap_days_from_index_targets", lambda t_days, targets: ([], 2.0))
    monkeypatch.setattr(workflow, "gap_days_for_index", lambda t_days, index_value: (1, 2.0))

    calls = []

    def fake_evaluate_algorithms_from_source(source, t_sec, t_days, args, sampled_points, n_jobs):
        calls.append(len(sampled_points))
        return pd.DataFrame([
            {"Algorithm": "NuFrost", "RMSE": 1.0, "MAE": 1.0, "R": 1.0, "OutlierRatio": 0.0},
            {"Algorithm": "Zhu2015", "RMSE": 1.0, "MAE": 1.0, "R": 1.0, "OutlierRatio": 0.0},
            {"Algorithm": "HANTS", "RMSE": 1.0, "MAE": 1.0, "R": 1.0, "OutlierRatio": 0.0},
        ])

    monkeypatch.setattr(workflow.src.evaluation, "evaluate_algorithms_from_source", fake_evaluate_algorithms_from_source)

    summary = workflow.run_local_evals_workflow(
        source_name="sentinel-2",
        project_dir=tmp_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
        max_images=1,
        n_jobs=1,
        run_ablation=False,
        run_sparse=True,
        run_gap=False,
        run_repeatability=False,
    )

    assert calls == [5000, 10000, 20000]
    assert summary["sparse"] == 9
