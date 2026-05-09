import argparse
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import rasterio


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VALID_RULE = "finite prediction and finite ground truth only; no numeric range filtering"


def _method_from_prediction_path(path: Path) -> str | None:
    match = re.match(r"^\[([^\]]+)\].*_prediction\.tif$", path.name)
    if match is None:
        return None
    method = match.group(1)
    if method.startswith("diff_") or method.endswith("_diff_qa") or method in {"ground_truth", "finite_only_rmse"}:
        return None
    return method


def _token_from_ground_truth(path: Path) -> str:
    stem = path.stem
    prefix = "[ground_truth]_"
    if not stem.startswith(prefix):
        raise ValueError(f"Ground truth filename must start with {prefix}: {path}")
    return stem[len(prefix):]


def _find_ground_truth(scene_dir: Path) -> Path:
    matches = sorted(path for path in scene_dir.iterdir() if path.name.startswith("[ground_truth]_") and path.suffix == ".tif")
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one [ground_truth]_*.tif in {scene_dir}, found {len(matches)}")
    return matches[0]


def _find_prediction_paths(scene_dir: Path, methods: Sequence[str] | None = None) -> Dict[str, Path]:
    wanted = set(methods) if methods else None
    found: Dict[str, Path] = {}
    for path in sorted(scene_dir.iterdir()):
        if path.suffix != ".tif":
            continue
        method = _method_from_prediction_path(path)
        if method is None:
            continue
        if wanted is not None and method not in wanted:
            continue
        found[method] = path
    return found


def _metrics_for_diff(
    method: str,
    pred_path: Path,
    pred: np.ndarray,
    gt: np.ndarray,
    descriptions: Sequence[str | None],
) -> Dict[str, Any]:
    valid = np.isfinite(pred) & np.isfinite(gt)
    diff = pred - gt
    all_diffs = diff[valid].astype(np.float64)
    if all_diffs.size == 0:
        overall_rmse = overall_mae = overall_bias = float("nan")
    else:
        overall_rmse = float(np.sqrt(np.mean(all_diffs ** 2)))
        overall_mae = float(np.mean(np.abs(all_diffs)))
        overall_bias = float(np.mean(all_diffs))

    bands: List[Dict[str, Any]] = []
    for band_idx in range(pred.shape[0]):
        band_valid = valid[band_idx]
        band_name = descriptions[band_idx] if band_idx < len(descriptions) and descriptions[band_idx] else f"band_{band_idx + 1}"
        pred_band = pred[band_idx]
        gt_band = gt[band_idx]
        pred_oob = band_valid & ((pred_band <= 0) | (pred_band >= 10000))
        gt_oob = band_valid & ((gt_band <= 0) | (gt_band >= 10000))
        if np.any(band_valid):
            band_diffs = diff[band_idx][band_valid].astype(np.float64)
            rmse = float(np.sqrt(np.mean(band_diffs ** 2)))
            mae = float(np.mean(np.abs(band_diffs)))
            bias = float(np.mean(band_diffs))
        else:
            rmse = mae = bias = float("nan")
        bands.append(
            {
                "band_index": band_idx + 1,
                "band_name": str(band_name),
                "valid_pixels": int(band_valid.sum()),
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "prediction_out_of_range_pixels": int(pred_oob.sum()),
                "ground_truth_out_of_range_pixels": int(gt_oob.sum()),
            }
        )

    return {
        "prediction": str(pred_path),
        "valid_pixels": int(valid.sum()),
        "prediction_out_of_range_pixels": int((valid & ((pred <= 0) | (pred >= 10000))).sum()),
        "ground_truth_out_of_range_pixels": int((valid & ((gt <= 0) | (gt >= 10000))).sum()),
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_bias": overall_bias,
        "bands": bands,
    }


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _write_diff(path: Path, diff: np.ndarray, profile: Dict[str, Any], method: str, descriptions: Sequence[str | None], label: str) -> None:
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="float32", count=diff.shape[0], nodata=np.nan, compress="deflate", predictor=3)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(diff.astype(np.float32))
        for band_idx in range(1, diff.shape[0] + 1):
            band_name = descriptions[band_idx - 1] if band_idx - 1 < len(descriptions) and descriptions[band_idx - 1] else f"band_{band_idx}"
            dst.set_band_description(band_idx, f"{method} {band_name} {label}")


def _write_markdown_report(scene_dir: Path, token: str, ground_truth_path: Path, results: Dict[str, Any]) -> Path:
    report_path = scene_dir / f"[diff_metrics]_{token}_metrics.md"
    lines = [
        "# Full-Scene Difference Metrics",
        "",
        f"- Scene directory: `{scene_dir}`",
        f"- Ground truth: `{ground_truth_path.name}`",
        f"- Valid rule: {VALID_RULE}",
        f"- Diff definition: `prediction - ground_truth`",
        "",
        "## Overall",
        "",
        "| Method | RMSE | MAE | Bias | Valid Pixels | Pred Out Of Range | GT Out Of Range | Diff GeoTIFF |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method in sorted(results["methods"]):
        item = results["methods"][method]
        lines.append(
            f"| {method} | {_format_float(item['overall_rmse'])} | {_format_float(item['overall_mae'])} | "
            f"{_format_float(item['overall_bias'])} | {item['valid_pixels']} | "
            f"{item['prediction_out_of_range_pixels']} | {item['ground_truth_out_of_range_pixels']} | `{Path(item['diff']).name}` |"
        )

    for method in sorted(results["methods"]):
        item = results["methods"][method]
        lines.extend(
            [
                "",
                f"## {method} Per Band",
                "",
                "| Band | RMSE | MAE | Bias | Valid Pixels | Pred Out Of Range | GT Out Of Range |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for band in item["bands"]:
            lines.append(
                f"| {band['band_name']} | {_format_float(band['rmse'])} | {_format_float(band['mae'])} | "
                f"{_format_float(band['bias'])} | {band['valid_pixels']} | "
                f"{band['prediction_out_of_range_pixels']} | {band['ground_truth_out_of_range_pixels']} |"
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def process_scene_dir(scene_dir: Path, methods: Sequence[str] | None = None) -> Dict[str, Any]:
    scene_dir = Path(scene_dir)
    ground_truth_path = _find_ground_truth(scene_dir)
    token = _token_from_ground_truth(ground_truth_path)
    prediction_paths = _find_prediction_paths(scene_dir, methods=methods)
    if not prediction_paths:
        raise FileNotFoundError(f"No method prediction GeoTIFFs found in {scene_dir}")

    with rasterio.open(ground_truth_path) as gt_src:
        gt = gt_src.read().astype(np.float32)
        profile = gt_src.profile.copy()
        descriptions = gt_src.descriptions

    results: Dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "ground_truth": str(ground_truth_path),
        "valid_rule": VALID_RULE,
        "diff_definition": "prediction - ground_truth",
        "shape": list(gt.shape),
        "methods": {},
    }

    for method, pred_path in prediction_paths.items():
        with rasterio.open(pred_path) as pred_src:
            pred = pred_src.read().astype(np.float32)
        if pred.shape != gt.shape:
            raise ValueError(f"{method} shape mismatch: prediction={pred.shape}, ground_truth={gt.shape}")
        diff = np.full(pred.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(pred) & np.isfinite(gt)
        diff[valid] = pred[valid] - gt[valid]
        diff_path = scene_dir / f"[diff_{method}]_{token}_prediction_minus_ground_truth.tif"
        _write_diff(diff_path, diff, profile, method, descriptions, "signed diff pred-ground_truth")
        method_metrics = _metrics_for_diff(method, pred_path, pred, gt, descriptions)
        method_metrics["diff"] = str(diff_path)
        results["methods"][method] = method_metrics

    report_path = _write_markdown_report(scene_dir, token, ground_truth_path, results)
    results["report"] = str(report_path)
    return results


def _iter_scene_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir() and any(child.name.startswith("[ground_truth]_") and child.suffix == ".tif" for child in path.iterdir()):
            yield path


def process_all_scene_dirs(root: Path, methods: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    return [process_scene_dir(scene_dir, methods=methods) for scene_dir in _iter_scene_dirs(Path(root))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute full-scene diff GeoTIFFs and RMSE/MAE Markdown reports.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene-dir", type=Path, help="One full-scene output directory containing ground truth and predictions.")
    group.add_argument("--all-scene-dirs", type=Path, help="Parent directory containing multiple scene output directories.")
    parser.add_argument("--methods", nargs="+", default=None, help="Optional method subset, e.g. nufrost hants zhu2015.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scene_dir is not None:
        result = process_scene_dir(args.scene_dir, methods=args.methods)
        print(f"Wrote report: {result['report']}")
        for method, item in sorted(result["methods"].items()):
            print(
                f"{method}: RMSE={_format_float(item['overall_rmse'])} MAE={_format_float(item['overall_mae'])} diff={item['diff']}"
            )
        return 0

    results = process_all_scene_dirs(args.all_scene_dirs, methods=args.methods)
    for result in results:
        print(f"Wrote report: {result['report']}")
    print(f"Processed {len(results)} scene directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
