#!/usr/bin/env python3
"""Evaluate reconstruction GeoTIFFs and write derived rasters/statistics.

Generated rasters:
- DIFF_[method]_*: per-band prediction - ground_truth
- NDVI_[method]_*: NDVI for ground_truth and each method
- DIFF_NDVI_[method]_*: NDVI(prediction) - NDVI(ground_truth)
- DIFF_EVI_[method]_*: EVI(prediction) - EVI(ground_truth)
- DIFF_NDWI_[method]_*: NDWI(prediction) - NDWI(ground_truth)
- DIFF_NDSI_[method]_*: NDSI(prediction) - NDSI(ground_truth)
- DIFF_NDMI_[method]_*: NDMI(prediction) - NDMI(ground_truth)
- DIFF_NBR_[method]_*: NBR(prediction) - NBR(ground_truth)

Generated tables:
- evaluation_summary.csv
- evaluation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


METHODS = ("nufrost", "hants", "zhu2015")
TRUTH_METHOD = "ground_truth"
BLUE_BAND_NAME = "B2"
GREEN_BAND_NAME = "B3"
RED_BAND_NAME = "B4"
NIR_BAND_NAME = "B8"
SWIR1_BAND_NAME = "B11"
SWIR2_BAND_NAME = "B12"
INDEX_NAMES = ("NDVI", "EVI", "NDWI", "NDSI", "NDMI", "NBR")
INDEX_VALID_RANGES = {
    "NDVI": (-1.0, 1.0),
    "EVI": (-1.0, 1.0),
    "NDWI": (-1.0, 1.0),
    "NDSI": (-1.0, 1.0),
    "NDMI": (-1.0, 1.0),
    "NBR": (-1.0, 1.0),
}

RAW_ERROR_BINS = (
    ("<=50", 0.0, 50.0),
    ("50-100", 50.0, 100.0),
    ("100-250", 100.0, 250.0),
    ("250-500", 250.0, 500.0),
    ("500-1000", 500.0, 1000.0),
    (">1000", 1000.0, math.inf),
)

INDEX_ERROR_BINS = (
    ("<=0.02", 0.0, 0.02),
    ("0.02-0.05", 0.02, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    (">0.20", 0.20, math.inf),
)

INDEX_CLASSES = (
    ("<0", -math.inf, 0.0),
    ("0-0.2", 0.0, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    (">=0.6", 0.6, math.inf),
)


@dataclass
class ContinuousAccumulator:
    n: int = 0
    sum: float = 0.0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)].astype("float64", copy=False)
        if finite.size == 0:
            return
        self.n += int(finite.size)
        self.sum += float(finite.sum())
        self.sum_abs += float(np.abs(finite).sum())
        self.sum_sq += float(np.square(finite).sum())
        self.min_value = min(self.min_value, float(finite.min()))
        self.max_value = max(self.max_value, float(finite.max()))

    def row(self) -> dict[str, object]:
        if self.n == 0:
            return stats_row_values(np.array([], dtype="float64"), 0)
        bias = self.sum / self.n
        rmse = math.sqrt(self.sum_sq / self.n)
        mae = self.sum_abs / self.n
        variance = max(self.sum_sq / self.n - bias * bias, 0.0)
        return {
            "n": self.n,
            "valid_ratio": "",
            "bias": bias,
            "mae": mae,
            "rmse": rmse,
            "std": math.sqrt(variance),
            "median": "",
            "p05": "",
            "p95": "",
            "min": self.min_value,
            "max": self.max_value,
            "fraction": "",
        }


def method_from_prediction_path(prediction_path: Path) -> str:
    name = prediction_path.name
    if not name.startswith("[") or "]_" not in name:
        raise ValueError(f"cannot parse method name from {prediction_path}")
    return name[1 : name.index("]")]


def prediction_stem(prediction_path: Path) -> str:
    method = method_from_prediction_path(prediction_path)
    return prediction_path.name.removeprefix(f"[{method}]_").removesuffix("_prediction.tif")


def truth_stem(truth_path: Path) -> str:
    return truth_path.name.removeprefix("[ground_truth]_").removesuffix(".tif")


def diff_path(prediction_path: Path) -> Path:
    name = prediction_path.name.replace("_prediction.tif", "_diff.tif")
    return prediction_path.with_name(f"DIFF_{name}")


def index_path(source_path: Path, method: str, stem: str, index_name: str) -> Path:
    lower = index_name.lower()
    return source_path.with_name(f"{index_name}_[{method}]_{stem}_{lower}.tif")


def ndvi_path(source_path: Path, method: str, stem: str) -> Path:
    return index_path(source_path, method, stem, "NDVI")


def ndvi_diff_path(prediction_path: Path) -> Path:
    method = method_from_prediction_path(prediction_path)
    stem = prediction_stem(prediction_path)
    return prediction_path.with_name(f"DIFF_NDVI_[{method}]_{stem}_ndvi_diff.tif")


def evi_diff_path(prediction_path: Path) -> Path:
    method = method_from_prediction_path(prediction_path)
    stem = prediction_stem(prediction_path)
    return prediction_path.with_name(f"DIFF_EVI_[{method}]_{stem}_evi_diff.tif")


def index_diff_path(prediction_path: Path, index_name: str) -> Path:
    method = method_from_prediction_path(prediction_path)
    stem = prediction_stem(prediction_path)
    lower = index_name.lower()
    return prediction_path.with_name(f"DIFF_{index_name}_[{method}]_{stem}_{lower}_diff.tif")


def band_index(ds: rasterio.DatasetReader, band_name: str) -> int:
    for idx, desc in enumerate(ds.descriptions, start=1):
        if desc == band_name:
            return idx
    sentinel2_fallback = {
        "B2": 1,
        "B3": 2,
        "B4": 3,
        "B8": 4,
        "B11": 5,
        "B12": 6,
    }
    fallback_idx = sentinel2_fallback.get(band_name)
    if fallback_idx is not None and fallback_idx <= ds.count:
        return fallback_idx
    raise ValueError(f"could not find band {band_name} in {ds.name}")


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return normalized_difference(nir, red)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.full(a.shape, np.nan, dtype="float32")
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (np.abs(denom) > 1.0e-6)
    out[valid] = (a[valid] - b[valid]) / denom[valid]
    return out


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    # Sentinel-2 files use scaled DN values, so the "+1" reflectance term is
    # represented as +10000 in this scale.
    denom = nir + 6.0 * red - 7.5 * blue + 10000.0
    out = np.full(nir.shape, np.nan, dtype="float32")
    valid = (
        np.isfinite(nir)
        & np.isfinite(red)
        & np.isfinite(blue)
        & np.isfinite(denom)
        & (np.abs(denom) > 1.0e-6)
    )
    out[valid] = 2.5 * (nir[valid] - red[valid]) / denom[valid]
    return out


def single_band_profile(ds: rasterio.DatasetReader) -> dict:
    profile = ds.profile.copy()
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
    return profile


def multi_band_profile(ds: rasterio.DatasetReader) -> dict:
    profile = ds.profile.copy()
    profile.update(dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
    return profile


def read_index_inputs(ds: rasterio.DatasetReader):
    blue = ds.read(band_index(ds, BLUE_BAND_NAME)).astype("float32", copy=False)
    green = ds.read(band_index(ds, GREEN_BAND_NAME)).astype("float32", copy=False)
    red = ds.read(band_index(ds, RED_BAND_NAME)).astype("float32", copy=False)
    nir = ds.read(band_index(ds, NIR_BAND_NAME)).astype("float32", copy=False)
    swir1 = ds.read(band_index(ds, SWIR1_BAND_NAME)).astype("float32", copy=False)
    swir2 = ds.read(band_index(ds, SWIR2_BAND_NAME)).astype("float32", copy=False)
    return blue, green, red, nir, swir1, swir2


def compute_indices(ds: rasterio.DatasetReader) -> dict[str, np.ndarray]:
    blue, green, red, nir, swir1, swir2 = read_index_inputs(ds)
    indices = {
        "NDVI": ndvi(nir, red),
        "EVI": evi(nir, red, blue),
        "NDWI": normalized_difference(green, nir),
        "NDSI": normalized_difference(green, swir1),
        "NDMI": normalized_difference(nir, swir1),
        "NBR": normalized_difference(nir, swir2),
    }
    return {name: mask_out_of_range_index(name, values) for name, values in indices.items()}


def mask_out_of_range_index(index_name: str, values: np.ndarray) -> np.ndarray:
    """Drop physically invalid index values before raster export/statistics."""
    low, high = INDEX_VALID_RANGES[index_name]
    out = values.astype("float32", copy=True)
    valid = np.isfinite(out) & (out >= low) & (out <= high)
    out[~valid] = np.nan
    return out


def finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)].astype("float64", copy=False)


def stats_row_values(values: np.ndarray, total_count: int) -> dict[str, object]:
    finite = finite_values(values)
    n = int(finite.size)
    if n == 0:
        return {
            "n": 0,
            "valid_ratio": 0.0 if total_count else "",
            "bias": "",
            "mae": "",
            "rmse": "",
            "std": "",
            "median": "",
            "p05": "",
            "p95": "",
            "min": "",
            "max": "",
            "fraction": "",
        }
    return {
        "n": n,
        "valid_ratio": n / total_count if total_count else "",
        "bias": float(finite.mean()),
        "mae": float(np.abs(finite).mean()),
        "rmse": float(np.sqrt(np.square(finite).mean())),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "fraction": "",
    }


def add_stats_row(
    rows: list[dict[str, object]],
    *,
    section: str,
    scene: str,
    method: str,
    product: str,
    band_or_index: str,
    values: np.ndarray,
    total_count: int,
    class_kind: str = "all",
    class_name: str = "all",
    truth_class: str = "",
    pred_class: str = "",
) -> None:
    row = {
        "section": section,
        "scene": scene,
        "method": method,
        "product": product,
        "band_or_index": band_or_index,
        "class_kind": class_kind,
        "class_name": class_name,
        "truth_class": truth_class,
        "pred_class": pred_class,
    }
    row.update(stats_row_values(values, total_count))
    rows.append(row)


def add_count_row(
    rows: list[dict[str, object]],
    *,
    section: str,
    scene: str,
    method: str,
    product: str,
    band_or_index: str,
    class_kind: str,
    class_name: str,
    n: int,
    denominator: int,
    truth_class: str = "",
    pred_class: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "scene": scene,
            "method": method,
            "product": product,
            "band_or_index": band_or_index,
            "class_kind": class_kind,
            "class_name": class_name,
            "truth_class": truth_class,
            "pred_class": pred_class,
            "n": int(n),
            "valid_ratio": "",
            "bias": "",
            "mae": "",
            "rmse": "",
            "std": "",
            "median": "",
            "p05": "",
            "p95": "",
            "min": "",
            "max": "",
            "fraction": int(n) / denominator if denominator else 0.0,
        }
    )


def add_error_bin_rows(
    rows: list[dict[str, object]],
    *,
    scene: str,
    method: str,
    product: str,
    band_or_index: str,
    diff: np.ndarray,
    bins: tuple[tuple[str, float, float], ...],
) -> None:
    finite = finite_values(diff)
    abs_diff = np.abs(finite)
    denominator = int(abs_diff.size)
    for label, low, high in bins:
        if math.isinf(high):
            n = int((abs_diff > low).sum())
        elif low == 0.0:
            n = int((abs_diff <= high).sum())
        else:
            n = int(((abs_diff > low) & (abs_diff <= high)).sum())
        add_count_row(
            rows,
            section="error_bin",
            scene=scene,
            method=method,
            product=product,
            band_or_index=band_or_index,
            class_kind="abs_error",
            class_name=label,
            n=n,
            denominator=denominator,
        )


def classify_index(values: np.ndarray) -> np.ndarray:
    classes = np.full(values.shape, -1, dtype="int16")
    finite = np.isfinite(values)
    for idx, (_label, low, high) in enumerate(INDEX_CLASSES):
        mask = finite & (values >= low) & (values < high)
        classes[mask] = idx
    return classes


def add_index_class_rows(
    rows: list[dict[str, object]],
    *,
    scene: str,
    method: str,
    index_name: str,
    pred_index: np.ndarray,
    truth_index: np.ndarray,
    diff: np.ndarray,
) -> None:
    valid = np.isfinite(pred_index) & np.isfinite(truth_index) & np.isfinite(diff)
    denominator = int(valid.sum())
    pred_cls = classify_index(pred_index)
    truth_cls = classify_index(truth_index)

    for class_idx, (label, _low, _high) in enumerate(INDEX_CLASSES):
        mask = valid & (truth_cls == class_idx)
        add_stats_row(
            rows,
            section="index_truth_class_stats",
            scene=scene,
            method=method,
            product=f"DIFF_{index_name}",
            band_or_index=index_name,
            values=np.where(mask, diff, np.nan),
            total_count=denominator,
            class_kind="truth_index_class",
            class_name=label,
        )

    for truth_idx, (truth_label, _tl, _th) in enumerate(INDEX_CLASSES):
        for pred_idx, (pred_label, _pl, _ph) in enumerate(INDEX_CLASSES):
            n = int((valid & (truth_cls == truth_idx) & (pred_cls == pred_idx)).sum())
            add_count_row(
                rows,
                section="index_confusion",
                scene=scene,
                method=method,
                product=index_name,
                band_or_index=index_name,
                class_kind="truth_pred_index_class",
                class_name=f"{truth_label}->{pred_label}",
                truth_class=truth_label,
                pred_class=pred_label,
                n=n,
                denominator=denominator,
            )


def iter_truths(recon_root: Path):
    for scene_dir in sorted(p for p in recon_root.iterdir() if p.is_dir()):
        for truth in sorted(scene_dir.iterdir()):
            if truth.is_file() and truth.name.startswith("[ground_truth]_") and truth.suffix == ".tif":
                yield scene_dir.name, truth


def iter_prediction_pairs(recon_root: Path):
    for scene_dir in sorted(p for p in recon_root.iterdir() if p.is_dir()):
        files = sorted(p for p in scene_dir.iterdir() if p.is_file())
        truths = [p for p in files if p.name.startswith("[ground_truth]_") and p.suffix == ".tif"]
        if not truths:
            continue
        truth_by_time = {}
        for truth in truths:
            stem = truth_stem(truth)
            truth_by_time[stem] = truth

        for method in METHODS:
            prefix = f"[{method}]_"
            for pred in files:
                if not (pred.name.startswith(prefix) and pred.name.endswith("_prediction.tif")):
                    continue
                truth = truth_by_time.get(prediction_stem(pred))
                if truth is not None:
                    yield method, scene_dir.name, pred, truth


def write_index_raster(
    source_path: Path,
    method: str,
    stem: str,
    index_name: str,
    skip_existing: bool,
) -> Path:
    out_path = index_path(source_path, method, stem, index_name)
    if out_path.exists() and skip_existing:
        return out_path
    if out_path.exists():
        out_path.unlink()

    with rasterio.open(source_path) as ds:
        source_index = compute_indices(ds)[index_name]
        source_index[~np.isfinite(source_index)] = np.nan
        with rasterio.open(out_path, "w", **single_band_profile(ds)) as out_ds:
            out_ds.write(source_index, 1)
            out_ds.set_band_description(1, index_name)
    return out_path


def write_all_index_rasters(source_path: Path, method: str, stem: str, skip_existing: bool) -> None:
    for index_name in INDEX_NAMES:
        write_index_raster(source_path, method, stem, index_name, skip_existing)


def evaluate_pair(
    rows: list[dict[str, object]],
    accumulators: dict[tuple[str, str, str], ContinuousAccumulator],
    method: str,
    scene: str,
    prediction_path: Path,
    truth_path: Path,
    skip_existing: bool,
) -> None:
    with rasterio.open(prediction_path) as pred_ds, rasterio.open(truth_path) as truth_ds:
        if pred_ds.count != truth_ds.count:
            raise ValueError(
                f"band count mismatch for {prediction_path}: "
                f"{pred_ds.count} vs {truth_ds.count}"
            )
        if pred_ds.width != truth_ds.width or pred_ds.height != truth_ds.height:
            raise ValueError(
                f"shape mismatch for {prediction_path}: "
                f"{pred_ds.height}x{pred_ds.width} vs {truth_ds.height}x{truth_ds.width}"
            )

        diff_out = diff_path(prediction_path)
        if not (diff_out.exists() and skip_existing):
            if diff_out.exists():
                diff_out.unlink()
            with rasterio.open(diff_out, "w", **multi_band_profile(pred_ds)) as out_ds:
                for band_idx in range(1, pred_ds.count + 1):
                    pred = pred_ds.read(band_idx).astype("float32", copy=False)
                    truth = truth_ds.read(band_idx).astype("float32", copy=False)
                    diff = pred - truth
                    diff[~np.isfinite(pred) | ~np.isfinite(truth)] = np.nan
                    out_ds.write(diff, band_idx)
                    desc = pred_ds.descriptions[band_idx - 1] or f"band_{band_idx}"
                    out_ds.set_band_description(band_idx, desc)

        for band_idx in range(1, pred_ds.count + 1):
            pred = pred_ds.read(band_idx).astype("float32", copy=False)
            truth = truth_ds.read(band_idx).astype("float32", copy=False)
            diff = pred - truth
            diff[~np.isfinite(pred) | ~np.isfinite(truth)] = np.nan
            desc = pred_ds.descriptions[band_idx - 1] or f"band_{band_idx}"
            add_stats_row(
                rows,
                section="continuous",
                scene=scene,
                method=method,
                product="DIFF",
                band_or_index=desc,
                values=diff,
                total_count=diff.size,
            )
            add_error_bin_rows(
                rows,
                scene=scene,
                method=method,
                product="DIFF",
                band_or_index=desc,
                diff=diff,
                bins=RAW_ERROR_BINS,
            )
            accumulators[(method, "DIFF", desc)].update(diff)

        pred_indices = compute_indices(pred_ds)
        truth_indices = compute_indices(truth_ds)

        for index_name in INDEX_NAMES:
            pred_index = pred_indices[index_name]
            truth_index = truth_indices[index_name]
            out_path = index_diff_path(prediction_path, index_name)
            index_diff = pred_index - truth_index
            index_diff[~np.isfinite(index_diff)] = np.nan
            if not (out_path.exists() and skip_existing):
                if out_path.exists():
                    out_path.unlink()
                with rasterio.open(out_path, "w", **single_band_profile(pred_ds)) as out_ds:
                    out_ds.write(index_diff, 1)
                    out_ds.set_band_description(1, f"{index_name}_diff")
            add_stats_row(
                rows,
                section="continuous",
                scene=scene,
                method=method,
                product=f"DIFF_{index_name}",
                band_or_index=index_name,
                values=index_diff,
                total_count=index_diff.size,
            )
            add_error_bin_rows(
                rows,
                scene=scene,
                method=method,
                product=f"DIFF_{index_name}",
                band_or_index=index_name,
                diff=index_diff,
                bins=INDEX_ERROR_BINS,
            )
            add_index_class_rows(
                rows,
                scene=scene,
                method=method,
                index_name=index_name,
                pred_index=pred_index,
                truth_index=truth_index,
                diff=index_diff,
            )
            accumulators[(method, f"DIFF_{index_name}", index_name)].update(index_diff)

    write_all_index_rasters(prediction_path, method, prediction_stem(prediction_path), skip_existing)


def evaluate_truth_ndvi(rows: list[dict[str, object]], scene: str, truth_path: Path, skip_existing: bool) -> None:
    write_all_index_rasters(truth_path, TRUTH_METHOD, truth_stem(truth_path), skip_existing)
    with rasterio.open(truth_path) as truth_ds:
        truth_indices = compute_indices(truth_ds)
        for index_name, truth_index in truth_indices.items():
            add_stats_row(
                rows,
                section="prediction_distribution",
                scene=scene,
                method=TRUTH_METHOD,
                product=index_name,
                band_or_index=index_name,
                values=truth_index,
                total_count=truth_index.size,
            )


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    fieldnames = [
        "section",
        "scene",
        "method",
        "product",
        "band_or_index",
        "class_kind",
        "class_name",
        "truth_class",
        "pred_class",
        "n",
        "valid_ratio",
        "fraction",
        "bias",
        "mae",
        "rmse",
        "std",
        "median",
        "p05",
        "p95",
        "min",
        "max",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt_value(value: object) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows: list[dict[str, object]], md_path: Path) -> None:
    continuous = [
        row
        for row in rows
        if row["section"] == "overall_continuous" and row["class_kind"] == "all"
    ]
    index_bins = [
        row
        for row in rows
        if row["section"] == "error_bin" and row["product"] in {f"DIFF_{name}" for name in INDEX_NAMES}
    ]
    lines = [
        "# Reconstruction Evaluation Summary",
        "",
        "Signed differences use `prediction - ground_truth`; absolute values are only used for error-bin classification.",
        "",
        "## Overall Continuous Metrics",
        "",
        "| method | product | band/index | n | bias | MAE | RMSE | std | min | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in continuous:
        lines.append(
            "| {method} | {product} | {band} | {n} | {bias} | {mae} | {rmse} | {std} | {minv} | {maxv} |".format(
                method=row["method"],
                product=row["product"],
                band=row["band_or_index"],
                n=row["n"],
                bias=fmt_value(row["bias"]),
                mae=fmt_value(row["mae"]),
                rmse=fmt_value(row["rmse"]),
                std=fmt_value(row["std"]),
                minv=fmt_value(row["min"]),
                maxv=fmt_value(row["max"]),
            )
        )

    lines.extend(
        [
            "",
            "## Index Error Bins",
            "",
            "| scene | method | product | bin | n | fraction |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in index_bins:
        lines.append(
            f"| {row['scene']} | {row['method']} | {row['product']} | {row['class_name']} | {row['n']} | {fmt_value(row['fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `DIFF_[method]_*_diff.tif`: per-band signed difference.",
            "- `INDEX_[method]_*_index.tif`: spectral indices for ground truth and predictions.",
            "- `DIFF_INDEX_[method]_*_index_diff.tif`: signed spectral-index difference.",
        ]
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")


def cleanup_legacy_outputs(recon_root: Path) -> None:
    patterns = (
        "diff_*_residual.tif",
        "diff_ndvi_*_ndvi_diff.tif",
        "[ground_truth_ndvi]_*_ndvi.tif",
    )
    for pattern in patterns:
        for path in recon_root.glob(f"*/{pattern}"):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recon-root",
        type=Path,
        default=Path("data/products/reconstruction/sentinel-2_recon"),
        help="Root directory containing per-location reconstruction outputs.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("data/assets/records/reconstruction_evaluation"),
        help="Directory for evaluation_summary.csv and evaluation_summary.md.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute raster outputs that already exist.",
    )
    parser.add_argument(
        "--keep-legacy",
        action="store_true",
        help="Keep old diff_* and [ground_truth_ndvi] outputs if present.",
    )
    args = parser.parse_args()

    if not args.keep_legacy:
        cleanup_legacy_outputs(args.recon_root)

    rows: list[dict[str, object]] = []
    accumulators: dict[tuple[str, str, str], ContinuousAccumulator] = defaultdict(ContinuousAccumulator)

    truth_count = 0
    for scene, truth in iter_truths(args.recon_root):
        evaluate_truth_ndvi(rows, scene, truth, args.skip_existing)
        truth_count += 1

    pair_count = 0
    for method, scene, pred, truth in iter_prediction_pairs(args.recon_root):
        evaluate_pair(rows, accumulators, method, scene, pred, truth, args.skip_existing)
        pair_count += 1
        print(f"{scene} {method}: evaluated {pred.name}")

    for (method, product, band_or_index), acc in sorted(accumulators.items()):
        row = {
            "section": "overall_continuous",
            "scene": "ALL",
            "method": method,
            "product": product,
            "band_or_index": band_or_index,
            "class_kind": "all",
            "class_name": "all",
            "truth_class": "",
            "pred_class": "",
        }
        row.update(acc.row())
        rows.append(row)

    csv_path = args.summary_root / "evaluation_summary.csv"
    md_path = args.summary_root / "evaluation_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"evaluated {truth_count} ground-truth rasters")
    print(f"evaluated {pair_count} prediction/truth pairs")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
