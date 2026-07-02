#!/usr/bin/env python3
"""Summarize per-site reconstruction and spectral-index errors.

Outputs:
- reflectance_site_method_metrics.csv/md
- spectral_index_site_method_metrics.csv/md

Reflectance metrics are reported in reflectance units by dividing Sentinel-2
scaled values by 10000 before error accumulation.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


METHODS = ("nufrost", "zhu2015", "hants")
METHOD_LABELS = {"nufrost": "NUFROST", "zhu2015": "Zhu2015", "hants": "HANTS"}
INDEX_NAMES = ("NDVI", "EVI", "NDWI", "NDSI", "NDMI", "NBR")
BAND_NAMES = ("B2", "B3", "B4", "B8", "B11", "B12")
SCALE = 10000.0


@dataclass
class Site:
    site_id: str
    lon: str
    lat: str

    @property
    def scene_name(self) -> str:
        return f"{float(self.lon):.4f}_{float(self.lat):.4f}"


def read_sites(path: Path) -> list[Site]:
    with path.open(newline="") as f:
        return [Site(row["id"], row["lon"], row["lat"]) for row in csv.DictReader(f)]


def method_from_name(path: Path) -> str:
    return path.name[1 : path.name.index("]")]


def find_truth(scene_dir: Path) -> Path | None:
    truths = sorted(scene_dir.glob("[[]ground_truth[]]_*.tif"))
    return truths[0] if truths else None


def find_prediction(scene_dir: Path, method: str) -> Path | None:
    preds = sorted(scene_dir.glob(f"[[]{method}[]]_*_prediction.tif"))
    return preds[0] if preds else None


def update_error_sums(diff: np.ndarray, sums: dict[str, float]) -> None:
    finite = diff[np.isfinite(diff)].astype("float64", copy=False)
    if finite.size == 0:
        return
    sums["n"] += int(finite.size)
    sums["sum"] += float(finite.sum())
    sums["abs"] += float(np.abs(finite).sum())
    sums["sq"] += float(np.square(finite).sum())


def finalize_sums(sums: dict[str, float]) -> dict[str, str]:
    n = int(sums["n"])
    if n == 0:
        return {"n": "0", "mse": "", "rmse": "", "mae": "", "bias": ""}
    mse = sums["sq"] / n
    return {
        "n": str(n),
        "mse": f"{mse:.8f}",
        "rmse": f"{math.sqrt(mse):.8f}",
        "mae": f"{sums['abs'] / n:.8f}",
        "bias": f"{sums['sum'] / n:.8f}",
    }


def reflectance_metrics(truth_path: Path, pred_path: Path) -> dict[str, str]:
    sums = {"n": 0.0, "sum": 0.0, "abs": 0.0, "sq": 0.0}
    with rasterio.open(truth_path) as truth, rasterio.open(pred_path) as pred:
        if truth.count != pred.count or truth.width != pred.width or truth.height != pred.height:
            raise ValueError(f"shape mismatch: {truth_path} vs {pred_path}")
        for band_idx in range(1, truth.count + 1):
            t = truth.read(band_idx).astype("float32", copy=False) / SCALE
            p = pred.read(band_idx).astype("float32", copy=False) / SCALE
            diff = p - t
            diff[~np.isfinite(t) | ~np.isfinite(p)] = np.nan
            update_error_sums(diff, sums)
    return finalize_sums(sums)


def band_index(ds: rasterio.DatasetReader, name: str) -> int:
    for idx, desc in enumerate(ds.descriptions, start=1):
        if desc == name:
            return idx
    fallback = BAND_NAMES.index(name) + 1
    if fallback <= ds.count:
        return fallback
    raise ValueError(f"{ds.name} missing band {name}")


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.full(a.shape, np.nan, dtype="float32")
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (np.abs(denom) > 1.0e-6)
    out[valid] = (a[valid] - b[valid]) / denom[valid]
    return out


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
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


def compute_indices(path: Path) -> dict[str, np.ndarray]:
    with rasterio.open(path) as ds:
        blue = ds.read(band_index(ds, "B2")).astype("float32", copy=False) / SCALE
        green = ds.read(band_index(ds, "B3")).astype("float32", copy=False) / SCALE
        red = ds.read(band_index(ds, "B4")).astype("float32", copy=False) / SCALE
        nir = ds.read(band_index(ds, "B8")).astype("float32", copy=False) / SCALE
        swir1 = ds.read(band_index(ds, "B11")).astype("float32", copy=False) / SCALE
        swir2 = ds.read(band_index(ds, "B12")).astype("float32", copy=False) / SCALE
    values = {
        "NDVI": normalized_difference(nir, red),
        "EVI": evi(nir, red, blue),
        "NDWI": normalized_difference(green, nir),
        "NDSI": normalized_difference(green, swir1),
        "NDMI": normalized_difference(nir, swir1),
        "NBR": normalized_difference(nir, swir2),
    }
    for index_name, arr in values.items():
        arr[(arr < -1.0) | (arr > 1.0)] = np.nan
    return values


def index_metric_rows(site: Site, truth_path: Path, pred_path: Path, method: str) -> list[dict[str, str]]:
    truth_indices = compute_indices(truth_path)
    pred_indices = compute_indices(pred_path)
    rows: list[dict[str, str]] = []
    for index_name in INDEX_NAMES:
        sums = {"n": 0.0, "sum": 0.0, "abs": 0.0, "sq": 0.0}
        diff = pred_indices[index_name] - truth_indices[index_name]
        diff[~np.isfinite(pred_indices[index_name]) | ~np.isfinite(truth_indices[index_name])] = np.nan
        update_error_sums(diff, sums)
        metric = finalize_sums(sums)
        rows.append(
            {
                "site_id": site.site_id,
                "scene": site.scene_name,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "index": index_name,
                **metric,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, str]], columns: list[str], labels: list[str]) -> None:
    with path.open("w") as f:
        f.write("| " + " | ".join(labels) + " |\n")
        f.write("|" + "|".join(["---"] * len(labels)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row.get(col, "") for col in columns) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites-csv", type=Path, required=True)
    parser.add_argument("--recon-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sites = read_sites(args.sites_csv)
    reflectance_rows: list[dict[str, str]] = []
    index_rows: list[dict[str, str]] = []
    skipped: list[str] = []

    for site in sites:
        scene_dir = args.recon_root / site.scene_name
        truth = find_truth(scene_dir)
        if truth is None:
            skipped.append(f"{site.site_id},{site.scene_name},missing_truth")
            continue
        for method in METHODS:
            pred = find_prediction(scene_dir, method)
            if pred is None:
                skipped.append(f"{site.site_id},{site.scene_name},missing_{method}")
                continue
            refl = reflectance_metrics(truth, pred)
            reflectance_rows.append(
                {
                    "site_id": site.site_id,
                    "scene": site.scene_name,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    **refl,
                }
            )
            index_rows.extend(index_metric_rows(site, truth, pred, method))
            print(f"{site.site_id} {site.scene_name} {method}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    refl_cols = ["site_id", "scene", "method", "method_label", "n", "mse", "rmse", "mae", "bias"]
    idx_cols = ["site_id", "scene", "method", "method_label", "index", "n", "mse", "rmse", "mae", "bias"]
    write_csv(args.output_dir / "reflectance_site_method_metrics.csv", reflectance_rows, refl_cols)
    write_csv(args.output_dir / "spectral_index_site_method_metrics.csv", index_rows, idx_cols)
    write_markdown_table(
        args.output_dir / "reflectance_site_method_metrics.md",
        reflectance_rows,
        ["site_id", "scene", "method_label", "mse", "rmse", "mae", "bias"],
        ["样区编号", "样区", "方法", "MSE", "RMSE", "MAE", "Bias"],
    )
    write_markdown_table(
        args.output_dir / "spectral_index_site_method_metrics.md",
        index_rows,
        ["site_id", "scene", "method_label", "index", "mse", "rmse", "mae", "bias"],
        ["样区编号", "样区", "方法", "指数", "MSE", "RMSE", "MAE", "Bias"],
    )
    if skipped:
        (args.output_dir / "skipped_sites.txt").write_text("\n".join(skipped) + "\n")
    print(f"Reflectance rows: {len(reflectance_rows)}")
    print(f"Index rows: {len(index_rows)}")
    if skipped:
        print("Skipped:")
        print("\n".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
