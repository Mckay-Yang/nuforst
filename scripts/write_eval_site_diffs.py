#!/usr/bin/env python3
"""Write signed prediction differences for evaluation sites.

Outputs one multi-band GeoTIFF per site and method:

    diff_id{id}_{method}.tif

The difference is prediction - ground truth in the original reflectance scale.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio


METHODS = ("nufrost", "zhu2015", "hants")


def site_dir_name(lon: str, lat: str) -> str:
    return f"{float(lon):.4f}_{float(lat):.4f}"


def find_truth(site_dir: Path, site_id: str) -> Path:
    truths = sorted(
        path for path in site_dir.glob("*.tif") if path.name.startswith("[ground_truth]_")
    )
    if truths:
        return truths[0]
    fallback = site_dir / f"{site_id}.tif"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"missing ground truth for site id={site_id}: {site_dir}")


def find_prediction(site_dir: Path, method: str) -> Path:
    prefix = f"[{method}]_"
    preds = sorted(
        path
        for path in site_dir.glob("*.tif")
        if path.name.startswith(prefix) and path.name.endswith("_prediction.tif")
    )
    if len(preds) != 1:
        raise FileNotFoundError(f"expected one {method} prediction in {site_dir}, found {len(preds)}")
    return preds[0]


def output_profile(reference: rasterio.DatasetReader) -> dict:
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def write_diff(prediction_path: Path, truth_path: Path, out_path: Path) -> None:
    with rasterio.open(prediction_path) as pred, rasterio.open(truth_path) as truth:
        if pred.count != truth.count or pred.width != truth.width or pred.height != truth.height:
            raise ValueError(
                f"shape mismatch: {prediction_path} has "
                f"{pred.count}x{pred.width}x{pred.height}, truth has "
                f"{truth.count}x{truth.width}x{truth.height}"
            )
        profile = output_profile(pred)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as out:
            for _, window in pred.block_windows(1):
                pred_arr = pred.read(window=window).astype("float32")
                truth_arr = truth.read(window=window).astype("float32")
                valid = np.isfinite(pred_arr) & np.isfinite(truth_arr)
                diff = np.full(pred_arr.shape, np.nan, dtype="float32")
                diff[valid] = pred_arr[valid] - truth_arr[valid]
                out.write(diff, window=window)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites-csv", type=Path, default=Path("../evaluation_sites_16.csv"))
    parser.add_argument(
        "--recon-root",
        type=Path,
        default=Path("data/output/full_scene_all_methods_20260622_final/sentinel-2_recon"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/full_scene_all_methods_20260622_final/diff_by_site_id"),
    )
    args = parser.parse_args()

    with args.sites_csv.open(newline="") as handle:
        sites = list(csv.DictReader(handle))

    written = 0
    for site in sites:
        site_id = site["id"]
        site_dir = args.recon_root / site_dir_name(site["lon"], site["lat"])
        truth_path = find_truth(site_dir, site_id)
        for method in METHODS:
            prediction_path = find_prediction(site_dir, method)
            out_path = args.output_dir / f"diff_id{site_id}_{method}.tif"
            write_diff(prediction_path, truth_path, out_path)
            written += 1
            print(f"wrote {out_path}")

    print(f"wrote {written} diff rasters")


if __name__ == "__main__":
    main()
