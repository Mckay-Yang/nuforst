#!/usr/bin/env python3
"""Compute spectral index rasters for prediction GeoTIFFs.

The script expects Sentinel-2 prediction rasters with six bands in the project
order B2, B3, B4, B8, B11, B12. It writes one single-band GeoTIFF per index and
keeps a CSV manifest of all generated products.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import rasterio


BANDS = {
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 4,
    "B11": 5,
    "B12": 6,
}

INDEX_NAMES = ("NDVI", "EVI", "NDWI", "NDSI", "NDMI", "NBR")
INDEX_RANGES = {
    "NDVI": (-1.0, 1.0),
    "EVI": (-1.0, 1.0),
    "NDWI": (-1.0, 1.0),
    "NDSI": (-1.0, 1.0),
    "NDMI": (-1.0, 1.0),
    "NBR": (-1.0, 1.0),
}


def parse_method(path: Path) -> str:
    match = re.match(r"^\[([^\]]+)\]_", path.name)
    if not match:
        raise ValueError(f"cannot parse method name from {path}")
    return match.group(1)


def prediction_stem(path: Path, method: str) -> str:
    stem = path.name.removeprefix(f"[{method}]_")
    return stem.removesuffix("_prediction.tif")


def band_index(ds: rasterio.DatasetReader, band_name: str) -> int:
    for idx, desc in enumerate(ds.descriptions, start=1):
        if desc == band_name:
            return idx
    fallback = BANDS[band_name]
    if fallback <= ds.count:
        return fallback
    raise ValueError(f"{ds.name} has no {band_name} band")


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.full(a.shape, np.nan, dtype="float32")
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (np.abs(denom) > 1.0e-6)
    out[valid] = (a[valid] - b[valid]) / denom[valid]
    return out


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    # Inputs are Sentinel-2 reflectance scaled by 10000, so the EVI "+1" term is
    # represented by +10000 in the same scale.
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


def clamp_index(index_name: str, values: np.ndarray) -> np.ndarray:
    low, high = INDEX_RANGES[index_name]
    out = values.astype("float32", copy=True)
    valid = np.isfinite(out) & (out >= low) & (out <= high)
    out[~valid] = np.nan
    return out


def compute_indices(ds: rasterio.DatasetReader) -> dict[str, np.ndarray]:
    blue = ds.read(band_index(ds, "B2")).astype("float32", copy=False)
    green = ds.read(band_index(ds, "B3")).astype("float32", copy=False)
    red = ds.read(band_index(ds, "B4")).astype("float32", copy=False)
    nir = ds.read(band_index(ds, "B8")).astype("float32", copy=False)
    swir1 = ds.read(band_index(ds, "B11")).astype("float32", copy=False)
    swir2 = ds.read(band_index(ds, "B12")).astype("float32", copy=False)

    indices = {
        "NDVI": normalized_difference(nir, red),
        "EVI": evi(nir, red, blue),
        "NDWI": normalized_difference(green, nir),
        "NDSI": normalized_difference(green, swir1),
        "NDMI": normalized_difference(nir, swir1),
        "NBR": normalized_difference(nir, swir2),
    }
    return {name: clamp_index(name, values) for name, values in indices.items()}


def output_path(output_scene_dir: Path, source_path: Path, index_name: str, method: str) -> Path:
    stem = prediction_stem(source_path, method)
    lower = index_name.lower()
    return output_scene_dir / f"{index_name}_[{method}]_{stem}_{lower}.tif"


def single_band_profile(ds: rasterio.DatasetReader) -> dict:
    profile = ds.profile.copy()
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
    return profile


def write_indices(source_path: Path, output_scene_dir: Path, overwrite: bool) -> list[dict[str, str]]:
    method = parse_method(source_path)
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    with rasterio.open(source_path) as ds:
        indices = compute_indices(ds)
        profile = single_band_profile(ds)
        for index_name, values in indices.items():
            out_path = output_path(output_scene_dir, source_path, index_name, method)
            if out_path.exists() and not overwrite:
                status = "exists"
            else:
                with rasterio.open(out_path, "w", **profile) as out_ds:
                    out_ds.write(values, 1)
                    out_ds.set_band_description(1, index_name)
                status = "written"
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "scene": source_path.parent.name,
                    "method": method,
                    "index": index_name,
                    "source": str(source_path),
                    "output": str(out_path),
                    "status": status,
                    "valid_pixels": str(int(finite.size)),
                    "mean": "" if finite.size == 0 else f"{float(finite.mean()):.8f}",
                    "std": "" if finite.size == 0 else f"{float(finite.std()):.8f}",
                }
            )
    return rows


def iter_predictions(input_root: Path):
    yield from sorted(input_root.glob("*/*_prediction.tif"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = list(iter_predictions(args.input_root))
    if not predictions:
        raise SystemExit(f"no prediction rasters found under {args.input_root}")

    all_rows: list[dict[str, str]] = []
    for idx, prediction in enumerate(predictions, start=1):
        scene_dir = args.output_root / prediction.parent.name
        print(f"[{idx:03d}/{len(predictions):03d}] {prediction.parent.name} {prediction.name}", flush=True)
        all_rows.extend(write_indices(prediction, scene_dir, args.overwrite))

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "spectral_indices_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("scene", "method", "index", "source", "output", "status", "valid_pixels", "mean", "std"),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote manifest: {manifest}")
    print(f"Products: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
