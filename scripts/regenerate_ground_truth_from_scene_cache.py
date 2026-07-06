#!/usr/bin/env python3
"""Regenerate a missing full-scene ground-truth GeoTIFF from scene cache.

The scene cache layout is written by the Rust GDAL crate as:

    cube.f32.bin: band_time_row_col, little-endian float32
    meta.json:    band timestamps, offsets, CRS, transform, shape

This script extracts one timestamp slice for every band and writes the standard
`[ground_truth]_...tif` stack used by the evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


def location_output_token(lon: float, lat: float) -> str:
    return f"lon{lon:.6f}_lat{lat:.6f}"


def build_ground_truth_output_path(
    output_root: Path,
    source_name: str,
    lon: float,
    lat: float,
    target_time: str,
) -> Path:
    loc_token = location_output_token(lon, lat)
    return (
        output_root
        / f"{source_name}_recon"
        / f"{lon:.4f}_{lat:.4f}"
        / f"[ground_truth]_{source_name}_{loc_token}_{target_time}.tif"
    )


def gdal_geotransform_to_affine(gt: list[float]) -> Affine:
    if len(gt) != 6:
        raise ValueError(f"expected 6 geotransform values, got {len(gt)}")
    return Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])


def read_target_from_cache(cache_dir: Path, target_time: str) -> tuple[np.ndarray, list[str], dict]:
    meta_path = cache_dir / "meta.json"
    cube_path = cache_dir / "cube.f32.bin"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    rows = int(meta["rows"])
    cols = int(meta["cols"])
    total_values = int(meta["total_values"])
    cube = np.memmap(cube_path, dtype="<f4", mode="r", shape=(total_values,))

    bands = list(meta["bands"])
    stack = np.empty((len(bands), rows, cols), dtype=np.float32)

    band_meta_by_name = {band_meta["name"]: band_meta for band_meta in meta["band_meta"]}
    for band_index, band in enumerate(bands):
        band_meta = band_meta_by_name[band]
        timestamps = band_meta["timestamps"]
        try:
            time_index = timestamps.index(target_time)
        except ValueError as exc:
            raise ValueError(f"target {target_time} not found for band {band}") from exc

        time_len = int(band_meta["time_len"])
        offset = int(band_meta["offset_values"])
        plane_size = rows * cols
        start = offset + time_index * plane_size
        end = start + plane_size
        if end > offset + time_len * plane_size:
            raise ValueError(f"computed slice for band {band} exceeds cache extent")
        stack[band_index, :, :] = np.asarray(cube[start:end]).reshape(rows, cols)

    return stack, bands, meta


def write_stack(path: Path, stack: np.ndarray, bands: list[str], meta: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")

    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = meta.get("nodata")
    profile = {
        "driver": "GTiff",
        "height": int(meta["rows"]),
        "width": int(meta["cols"]),
        "count": len(bands),
        "dtype": "float32",
        "transform": gdal_geotransform_to_affine(meta["geo_transform"]),
        "crs": meta.get("crs_wkt"),
        "interleave": "pixel",
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        for idx, band in enumerate(bands, start=1):
            dst.set_band_description(idx, band)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-name", default="sentinel-2")
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--target-time", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stack, bands, meta = read_target_from_cache(args.cache_dir, args.target_time)
    output_path = build_ground_truth_output_path(
        args.output_root,
        args.source_name,
        args.lon,
        args.lat,
        args.target_time,
    )
    write_stack(output_path, stack, bands, meta, args.overwrite)
    print(output_path)


if __name__ == "__main__":
    main()
