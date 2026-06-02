"""
Python golden contract fixture for Rust parity testing.

Runs reconstruct_full_scene_for_location() on a small sentinel-2 location
and records ALL expected outputs (paths, shapes, band info, NaN ratios,
value ranges, summary JSON fields) into full_scene_contract.json.

This fixture IS the golden contract — Rust outputs must match path naming
and multiband structure.
"""

import json
import sys
from pathlib import Path

import rasterio
import numpy as np

# Ensure the src directory is on sys.path
WORKTREE = Path(__file__).resolve().parents[3]
SRC_DIR = WORKTREE / "src"
OUTPUT_ROOT = WORKTREE / "data" / "output"
CACHE_DIR = WORKTREE / "data" / "cache" / "local"

sys.path.insert(0, str(WORKTREE))


def _read_raster_metadata(tif_path: Path) -> dict:
    """Extract metadata and per-band stats from a GeoTIFF."""
    info: dict = {"path": str(tif_path), "exists": tif_path.exists()}
    if not tif_path.exists():
        info["error"] = "file not found"
        return info

    with rasterio.open(tif_path) as src:
        info["shape"] = list(src.shape)  # (bands, H, W) or (H, W)
        info["count"] = src.count
        info["crs"] = str(src.crs) if src.crs else None
        info["transform"] = list(src.transform) if src.transform else None
        info["dtype"] = str(src.dtypes[0])
        info["nodata"] = src.nodata

        band_info = {}
        for b in range(1, src.count + 1):
            arr = src.read(b)
            band_name = src.descriptions[b - 1] or f"band_{b}"
            nan_mask = np.isnan(arr)
            nan_count = int(np.sum(nan_mask))
            total = int(arr.size)
            nan_ratio = nan_count / max(total, 1)
            finite = arr[~nan_mask]
            if len(finite) > 0:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                vmean = float(np.mean(finite))
            else:
                vmin = vmax = vmean = None
            band_info[band_name] = {
                "index": b,
                "description": band_name,
                "nan_ratio": nan_ratio,
                "nan_count": nan_count,
                "total_pixels": total,
                "value_range": [vmin, vmax],
                "mean": vmean,
            }
        info["bands"] = band_info
        info["band_order"] = [src.descriptions[b - 1] or f"band_{b}" for b in range(1, src.count + 1)]
    return info


def generate_contract():
    """Run the reconstruction pipeline and serialize the golden contract."""
    from src.full_scene_reconstruction import reconstruct_full_scene_for_location

    print("=" * 60)
    print("Running reconstruct_full_scene_for_location() ...")
    print(f"  Location: lon=104.2595 lat=31.2170")
    print(f"  Methods: nufrost, hants, zhu2015")
    print("=" * 60)

    payload = reconstruct_full_scene_for_location(
        source_name="sentinel-2",
        lon=104.2595,
        lat=31.2170,
        methods=["nufrost", "hants", "zhu2015"],
        window_size=None,
        output_root=OUTPUT_ROOT,
        cache_dir=CACHE_DIR,
    )

    print("\nReconstruction complete.")
    print(f"  target_time: {payload['target_time']}")
    print(f"  bands: {payload['bands']}")

    # Build contract from payload
    contract: dict = {
        "contract_version": "1.0",
        "description": "Golden contract for Rust parity — NUFROST / HANTS / Zhu2015 on sentinel-2 lon=104.2595 lat=31.2170",
        "source": payload["source"],
        "lon": payload["lon"],
        "lat": payload["lat"],
        "target_time": payload["target_time"],
        "methods": payload["methods"],
        "ordered_bands": payload["bands"],
        "window_size": payload.get("window_size"),
        "min_valid_ratio": payload["min_valid_ratio"],
        "late_fraction": payload["late_fraction"],
        "mask_indices": payload["mask_indices"],
        "counts_before": payload["counts_before"],
        "counts_after": payload["counts_after"],
        "completeness": payload["completeness"],
        "timing_seconds": {m: dict(payload["timing_seconds"].get(m, {})) for m in payload["methods"]},
        # Paths
        "merged_prediction_outputs": payload["merged_prediction_outputs"],
        "ground_truth_output": payload["ground_truth_output"],
        "summary_path": payload["summary_path"],
        # Source files used
        "source_files": {k: list(v) for k, v in payload["source_files"].items()},
    }

    # Include optional params if present
    for key in ("frequency_selection", "spectral_top_k", "preferred_top_k", "num_peaks"):
        if key in payload:
            contract[key] = payload[key]

    # Inspect each output GeoTIFF
    print("\nInspecting output GeoTIFFs ...")
    contract["ground_truth"] = _read_raster_metadata(Path(payload["ground_truth_output"]))

    contract["predictions"] = {}
    for method_name in payload["methods"]:
        merged_path = Path(payload["merged_prediction_outputs"][method_name])
        print(f"  [{method_name}] {merged_path.name}")
        contract["predictions"][method_name] = _read_raster_metadata(merged_path)

    # Read summary JSON
    print(f"\nReading summary JSON: {payload['summary_path']}")
    summary_text = Path(payload["summary_path"]).read_text(encoding="utf-8")
    contract["summary_json"] = json.loads(summary_text)

    # Write contract
    contract_path = Path(__file__).parent / "full_scene_contract.json"
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nContract written to: {contract_path}")
    print(f"Contract size: {contract_path.stat().st_size:,} bytes")

    # Verification summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  target_time:        {contract['target_time']}")
    print(f"  ordered_bands:      {contract['ordered_bands']}")
    print(f"  mask_indices:       {contract['mask_indices']}")
    print(f"  counts_before:      {contract['counts_before']}")
    print(f"  counts_after:       {contract['counts_after']}")
    print(f"  completeness:       {contract['completeness']}")
    print(f"  nufrost output:     {contract['merged_prediction_outputs'].get('nufrost', 'N/A')}")
    print(f"  hants output:       {contract['merged_prediction_outputs'].get('hants', 'N/A')}")
    print(f"  zhu2015 output:     {contract['merged_prediction_outputs'].get('zhu2015', 'N/A')}")
    print(f"  ground truth:       {contract['ground_truth_output']}")
    print(f"  summary path:       {contract['summary_path']}")
    print(f"  gt shape:           {contract['ground_truth']['shape']}")
    print(f"  gt bands:           {contract['ground_truth']['band_order']}")
    for method in contract["methods"]:
        pred = contract["predictions"][method]
        print(f"  [{method}] shape:   {pred['shape']}")
        print(f"  [{method}] bands:   {pred['band_order']}")
        for bn, bi in pred["bands"].items():
            print(f"    {bn}: nan={bi['nan_ratio']:.4f} range={bi['value_range']}")
    print("=" * 60)

    return contract


if __name__ == "__main__":
    generate_contract()
