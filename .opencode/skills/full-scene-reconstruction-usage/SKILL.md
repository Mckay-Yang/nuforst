---
name: full-scene-reconstruction-usage
description: Use when running or interpreting full-scene leave-one-time reconstruction outputs for NUFROST, HANTS, and Zhu2015 across data/sentinel-2 or data/hls, especially when you need to locate per-band predictions, residual QA rasters, merged multiband GeoTIFFs, or the summary JSON manifest.
---

# Full-Scene Reconstruction Usage

## Overview

Use this when you need to run or explain the repository's full-scene leave-one-time reconstruction workflow.

The core rule is simple: for one location and one source, the workflow masks one shared timestamp across all selected bands, reconstructs that held-out scene with all three algorithms, and writes both prediction rasters and residual QA rasters.

## When to Use

- You need one shared target timestamp across all bands for one location.
- You need outputs that are safe to combine into RGB or false-color composites.
- You need to explain the difference between per-band files, merged stacks, and the summary JSON.
- You need to interpret QA rasters correctly.

Do not use this skill for streaming evaluation experiments or point-sampling sweeps. Use the evaluation workflow instead.

## Quick Reference

- Entry point: `src.full_scene_reconstruction.reconstruct_full_scene_for_location(...)`
- Sources: `data/sentinel-2`, `data/hls`
- Algorithms: `nufrost`, `hants`, `zhu2015`
- Shared timestamp: one timestamp per location/source, reused across all bands and all methods
- QA meaning: `abs(prediction - held_out_truth)` at the masked timestamp for all three methods

## How To Run

Example:

```python
from src.full_scene_reconstruction import reconstruct_full_scene_for_location

result = reconstruct_full_scene_for_location(
    source_name="sentinel-2",
    lon=94.2605,
    lat=29.7733,
    n_jobs=4,
)
```

Key fields in `result`:

- `target_time`: the shared held-out timestamp
- `outputs[method][band]`: per-band prediction GeoTIFF
- `qa_outputs[method][band]`: per-band QA GeoTIFF
- `merged_prediction_outputs[method]`: one multiband prediction GeoTIFF per method
- `merged_qa_outputs[method]`: one multiband QA GeoTIFF per method
- `summary_path`: JSON manifest for the run

At the moment, `target_time` is auto-selected by the workflow. It is not currently a required user-supplied argument for the full-scene entrypoint.

## Output Layout

For each method, the workflow writes files under `data/output/<method>/`.

Per-band prediction example:

```text
data/output/nufrost/COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733_2026-02-06T04-18-39__nufrost.tif
```

Per-band QA example:

```text
data/output/nufrost/COPERNICUS_S2_HARMONIZED_B2_lon94.2605_lat29.7733_2026-02-06T04-18-39__nufrost_QA.tif
```

Merged prediction stack example:

```text
data/output/nufrost/sentinel-2_lon94.260500_lat29.773300_2026-02-06T04-18-39__nufrost_prediction.tif
```

Merged QA stack example:

```text
data/output/nufrost/sentinel-2_lon94.260500_lat29.773300_2026-02-06T04-18-39__nufrost_QA_stack.tif
```

Summary JSON example:

```text
data/output/run_summaries/reconstruction_summary_sentinel-2_lon94.260500_lat29.773300_2026-02-06T04-18-39.json
```

## QA Interpretation

QA is not a categorical confidence code in this workflow.

QA is the absolute residual at the held-out timestamp:

```text
QA = abs(prediction - held_out_truth)
```

That definition is the same for:

- `nufrost`
- `hants`
- `zhu2015`

This means smaller QA values are better.

Important: Zhu2015 has its own native internal QA output in the original algorithm, but the full-scene workflow does not export that as the main QA product. It exports residual QA so all three methods are directly comparable.

## Merged Stack Interpretation

Merged prediction stacks and merged QA stacks are one GeoTIFF per method per location/time.

- Each raster band corresponds to one spectral band.
- Band descriptions are written into the GeoTIFF itself.
- Band order also appears in `summary.json` under `bands`.

Use the merged prediction stack for:

- true color
- false color
- multispectral comparisons

Use the merged QA stack for:

- locating high-error regions
- comparing residual patterns between algorithms
- masking or highlighting uncertain areas in figures

## Common Mistakes

- Do not treat Zhu2015's old second output band as the final QA product for this workflow.
- Do not mix prediction bands and QA bands into one visualization stack.
- Do not assume QA is unitless confidence. It is residual magnitude in the same value scale as the reconstructed band.
- Do not guess band order from filenames alone when a merged stack is available. Read band descriptions or `summary.json`.

## Recommended Reading Pattern

When explaining a run to someone else, use this order:

1. Read the `reconstruction_summary_*.json` manifest for `target_time`, `bands`, and output paths.
2. Open the merged prediction stack for image interpretation.
3. Open the merged QA stack to inspect reconstruction error.
4. Fall back to per-band files only when debugging a specific band.
