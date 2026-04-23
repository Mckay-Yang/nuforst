---
name: full-scene-reconstruction-usage
description: Use when running or interpreting full-scene leave-one-time reconstruction outputs for NUFROST, HANTS, and Zhu2015 across data/sentinel-2 or data/hls, especially when you need to locate merged predictions, grand truth GeoTIFF, or the summary JSON manifest.
---

# Full-Scene Reconstruction Usage

## Overview

Use this when you need to run or explain the repository's full-scene leave-one-time reconstruction workflow.

For one location and one source, the workflow masks one shared timestamp across all selected bands, reconstructs that held-out scene with all three algorithms, and writes merged prediction rasters plus one merged ground-truth raster.

## When to Use

- You need one shared target timestamp across all bands for one location.
- You need outputs that are safe to combine into RGB or false-color composites.
- You need to explain the difference between merged prediction stacks, `grand_truth.tif`, and the summary JSON.

Do not use this skill for streaming evaluation experiments or point-sampling sweeps. Use the evaluation workflow instead.

## Quick Reference

- Entry point: `src.full_scene_reconstruction.reconstruct_full_scene_for_location(...)`
- Sources: `data/sentinel-2`, `data/hls`
- Algorithms: `nufrost`, `hants`, `zhu2015`
- Shared timestamp: one timestamp per location/source, reused across all bands and all methods
- Ground truth meaning: held-out observed scene at the masked timestamp, stacked in the same band order

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
- `merged_prediction_outputs[method]`: one multiband prediction GeoTIFF per method
- `ground_truth_output`: one multiband held-out observation GeoTIFF (`grand_truth.tif`)
- `summary_path`: JSON manifest for the run

At the moment, `target_time` is auto-selected by the workflow. It is not currently a required user-supplied argument for the full-scene entrypoint.

## Output Layout

For each method, the workflow writes merged files under `data/output/<method>/`.

Merged prediction stack example:

```text
data/output/nufrost/sentinel-2_lon94.260500_lat29.773300_2026-02-06T04-18-39__nufrost_prediction.tif
```

Merged ground truth stack example:

```text
data/output/sentinel-2_lon94.260500_lat29.773300_2026-02-06T04-18-39__grand_truth.tif
```

Summary JSON example:

```text
data/output/run_summaries/reconstruction_summary_sentinel-2_lon94.260500_lat29.773300_2026-02-06T04-18-39.json
```

## Ground Truth Interpretation

`grand_truth.tif` is not model output. It is the held-out observed scene at `target_time`.

Use it as a direct reference in QGIS against each method's merged prediction stack.

## Merged Stack Interpretation

Merged prediction stacks are one GeoTIFF per method per location/time.

- Each raster band corresponds to one spectral band.
- Band descriptions are written into the GeoTIFF itself.
- Band order also appears in `summary.json` under `bands`.

`grand_truth.tif` is one GeoTIFF per location/time and shares the same band order, transform, CRS, and shape checks as the merged prediction outputs.

## Common Mistakes

- Do not treat Zhu2015's old second output band as a full-scene QA product in this workflow.
- Do not expect per-band intermediate prediction files to persist after a run. They are deleted after merge.
- Do not guess band order from filenames alone when a merged stack is available. Read band descriptions or `summary.json`.

## Recommended Reading Pattern

When explaining a run to someone else, use this order:

1. Read the `reconstruction_summary_*.json` manifest for `target_time`, `bands`, and output paths.
2. Open the merged prediction stack for image interpretation.
3. Open `grand_truth.tif` and compare against each merged prediction stack in QGIS.
