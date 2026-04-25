---
name: nufrost-project
description: Use when working in the NUFROST repo and needing the codebase map, notebook or script entrypoints, exported Python interfaces, configuration precedence, or reconstruction and evaluation workflow guidance
---

# NUFROST Project Reference

## Overview

NUFROST is a notebook-first remote sensing time-series reconstruction repo. Treat it as four connected layers: data loading, reconstruction algorithms, evaluation workflows, and notebook/script entrypoints.

## When to Use

- Explaining the repo architecture or where a workflow lives
- Choosing between notebooks, Python APIs, and `scripts/*.py`
- Finding public entrypoints such as `reconstruct_nufrost`, `RSCube`, or full-scene helpers
- Tracing how GeoTIFF inputs become VRTs, NPZ caches, GeoTIFF outputs, CSVs, and summary JSON
- Debugging config precedence or output-shape differences between NUFROST, Zhu2015, and HANTS

Do not use this skill as a replacement for reading the target implementation file when exact internal behavior matters.

## Repository Map

```text
src/
  __init__.py                   # Public Python API re-exports
  data_loader.py                # TIFF/VRT loading, timestamp parsing, NPZ cache, streaming raster source
  nufrost.py                    # NUFROST reconstruction entrypoint + core algorithm
  zhu2015.py                    # Zhu2015 baseline reconstruction
  hants.py                      # HANTS baseline reconstruction
  evaluation.py                 # Random-point / gap evaluation helpers
  local_eval_workflow.py        # Notebook-free local evaluation orchestration
  full_scene_reconstruction/
    __init__.py                 # Full-scene public exports
    pipeline.py                 # Location discovery, band stacks, target timestamp selection, GeoTIFF writers
  logger.py                     # Async file logger for long-running workflows

config/
  config.yaml                   # Canonical defaults
  settings.py                   # Args dataclass, CLI parser, build_args()

scripts/
  run_local_evals.py            # Notebook-free local evaluation entrypoint
  run_full_scene_reconstruction.py
  run_small_window_full_scene.py

notebooks/
  local_evals.ipynb             # Main local evaluation workflow
  ...                           # Reconstruction, plotting, Earth Engine export notebooks
```

## Core Logic

### Three algorithms are independent

- `NUFROST`: proposed method. `NUFFT -> frequency selection -> optional parabolic refinement -> Huber-Ridge IRLS`
- `Zhu2015`: comparison baseline. Piecewise harmonic fitting with `Lasso`
- `HANTS`: comparison baseline. Harmonic fitting with iterative outlier rejection

Do not describe Zhu2015 or HANTS as sub-parts of NUFROST.

### Data flow

```text
GeoTIFF tiles
  -> find_image_chunks() / discover_location_band_stacks()
  -> VRT cache under <cache_dir>/vrts/
  -> one of two read paths:
     - RSCube.load() -> full cube + NPZ cache under <cache_dir>/npz/
     - TimeSeriesRasterSource -> streaming reads for evaluation / full-scene workflows
  -> reconstruction or evaluation
  -> outputs under data/output/
```

### Execution modes

- Notebook-first workflows: preferred for research and ad hoc experiments
- Python API: best for calling one reconstruction method programmatically
- `scripts/*.py`: best for repeatable local evaluation or full-scene batch runs

## Public Python Interfaces

### `src/__init__.py` exports

| Export | Purpose | Notes |
|---|---|---|
| `reconstruct_nufrost(image, target_time, output_path=None, **kwargs)` | Run NUFROST on one cube | Returns `np.ndarray` with shape `(H, W)` |
| `reconstruct_zhu2015(image, target_time, output_path=None, lasso_alpha=..., n_jobs=..., cache_dir=..., force_refresh=False)` | Run Zhu2015 on one cube | Returns shape `(2, H, W)` for prediction + QA |
| `reconstruct_hants(image, target_time, output_path=None, nof=..., sf=..., fet=..., dod=..., n_jobs=..., cache_dir=..., force_refresh=False)` | Run HANTS on one cube | Returns shape `(H, W)` |
| `RSCube(tif_path, cache_dir=None, force_refresh=False)` | Load TIFF/VRT inputs and cache them as NPZ | `load()` returns `cube`, `timestamps`, `band_names`, and metadata |
| `build_args(overrides=None)` | Merge config values | Precedence: Python overrides > CLI flags > YAML defaults > dataclass fallbacks |
| `Args` | Runtime config dataclass | Used across notebook and Python flows |

### `src/full_scene_reconstruction/__init__.py` exports

| Export | Purpose |
|---|---|
| `reconstruct_full_scene_for_location(...)` | Reconstruct all discovered bands for one `(lon, lat)` |
| `reconstruct_full_scene_for_all_locations(...)` | Batch all discovered coordinates for a source |
| `discover_available_locations(...)` | Find coordinates from filenames in `data/<source>/` |
| `discover_location_band_stacks(...)` | Build per-band TIFF/VRT stacks for one location |
| `choose_shared_target_timestamp(...)` | Pick a valid shared timestamp across selected bands |
| `write_run_summary(...)` | Persist summary JSON for a reconstruction run |

## Entry Points

### Start here by task

| Goal | Best entrypoint | Why |
|---|---|---|
| Explore or tune algorithms interactively | `notebooks/` | This repo is notebook-first |
| Run one reconstruction from Python | `src.reconstruct_nufrost`, `src.reconstruct_zhu2015`, `src.reconstruct_hants` | Lowest ceremony |
| Run local evaluation without a notebook | `python scripts/run_local_evals.py ...` | Wraps `run_local_evals_workflow()` |
| Reconstruct a whole location across bands | `python scripts/run_full_scene_reconstruction.py ...` | Uses full-scene pipeline |
| Run a cheap regression on a cropped scene | `python scripts/run_small_window_full_scene.py ... --window-size N` | Same pipeline, smaller window |

### Script quick reference

| Script | Main flags | Behavior |
|---|---|---|
| `scripts/run_local_evals.py` | `--source-name`, `--output-dir`, `--cache-dir`, `--max-images`, `--n-jobs`, `--run-ablation`, `--run-sparse`, `--run-gap`, `--run-repeatability` | Runs notebook-free evaluation sweeps and writes CSVs. `--source-name` is limited to `sentinel-2` or `hls` |
| `scripts/run_full_scene_reconstruction.py` | `--source-name`, `--lon`, `--lat`, `--all-coordinates`, `--output-root`, `--data-root`, `--cache-dir`, `--methods`, `--n-jobs`, `--force-refresh` | Runs full-scene reconstruction for one location or all locations. Defaults: `data/`, `data/output/`, `data/cache/local/` |
| `scripts/run_small_window_full_scene.py` | Same as full-scene script plus `--window-size` | Runs the full-scene pipeline on a cropped window |

### Typical commands

```bash
python scripts/run_local_evals.py --source-name sentinel-2 --max-images 2 --n-jobs -1
python scripts/run_full_scene_reconstruction.py --source-name sentinel-2 --lon 100.112 --lat 25.654 --methods nufrost hants zhu2015
python scripts/run_small_window_full_scene.py --source-name sentinel-2 --lon 100.112 --lat 25.654 --window-size 512
```

## Configuration Rules

- `build_args(overrides=...)` merges Python overrides on top of `config/config.yaml`
- CLI parsing uses YAML values as parser defaults
- If `target_time` is omitted, `build_args()` falls back to `start_time`
- Important defaults live in `config/config.yaml`; `Args` is the code-level fallback

Known parameter mismatch to remember when reasoning about old code or direct function calls:

- `ridge`
- `ignore_dc_hz`
- `num_peaks`

When using `build_args({})`, YAML wins. When calling low-level functions directly without passing overrides, the function signature or dataclass default may win instead.

## Outputs And Artifacts

- NUFROST and HANTS write single-band GeoTIFFs
- Zhu2015 writes a 2-band GeoTIFF: prediction + QA
- Full-scene scripts write outputs under `data/output/<source>_recon/<lon>_<lat>/`
- Full-scene scripts also write run summaries under `data/output/run_summaries/`
- Evaluation workflows append CSVs such as `sentinel-2_ablation_results.csv`, `*_sparse_sweep_results.csv`, `*_gap_sweep_results.csv`, and `*_repeatability_results.csv`

### Full-scene naming rules

- Per-method scene output: `[<method>]_<source>_lon<lon6>_lat<lat6>_<target-time>.tif`
- Ground truth output: `[ground_truth]_<source>_lon<lon6>_lat<lat6>_<target-time>.tif`
- Run summary JSON: `data/output/run_summaries/reconstruction_summary_<source>_lon<lon6>_lat<lat6>_<target-time>.json`

## One Good Example

Use this pattern when you want one programmatic reconstruction and do not need the notebook or full-scene pipeline:

```python
from src.data_loader import find_image_chunks
import src

image_paths = find_image_chunks(
    data_dir="data/hls",
    lon=91.2734,
    lat=29.7904,
    band="BLUE",
    cache_dir="data/cache/local",
)

recon = src.reconstruct_nufrost(
    image=image_paths,
    target_time="2023-06-15T00:00:00",
    output_path="data/output/example_nufrost.tif",
    cache_dir="data/cache/local",
    n_jobs=-1,
)
```

`find_image_chunks()` returns a `List[str]`, usually a one-element list containing the final VRT path. Do not treat it as a single string in prose or code review.

## Common Mistakes

- Assuming this repo is CLI-first. It is notebook-first.
- Treating Zhu2015 or HANTS as internal stages of NUFROST.
- Forgetting that Zhu2015 output has two bands.
- Assuming `pytest tests/` is a reliable unattended smoke test. It is not.
- Treating `find_image_chunks()` as a single path instead of a list.
- Forgetting that `RSCube.load()` caches full cubes, while evaluation and full-scene paths often use streaming reads.
- Forgetting `--lon` and `--lat` are required for `run_full_scene_reconstruction.py` unless `--all-coordinates` is set.

## Environment And Testing Notes

- Preferred environment file is `environment.yml` (`conda env create -f environment.yml`, env name `geo-science`)
- `requirements.txt` is only the minimal runtime set, not the full notebook stack
- There is no lint, formatter, typecheck, or CI config in the repo
- The checked-in test coverage is minimal and loader-focused

### Environment setup

```bash
conda env create -f environment.yml
conda activate geo-science
```

Use `pip install -r requirements.txt` only when the notebook and Earth Engine dependencies are not needed.
