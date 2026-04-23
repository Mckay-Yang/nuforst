# NUFROST

NUFROST is a notebook-first research codebase for reconstructing missing observations in optical satellite image time series and comparing multiple reconstruction methods on the same data.

The repository currently focuses on three algorithms:

- `NUFROST`: non-uniform FFT based frequency discovery with robust ridge fitting.
- `Zhu2015`: piecewise harmonic regression inspired by Zhu et al. (2015), with a QA output band.
- `HANTS`: harmonic analysis of time series with iterative outlier rejection.

The codebase is designed around exported multi-band GeoTIFF time-series cubes, especially NASA HLS and Sentinel-2 data prepared from Google Earth Engine.

## What This Project Can Do

- Reconstruct a target date from a time-series cube using NUFROST, Zhu2015, or HANTS.
- Auto-discover multi-file time-series chunks for a given `lon`, `lat`, and spectral band.
- Build a single time-stacked VRT from multiple TIFF parts and tiles.
- Cache GeoTIFF cubes to compressed `npz` files for faster repeated experiments.
- Evaluate reconstruction quality by masking valid observations and predicting them back.
- Plot per-pixel fitted curves across different algorithms.
- Generate summary plots and publication-style figures from evaluation CSV files.
- Export HLS and Sentinel-2 time-series cubes from Google Earth Engine with notebook workflows.

## End-to-End Workflow

1. Export time-series imagery from Google Earth Engine as multi-band GeoTIFFs.
2. Store the exported files under `data/hls/` or `data/sentinel-2/`.
3. Use `find_image_chunks()` to group all TIFF files for one location and one band into a single VRT.
4. Use `RSCube` to read the VRT or TIFF files and cache the result as `npz`.
5. Run one of the reconstruction algorithms on a target date.
6. Evaluate the reconstruction with notebook experiments in `notebooks/`.
7. Save images, CSVs, and figures under `data/output/`.

## Repository Layout

```text
nufrost/
|- config/
|  |- __init__.py
|  |- config.yaml
|  |- settings.py
|- data/
|  |- colab_cache/
|  |- hls/
|  |- local_cache/
|  |- output/
|  |- sentinel-2/
|  |- test_sample/
|- notebooks/
|  |- colab_evaluate.ipynb
|  |- colab_hants_launcher.ipynb
|  |- eeexport.ipynb
|  |- eeexport2.ipynb
|  |- local_evaluate_gap_index.ipynb
|  |- local_evaluate.ipynb
|  |- local_nufrost_launcher.ipynb
|  |- local_plot_curves.ipynb
|  |- nufrost_colab_launcher.ipynb
|  |- plot_evaluation_results.ipynb
|  |- sentinel_ieee_figure.ipynb
|  |- show_coord.ipynb
|  |- zhu2015_colab_launcher.ipynb
|- src/
|  |- __init__.py
|  |- data_loader.py
|  |- evaluation.py
|  |- hants.py
|  |- nufrost.py
|  |- zhu2015.py
|- tests/
|  |- test_data_loader.py
|- temp_debug/
|- environment.yml
|- requirements.txt
```

## Core Code Structure

### `src/`

- `src/__init__.py`: re-exports the main public entry points, including `reconstruct_nufrost`, `reconstruct_zhu2015`, `reconstruct_hants`, `RSCube`, and `build_args`.
- `src/data_loader.py`: data ingestion and caching layer. It resolves TIFF paths, builds VRTs, parses timestamps from band descriptions, and stores cached cubes as compressed `npz` files.
- `src/nufrost.py`: NUFROST algorithm implementation. It contains timestamp conversion helpers, frequency selection logic, robust ridge fitting, per-pixel prediction, curve prediction, and full-image reconstruction.
- `src/zhu2015.py`: Zhu2015 implementation. It builds harmonic regression models, splits time series into segments, predicts a target date, and outputs a QA band together with the reconstructed image.
- `src/hants.py`: HANTS implementation. It applies harmonic fitting with iterative outlier rejection and supports both single-pixel and full-image reconstruction.
- `src/evaluation.py`: evaluation utilities for random point masking, continuous temporal gap simulation, and metric calculation.

### `config/`

- `config/config.yaml`: default runtime settings such as frequency selection, ridge strength, number of peaks, cache path, and output path.
- `config/settings.py`: the `Args` dataclass plus `build_args()` and `build_arg_parser()`. This is the main configuration entry point used by notebooks and Python calls.

### `data/`

- `data/hls/`: expected location for HLS time-series GeoTIFF inputs.
- `data/sentinel-2/`: expected location for Sentinel-2 time-series GeoTIFF inputs.
- `data/cache/local/`: local `npz` cube cache and VRT cache.
- `data/cache/colab/`: Colab-specific cache location used by the Colab notebooks.
- `data/output/`: reconstructed GeoTIFFs, evaluation CSVs, and generated figures.
- `data/test_sample/`: small sample data used for loader testing and local checks.

### Other Top-Level Directories

- `tests/`: small amount of test coverage, currently focused on the loader.
- `temp_debug/`: scratch scripts and one-off debugging experiments. Useful for development history, but not part of the stable public workflow.
- `cache/` and `log/`: runtime artifacts and local experimentation outputs.

## Notebook Guide

The repository is primarily operated through notebooks. In practice, this is the easiest way to use the project.

### Reconstruction Notebooks

- `notebooks/local_nufrost_launcher.ipynb`: run NUFROST locally for one target location and band. If `IMAGE_NAMES` is empty, it auto-discovers the input cube from `TARGET_LON`, `TARGET_LAT`, and `TARGET_BAND`, then writes a reconstruction GeoTIFF.
- `notebooks/nufrost_colab_launcher.ipynb`: run NUFROST in Google Colab with Google Drive mounted as the working storage.
- `notebooks/zhu2015_colab_launcher.ipynb`: same Colab launcher pattern as above, but for the Zhu2015 reconstruction method.
- `notebooks/colab_hants_launcher.ipynb`: same Colab launcher pattern as above, but for HANTS.

### Evaluation Notebooks

- `notebooks/local_evaluate.ipynb`: local random-point evaluation for NUFROST, Zhu2015, and HANTS. It masks valid points across space and time for one configured cube and compares prediction quality.
- `notebooks/colab_evaluate.ipynb`: batch evaluation in Colab. It can scan many HLS files, build grouped cubes by location and band, resume from an existing CSV, and append new results incrementally.
- `notebooks/local_evaluate_gap_index.ipynb`: more specialized temporal-gap experiment. It varies the size of artificially removed time blocks and compares algorithm performance against a temporal gap index. This notebook is much heavier than the standard evaluation workflow.

### Curve and Results Visualization

- `notebooks/local_plot_curves.ipynb`: extract one pixel time series and plot the fitted curves from NUFROST, Zhu2015, and HANTS on the same figure. It can use the center pixel by default or a manually chosen row and column.
- `notebooks/plot_evaluation_results.ipynb`: read an evaluation CSV and create standard comparison plots for `RMSE`, `MAE`, `R`, and `OutlierRatio`.
- `notebooks/sentinel_ieee_figure.ipynb`: produce a more polished figure for Sentinel-2 evaluation results, intended for reports or paper-style presentation.

### Data Export and ROI Utilities

- `notebooks/eeexport.ipynb`: export Sentinel-2 time-series cubes from Earth Engine for a list of coordinates and bands.
- `notebooks/eeexport2.ipynb`: export NASA HLS v002 time-series cubes from Earth Engine, including cloud masking, Landsat and Sentinel merge, and chunking when the number of output bands is large.
- `notebooks/show_coord.ipynb`: visualize a target region of interest around a coordinate with `geemap`.

## Which Notebook Should I Start With?

- If you want one local NUFROST reconstruction, start with `notebooks/local_nufrost_launcher.ipynb`.
- If you want a local quality benchmark, start with `notebooks/local_evaluate.ipynb`.
- If you want to inspect a single pixel and understand curve shape differences, use `notebooks/local_plot_curves.ipynb`.
- If your workflow is Colab-first, use the corresponding `colab_*.ipynb` notebook.
- If you still need to export the source imagery, use `notebooks/eeexport2.ipynb` for HLS or `notebooks/eeexport.ipynb` for Sentinel-2.

## Installation

### Option 1: Conda environment

```bash
conda env create -f environment.yml
conda activate geo-science
```

This is the easiest way to get the Earth Engine and plotting notebooks working, because `environment.yml` includes a wider set of geospatial and notebook dependencies.

### Option 2: Minimal pip install

```bash
pip install -r requirements.txt
```

The minimal requirements are:

- `numpy`
- `pandas`
- `scikit-learn`
- `rasterio`
- `tqdm`
- `joblib`
- `PyYAML`
- `finufft`
- `scikit-image`

### Notes on system packages

- `rasterio` and GDAL compatibility matters. In Colab, the notebooks explicitly install `gdal-bin` before `pip install -r requirements.txt`.
- Earth Engine notebooks need packages such as `earthengine-api` and `geemap`, which are available in `environment.yml` but not in `requirements.txt`.

## Configuration

Most runtime parameters are controlled through `config/config.yaml` and loaded with `build_args()`.

Important settings include:

- `cache_dir`: where `npz` caches and VRT caches are stored.
- `target_time`: the reconstruction timestamp.
- `modes`, `eps`, `num_peaks`, `power_cum`: NUFROST spectral settings.
- `frequency_selection`, `preferred_periods_days`, `preferred_top_k`, `spectral_top_k`: NUFROST frequency selection strategy.
- `ridge`, `freq_weight`, `huber_iters`, `huber_delta`: NUFROST fitting regularization and robustness settings.
- `min_obs`: minimum number of valid observations needed per pixel.
- `n_jobs`: parallelism for reconstruction and evaluation.
- `output_path`: where reconstructed GeoTIFFs are written.

The notebooks often do the following:

```python
from config import build_args

args = build_args({})
args.image = image_paths
args.cache_dir = "data/cache/local"
args.n_jobs = -1
```

## Practical Usage Examples

### Reconstruct one scene with NUFROST from Python

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

To switch methods, replace `reconstruct_nufrost` with `reconstruct_zhu2015` or `reconstruct_hants`.

### Run the standard evaluation pipeline from Python

```python
from config import build_args
from src.data_loader import find_image_chunks
import src.evaluation as evaluation

image_paths = find_image_chunks(
    data_dir="data/hls",
    lon=91.2734,
    lat=29.7904,
    band="BLUE",
    cache_dir="data/cache/local",
)

args = build_args({})
args.image = image_paths
args.cache_dir = "data/cache/local"
args.n_jobs = -1

df = evaluation.evaluate_algorithms(
    image_path=args.image,
    args=args,
    num_points=40000,
)
```

## Data and File Naming Conventions

Auto-discovery in `find_image_chunks()` expects filenames that encode the spectral band and coordinate in the filename, for example:

```text
NASA_HLS_v002_BLUE_lon91.2734_lat29.7904_part1-0000000000-0000000000.tif
```

Important conventions:

- `BLUE`, `GREEN`, `RED`, `B2`, `B3`, and similar tokens identify the spectral band.
- `lon..._lat...` identifies the target location.
- `partN` indicates temporal chunking across multiple exported files.
- A suffix like `-0000000000-0000000000` is treated as a spatial tile suffix and is mosaiced inside a part before time stacking.

`find_image_chunks()` returns a list containing the final VRT path for one location and one band. The list wrapper is kept so that the rest of the code can continue to treat the result as `image_paths`.

## Outputs

- `NUFROST` reconstruction writes a single-band GeoTIFF.
- `HANTS` reconstruction writes a single-band GeoTIFF.
- `Zhu2015` reconstruction writes a two-band GeoTIFF:
  `band 1 = predicted reflectance`, `band 2 = QA`.
- Evaluation notebooks typically write CSV summaries to `data/output/`.
- Plotting notebooks write figures to `data/output/figures/` or display them inline.

## Important Implementation Notes

- The repository is notebook-first. There is no dedicated production CLI yet.
- `RSCube` caches loaded cubes as compressed `npz` files to avoid repeatedly reading large GeoTIFF stacks.
- The current `RSCube._read_tif()` implementation reads only the top-left `512 x 512` window of each source cube for memory safety. This is important if you expect full-scene reconstruction.
- Timestamp parsing relies on TIFF band descriptions. If band descriptions are missing or malformed, timestamp parsing may fall back to generic names and degrade time handling.
- If source TIFF files change, move, or are regenerated, rebuild caches or use `force_refresh=True`.
- The Colab notebooks assume the project lives in Google Drive and adjust paths accordingly.

## Evaluation Methods in This Repository

There are two main evaluation styles in the codebase:

- `evaluate_algorithms()` in `src/evaluation.py`: randomly masks valid spatiotemporal points and predicts them back. This is the standard benchmark used by `local_evaluate.ipynb` and `colab_evaluate.ipynb`.
- `evaluate_timeseries_comprehensive()` in `src/evaluation.py`: simulates a continuous temporal gap for selected pixels and evaluates predictions across that missing interval.

The main reported metrics are:

- `RMSE`
- `MAE`
- `R`
- `OutlierRatio`
- `SSIM` for image-style comparisons when applicable

## Testing

Test coverage is currently minimal.

- `tests/test_data_loader.py` is a loader smoke test built around the sample data in `data/test_sample/`.
- The repository does not currently have broad automated coverage for all notebooks and algorithms.
- `temp_debug/` contains many one-off validation and debugging scripts that are useful during development, but they are not a replacement for a formal test suite.

## Current Limitations

- The main user interface is still a set of notebooks rather than a polished package or CLI.
- Some workflows are dataset-specific and depend on filename conventions.
- Earth Engine export notebooks assume manual project credentials and storage setup.
- Current tests are not yet comprehensive enough for production-style confidence.

## Recommended Reading Order

If you are new to the repository, this order works well:

1. Read `src/data_loader.py` to understand how TIFF files become VRTs and cached cubes.
2. Read `src/nufrost.py`, `src/zhu2015.py`, and `src/hants.py` to understand the three reconstruction methods.
3. Open `notebooks/local_nufrost_launcher.ipynb` to run a first local NUFROST reconstruction.
4. Open `notebooks/local_plot_curves.ipynb` to compare the fitted behavior of the three methods on one pixel.
5. Open `notebooks/local_evaluate.ipynb` or `notebooks/colab_evaluate.ipynb` to benchmark the methods.

## Summary

This repository is best understood as a practical research workspace for satellite time-series reconstruction. The code in `src/` contains the reusable core logic, while the notebooks are the real operational entry points for exporting data, reconstructing images, evaluating quality, and preparing figures.
