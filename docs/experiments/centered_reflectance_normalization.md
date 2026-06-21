# Centered Reflectance Normalization Experiment

Date: 2026-06-22

## Purpose

This experiment tested whether NUFROST should keep a fixed Sentinel-2 reflectance scale while removing only the per-pixel band offset. The motivation was that robust median/MAD normalization can make fitted parameters less comparable across pixels, while plain reflectance scaling can keep large band offsets that destabilize the time-series fit.

The tested normalization is:

```text
z_b(t_i) = (y_b(t_i) - median_i y_b(t_i)) / 10000
```

This keeps the Sentinel-2 reflectance amplitude scale fixed and avoids per-pixel MAD rescaling.

## Reproduction Config

The experiment config is saved as:

```text
config/nufrost_centered_reflectance_best_rmse.json
```

The final active `config/nufrost.json` uses the same parameter set:

```json
{
  "frequency_selection": "all",
  "ridge": 0.2,
  "freq_weight": 16.0,
  "multiband_shrinkage": 1.0,
  "normalization_mode": "centered_reflectance",
  "huber_iters": 3,
  "huber_delta": 0.18,
  "outlier_reject_iters": 2,
  "outlier_reject_max_fraction": 0.35,
  "lambda_step": 0.05,
  "lambda_high": 0.005
}
```

## Implementation

The Rust implementation adds a new NUFROST normalization mode:

```text
centered_reflectance
```

For each pixel and each band, NUFROST now supports three normalization modes:

- `robust`: subtract per-pixel band median and divide by per-pixel band MAD scale.
- `reflectance`: use raw Sentinel-2 reflectance scale, `y / 10000`.
- `centered_reflectance`: subtract per-pixel band median and divide by fixed `10000`.

The active method remains vector-valued NUFROST: shared vector NUFFT frequency support, date-level vector Huber weights, and multi-output ridge fitting.

## Full-Scene Validation Snapshot

The centered-reflectance experiment was run on the previous full-scene validation set before the Sentinel-2 imagery was replaced.

Experiment output root:

```text
data/output/experiments/centered_reflectance_all_20260621
```

Preview image:

```text
data/output/experiments/centered_reflectance_all_20260621/preview_rgb_contact_sheet.png
```

Overall weighted metrics across 14 scenes:

| Method | MAE | RMSE | Bias |
|---|---:|---:|---:|
| NUFROST centered reflectance | 92.998692 | 158.479820 | -4.807358 |
| NUFROST fixed reflectance | 98.453014 | 181.497597 | -12.563866 |
| HANTS | 101.924340 | 194.674613 | -24.138756 |
| Zhu2015 | 119.804580 | 217.618617 | 18.839283 |

Metrics excluding `115.8977_33.0074`:

| Method | MAE | RMSE | Bias |
|---|---:|---:|---:|
| NUFROST centered reflectance | 91.171520 | 152.038073 | -0.179857 |
| NUFROST fixed reflectance | 94.340060 | 156.393947 | -5.373205 |
| HANTS | 98.271026 | 191.422236 | - |
| Zhu2015 | 114.559564 | 215.065159 | - |

For `115.8977_33.0074`, centered reflectance reduced NUFROST error relative to plain reflectance scaling:

| Method | MAE | RMSE |
|---|---:|---:|
| NUFROST centered reflectance | 118.153972 | 229.465360 |
| NUFROST fixed reflectance | 155.077428 | 386.936946 |
| HANTS | - | 234.917050 |
| Zhu2015 | - | 250.136600 |

## Negative Prediction Check

Negative and extreme-value diagnostics were saved at:

```text
data/output/experiments/centered_reflectance_all_20260621/negative_extreme_stats.csv
```

Global negative ratios for centered reflectance:

| Band | Negative ratio |
|---|---:|
| B2 | 0.089379% |
| B3 | 0.109124% |
| B4 | 0.144496% |
| B8 | 0.011945% |
| B11 | 0.028461% |
| B12 | 0.119749% |

Compared with plain reflectance scaling, centered reflectance substantially reduced negative predictions in every band.

## Cache Note After Imagery Replacement

After the Sentinel-2 imagery was replaced under `data/sentinel-2`, old derived caches were removed:

```text
data/cache/local/vrts
data/cache/scenes/sentinel-2
data/cache/samples
```

The new imagery was not cached because the available GeoTIFF files did not have aligned band dimensions for any complete 6-band location. The cache should be rebuilt only after the imagery is re-exported with a consistent AOI and pixel grid across B2, B3, B4, B8, B11, and B12.

